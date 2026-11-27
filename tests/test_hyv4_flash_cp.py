import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import unittest
from unittest.mock import patch

_ROOT=Path(__file__).resolve().parents[1]


def load_policy():
    spec=importlib.util.spec_from_file_location("flash_cp_test",_ROOT/"hcu_sglang/models/hyv4_flash/cp.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


class FlashCPTests(unittest.TestCase):
    def test_indexer_uses_full_context_then_restores_rank_rows(self):
        mod=load_policy();local_x=object();local_q=object();full_x=object();full_q=object();full_indices=object();expected=object()
        batch=SimpleNamespace(positions=object())
        def forward(x,q,positions,b,layer_id,return_indices):
            self.assertIs(x,full_x);self.assertIs(q,full_q);self.assertIs(positions,batch.positions)
            self.assertTrue(b.hy4f_cp_full_indexer)
            return full_indices
        with patch.object(mod,"gather",side_effect=[full_x,full_q]), patch.object(mod,"shard",return_value=expected) as shard:
            self.assertIs(mod.run_indexer(forward,local_x,local_q,None,batch,3,True),expected)
            shard.assert_called_once_with(full_indices,batch)
        self.assertFalse(batch.hy4f_cp_full_indexer)
        def fail(*args):raise RuntimeError("kernel failure")
        with patch.object(mod,"gather",return_value=object()), self.assertRaisesRegex(RuntimeError,"kernel failure"):
            mod.run_indexer(fail,local_x,local_q,None,batch,3,True)
        self.assertFalse(batch.hy4f_cp_full_indexer)

    def test_full_metadata_uses_original_request_page_table(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is required")
        mod=load_policy();name="sglang.srt.model_executor.forward_context";stub=ModuleType(name)
        table=torch.arange(3*192).reshape(3,192);stub.get_req_to_token_pool=lambda:SimpleNamespace(req_to_token=table)
        batch=SimpleNamespace(req_pool_indices=torch.tensor([2,0]),positions=torch.tensor([0,1,3,4]))
        previous=SimpleNamespace(attn_metadata=object(),topk_transform_method=object())
        with patch.dict(sys.modules,{name:stub}):meta=mod.full_indexer_metadata(batch,previous)
        torch.testing.assert_close(meta.get_page_table_1(),table[[2,0]])
        torch.testing.assert_close(meta.get_page_table_64(),table[[2,0]][:,::64]//64)
        torch.testing.assert_close(meta.get_seqlens_expanded(),batch.positions+1)

    def test_deepep_combined_routed_output_is_not_reduced_twice(self):
        tree=ast.parse((_ROOT/"hcu_sglang/models/hyv4_flash/model.py").read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="HYV4FlashMoE")
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="_forward_full")
        fn.decorator_list=[]
        backend=SimpleNamespace(is_deepep=lambda:True)
        scope={"dp_runtime":SimpleNamespace(active=lambda:False),"cp_runtime":SimpleNamespace(active=lambda batch:False),"get_moe_a2a_backend":lambda:backend,"_maybe_all_reduce":lambda x:x*8}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),"moe","exec"),scope)
        obj=SimpleNamespace(gate=lambda x:0,topk=lambda x,y:None,experts=lambda x,y:3.0,
                            routed_latent_down=None,routed_latent_up=None,shared_experts=lambda x:2.0,combine_output_norm=None)
        self.assertEqual(scope["_forward_full"](obj,SimpleNamespace(shape=(1,2))),19.0)
        backend.is_deepep=lambda:False
        self.assertEqual(scope["_forward_full"](obj,SimpleNamespace(shape=(1,2))),40.0)

        try:
            import torch
        except ImportError:
            self.skipTest("torch is required")
        full=torch.arange(16,dtype=torch.float32).reshape(8,2)
        local=full[1::2].contiguous()
        seen=[]
        def experts(x, topk):
            self.assertIs(x,local)
            seen.append("routed")
            return x*3
        def shared(x):
            self.assertIs(x,full)
            seen.append("shared")
            return x*2
        scope["cp_runtime"]=SimpleNamespace(active=lambda batch:True,gather=lambda x,b:full,shard=lambda x,b:x[1::2])
        obj.experts=experts;obj.shared_experts=shared
        torch.testing.assert_close(scope["_forward_full"](obj,local,object()),local*19)
        self.assertEqual(seen,["routed","shared"])

        class CTExperts:
            quant_method=SimpleNamespace(hcu_w4a8_hipc=True)
            should_fuse_routed_scaling_factor_in_topk=False
            def __call__(self,x,topk):return x*3
        backend.is_deepep=lambda:True
        obj.experts=CTExperts();obj.routed_scaling_factor=4.0
        torch.testing.assert_close(scope["_forward_full"](obj,local,object()),local*28)
        obj.experts.should_fuse_routed_scaling_factor_in_topk=True
        torch.testing.assert_close(scope["_forward_full"](obj,local,object()),local*19)

    def test_ragged_offsets_expand_local_queries_to_full_indexer_rows(self):
        import torch
        from dataclasses import MISSING, fields
        from sglang.srt.layers.attention.dsa_backend import DSAMetadata
        from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
        mod = load_policy()
        full_rows = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 2, 2])
        offsets = torch.tensor([0, 10, 30], dtype=torch.int32)
        required = {f.name: None for f in fields(DSAMetadata) if f.default is MISSING and f.default_factory is MISSING}
        original = DSAMetadata(**dict(required, topk_indices_offset=offsets))
        metadata = SimpleNamespace(attn_metadata=original, topk_transform_method=TopkTransformMethod.RAGGED)
        batch = SimpleNamespace(batch_size=3, extend_seq_lens=torch.tensor([2, 6, 2]), req_pool_indices=torch.tensor([0, 1, 2]))
        name = "sglang.srt.model_executor.forward_context"
        stub = ModuleType(name)
        stub.get_req_to_token_pool = lambda: SimpleNamespace(req_to_token=torch.arange(192).reshape(3, 64))
        strategy = SimpleNamespace(shard_local_tokens=lambda x: x[::4])
        with patch.dict(sys.modules, {name: stub}), patch("sglang.srt.layers.cp.utils.get_cp_strategy", return_value=strategy):
            expanded = mod.full_indexer_metadata(batch, metadata)
        tree = ast.parse((_ROOT / "hcu_sglang/models/hyv4_flash/model.py").read_text())
        consumer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_ragged_offsets")
        consumer.returns = None
        for arg in consumer.args.args:
            arg.annotation = None
        scope = {"TopkTransformMethod": TopkTransformMethod, "torch": torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[consumer], type_ignores=[])), "ragged_consumer", "exec"), scope)
        actual = scope["_ragged_offsets"](SimpleNamespace(metadata=expanded, num_rows=10))
        torch.testing.assert_close(actual, torch.tensor([0, 0, 10, 10, 10, 10, 10, 10, 30, 30], dtype=torch.int64))
        torch.testing.assert_close(original.topk_indices_offset, offsets)
        self.assertIsNot(expanded.attn_metadata, original)
        actual = mod.expand_ragged_offsets(torch.tensor([0, 0]), full_rows, full_rows[3::4], 3)
        torch.testing.assert_close(actual[3::4], torch.zeros(2, dtype=torch.int64))
        with self.assertRaises(ValueError):
            mod.expand_ragged_offsets(offsets[:1], full_rows, full_rows[::4], 3)

    def test_only_supported_non_speculative_profile_is_admitted(self):
        mod=load_policy();cfg=dict(cp_strategy="interleave",tp_size=8,attn_cp_size=8,ep_size=8,moe_dense_tp_size=8,deepep_mode="normal",moe_a2a_backend="deepep",attn_dp_size=1,dp_size=1,pp_size=1,dcp_size=1,speculative_algorithm=None,disaggregation_mode="null",language_model_only=True,enable_dsa_cache_layer_split=False)
        mod.validate_profile(SimpleNamespace(**cfg))
        for key,value in [("deepep_mode","auto"),("deepep_mode","low_latency"),("moe_dense_tp_size",1),("speculative_algorithm","DFLY"),("language_model_only",False),("moe_a2a_backend","none"),("cp_strategy","zigzag")]:
            with self.subTest(key=key),self.assertRaises(ValueError):mod.validate_profile(SimpleNamespace(**dict(cfg,**{key:value})))


if __name__=="__main__":unittest.main()
