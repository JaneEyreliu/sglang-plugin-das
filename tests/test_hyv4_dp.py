import ast
import importlib.util
import types
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def policy():
    spec = importlib.util.spec_from_file_location("hyv4_dp_test", ROOT / "hcu_sglang/models/hyv4_dp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def method(class_name, name, scope):
    tree = ast.parse((ROOT / "hcu_sglang/models/hyv4_flash/model.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "model", "exec"), scope)
    return scope[name]


class Hyv4DPTests(unittest.TestCase):
    def test_dp_layout_prepared_before_each_embedding(self):
        events = []
        class ReachedEmbedding(Exception):
            pass
        def embed(ids):
            events.append("embedding")
            raise ReachedEmbedding
        obj = SimpleNamespace(embed_tokens=embed)
        scope = {"dp_runtime": SimpleNamespace(active=lambda: True, prepare=lambda batch, rows: events.append(rows))}
        forward = method("HYV4FlashModel", "forward", scope)
        for rows in (3, 1):
            with self.assertRaises(ReachedEmbedding):
                forward(obj, SimpleNamespace(shape=(rows,)), None, object())
        self.assertEqual(events, [3, "embedding", 1, "embedding"])

    def test_engram_layout_ignores_local_deepep_metadata(self):
        import sys
        import torch
        from unittest.mock import patch
        counts = [2, 0, 0, 1, 0, 0, 0, 0]
        parts = [torch.arange(n * 2, dtype=torch.int64).reshape(n, 2) + (1 << 40) + rank * 100 for rank, n in enumerate(counts)]
        expected = torch.cat(parts)
        cpu_group, device_group = object(), object()
        context = types.ModuleType("sglang.srt.runtime_context")
        mod = policy()
        for rank in (0, 1, 3, 7):
            context.get_parallel = lambda: SimpleNamespace(tp_size=8, tp_rank=rank, tp_group=SimpleNamespace(cpu_group=cpu_group, device_group=device_group))
            def collect_counts(out, rows, group):
                self.assertIs(group, cpu_group)
                self.assertEqual(rows, counts[rank])
                out[:] = counts
            def collect_rows(out, local, group):
                self.assertIs(group, device_group)
                torch.testing.assert_close(local[:counts[rank]], parts[rank])
                out.zero_()
                for i, part in enumerate(parts):
                    out[i * 2:i * 2 + len(part)].copy_(part)
            batch = SimpleNamespace(global_num_tokens_cpu=[counts[rank]], global_dp_buffer_len=counts[rank])
            with patch.dict(sys.modules, {context.__name__: context}), patch("torch.distributed.all_gather_object", collect_counts), patch("torch.distributed.all_gather_into_tensor", collect_rows):
                mod.prepare(batch, counts[rank])
                torch.testing.assert_close(mod.gather(parts[rank], batch), expected)
                torch.testing.assert_close(mod.shard(expected, counts[rank], batch), parts[rank])
                with self.assertRaises(ValueError):
                    mod.shard(expected, counts[rank] + 1, batch)

    def test_attention_empty_rank_uses_actual_upstream_prepare(self):
        import torch
        src = ROOT.parent / "sglang/python/sglang/srt/models/deepseek_v2.py"
        if not src.exists():
            self.skipTest("requires the deployed SGLang source tree")
        tree = ast.parse(src.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV2AttentionMLA")
        prepare = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_prepare")
        prepare.decorator_list = []
        for arg in prepare.args.args + prepare.args.kwonlyargs:
            arg.annotation = None
        prepare.returns = None
        scope = {"torch": torch, "get_attn_tp_context": lambda: SimpleNamespace(input_scattered=False)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[prepare], type_ignores=[])), "upstream_prepare", "exec"), scope)
        flash_tree = ast.parse((ROOT / "hcu_sglang/models/hyv4_flash/model.py").read_text())
        flash_cls = next(n for n in flash_tree.body if isinstance(n, ast.ClassDef) and n.name == "HYV4FlashAttention")
        init = next(n for n in flash_cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        assignment = next(n for n in init.body if isinstance(n, ast.Assign) and any(ast.unparse(t) == "self.o_proj.reduce_results" for t in n.targets))
        for size in (1, 8):
            obj = SimpleNamespace(o_proj=SimpleNamespace(reduce_results=False), attn_mha=SimpleNamespace(kv_b_proj=object()))
            policy_scope = {"self": obj, "get_parallel": lambda: SimpleNamespace(attn_tp_size=size)}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])), "flash_constructor", "exec"), policy_scope)
            self.assertEqual(obj.o_proj.reduce_results, size > 1)
            if size == 1:
                hidden = torch.empty(0, 4)
                batch = object()
                output = scope["forward_prepare"](obj, None, hidden, batch, None)
                self.assertIs(output[0], hidden)
                self.assertIs(output[2], batch)

    def test_profile_rejects_unimplemented_combinations(self):
        cfg = dict(enable_dp_attention=True, tp_size=8, ep_size=8, dp_size=1,
                   attn_dp_size=8, moe_dense_tp_size=1, moe_a2a_backend="deepep",
                   attn_cp_size=1, dcp_size=1, pp_size=1, enable_prefill_cp=False,
                   speculative_algorithm=None, disaggregation_mode="null",
                   language_model_only=True, enable_dsa_cache_layer_split=False,
                   deepep_mode="normal", max_running_requests=64)
        mod = policy()
        for flash in (False, True):
            mod.validate_profile(SimpleNamespace(**cfg), flash=flash)
            for key, value in [("dp_size", 4), ("attn_cp_size", 8), ("ep_size", 4),
                               ("moe_dense_tp_size", 8), ("moe_a2a_backend", "none"),
                               ("speculative_algorithm", "EAGLE"), ("max_running_requests", 4)]:
                with self.subTest(flash=flash, key=key), self.assertRaises(ValueError):
                    mod.validate_profile(SimpleNamespace(**dict(cfg, **{key: value})), flash=flash)
        mod.validate_profile(SimpleNamespace(**dict(cfg, deepep_mode="low_latency")), flash=False)
        for mode in ("auto", "low_latency"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                mod.validate_profile(SimpleNamespace(**dict(cfg, deepep_mode=mode)), flash=True)
        dfly = dict(cfg, speculative_algorithm="DFLY", enable_dp_lm_head=True,
                    kv_cache_dtype="auto", dtype="bfloat16", disable_cuda_graph=True,
                    disable_prefill_cuda_graph=True)
        for flash in (False, True):
            mod.validate_profile(SimpleNamespace(**dfly), flash=flash)
            for key, value in [("enable_dp_lm_head", False), ("kv_cache_dtype", "fp8_e4m3"),
                               ("dtype", "float16"), ("disable_cuda_graph", False),
                               ("disable_prefill_cuda_graph", False)]:
                with self.subTest(flash=flash, key=key), self.assertRaises(ValueError):
                    mod.validate_profile(SimpleNamespace(**dict(dfly, **{key: value})), flash=flash)

    def test_engram_projects_aligned_global_hashes_before_local_combine(self):
        import torch
        hashes = torch.tensor([[3, 9], [4, 1], [8, 6]])
        full_kv = torch.arange(15, dtype=torch.float32).reshape(3, 5)
        for start, end in [(0, 1), (1, 3), (3, 3)]:
            seen = []
            local_hashes = hashes[start:end].clone()
            hidden = torch.ones(end-start, 2, 4)
            def gather(x, batch):
                self.assertIs(x, local_hashes)
                seen.append("gather")
                return hashes
            def project(x, dtype, native):
                self.assertIs(x, hashes)
                seen.append("project")
                return full_kv
            def shard(x, rows, batch):
                self.assertIs(x, full_kv)
                self.assertEqual(rows, end-start)
                seen.append("shard")
                return x[start:end].clone()
            def combine(kv, x, h):
                self.assertIs(x, hidden)
                self.assertIs(h, local_hashes)
                torch.testing.assert_close(kv, full_kv[start:end])
                return kv
            scope = {"dp_runtime": SimpleNamespace(active=lambda: True, gather=gather, shard=shard)}
            obj = SimpleNamespace(_project=project, _combine=combine)
            output = method("FlashEngram", "forward", scope)(obj, hidden, local_hashes, object())
            torch.testing.assert_close(output, full_kv[start:end])
            self.assertEqual(seen, ["gather", "project", "shard"])

    def test_engram_sharded_projection_matches_full_weight_oracle(self):
        import torch
        import torch.nn.functional as F
        hashes = torch.ones(3, 2, dtype=torch.int64)
        values = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        weight = (torch.arange(24).reshape(6, 4) % 3).to(torch.bfloat16)
        bias = torch.ones(6, dtype=torch.bfloat16)
        oracle = F.linear(values, weight, bias)
        hidden = torch.arange(12, dtype=torch.bfloat16).reshape(3, 2, 2) / 8
        for rank, (start, end) in enumerate([(0, 1), (1, 3)]):
            part = slice(rank*2, (rank+1)*2)
            local_bias = bias if rank == 0 else torch.zeros_like(bias)
            expected_partial = F.linear(values[:, part], weight[:, part], local_bias)
            sums = []
            def reduce(x):
                torch.testing.assert_close(x, expected_partial)
                sums.append(x.clone())
                return oracle
            scope = {"torch": torch, "F": F, "_maybe_all_reduce": reduce,
                     "dp_runtime": SimpleNamespace(active=lambda: True,
                        gather=lambda x, b: hashes,
                        shard=lambda x, rows, b: x[start:end].clone())}
            obj = SimpleNamespace(multi_head_embedding=SimpleNamespace(weight=torch.empty(0)),
                                  _lookup_eager=lambda h, dtype: values[:, part],
                                  _fused_kv=(weight[:, part], local_bias),
                                  hc_mult=2, hidden_size=2,
                                  norm_key=torch.ones(2, 2, dtype=torch.bfloat16),
                                  norm_query=torch.ones(2, 2, dtype=torch.bfloat16),
                                  unified_pre_gamma=torch.ones(2, dtype=torch.bfloat16),
                                  unified_per_hc_scales=torch.zeros(2, 2, dtype=torch.bfloat16))
            for name in ["_project", "_combine", "_combine_eager"]:
                setattr(obj, name, types.MethodType(method("FlashEngram", name, scope), obj))
            expected = obj._combine_eager(oracle, hidden, hashes)[start:end]
            forward = method("FlashEngram", "forward", scope)
            actual = forward(obj, hidden[start:end], hashes[start:end], object())
            torch.testing.assert_close(actual, expected)
            self.assertEqual(len(sums), 1)

    def test_dp_shared_expert_is_local_and_empty_rank_dispatches(self):
        import torch
        from collections import namedtuple
        Output = namedtuple("Output", "topk_ids topk_weights")
        calls = []
        class Experts:
            should_fuse_routed_scaling_factor_in_topk = True
            def __call__(self, x, topk):
                calls.append((x.shape[0], topk.topk_ids.clone(), topk.topk_weights.clone()))
                return x * 3
        class TopK:
            def __call__(self, x, logits):
                return Output(torch.zeros(x.shape[0], 2, dtype=torch.int32), torch.full((x.shape[0], 2), 0.5))
            def empty_topk_output(self, device, layer_id):
                return Output(torch.empty(0, 2, dtype=torch.int32), torch.empty(0, 2))
        def forbidden(x): raise AssertionError("DP shared expert reduced across unrelated tokens")
        scope = {"dp_runtime": SimpleNamespace(active=lambda: True),
                 "cp_runtime": SimpleNamespace(active=lambda batch: False),
                 "get_moe_a2a_backend": lambda: SimpleNamespace(is_deepep=lambda: True),
                 "_maybe_all_reduce": forbidden}
        obj = SimpleNamespace(layer_id=1, gate=lambda x: x, topk=TopK(), experts=Experts(),
                              routed_latent_down=None, routed_latent_up=None,
                              shared_experts=lambda x: x*2, combine_output_norm=None)
        forward = method("HYV4FlashMoE", "_forward_full", scope)
        for rows in (3, 0):
            x = torch.ones(rows, 2)
            torch.testing.assert_close(forward(obj, x, object()), x*5)
        self.assertEqual([n for n, _, _ in calls], [3, 0])
        self.assertEqual(tuple(calls[1][1].shape), (0, 2))
        self.assertEqual(tuple(calls[1][2].shape), (0, 2))


if __name__ == "__main__":
    unittest.main()
