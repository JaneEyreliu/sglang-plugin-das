import importlib.util
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sink = load_module("hyv4_sink_test", "hcu_sglang/layers/attention/current_dsa/hyv4_sink.py")
model = load_module("hyv4_dcp_model_test", "hcu_sglang/models/hyv4_dcp.py")


class SinkTests(unittest.TestCase):
    def test_partition_merge_matches_one_global_sink(self):
        gen = torch.Generator().manual_seed(532)
        for dcp in (2, 4, 8):
            for base2 in (False, True):
                for dtype in (torch.float32, torch.bfloat16):
                    with self.subTest(dcp=dcp, base2=base2, dtype=dtype):
                        logits = torch.randn(3, 4, dcp * 3, generator=gen)
                        values = torch.randn(3, 4, dcp * 3, 5, generator=gen)
                        bias = torch.randn(4, generator=gen)
                        valid = torch.ones(3, dcp * 3, dtype=torch.bool)
                        valid[0, :3] = False
                        valid[1] = False
                        logits = logits.masked_fill(~valid[:, None], -torch.inf)
                        ref_lse = torch.logaddexp(torch.logsumexp(logits, -1), bias)
                        ref = (torch.exp(logits - ref_lse[..., None])[..., None] * values).sum(-2)
                        outs, lses = [], []
                        for rank in range(dcp):
                            q = slice(rank * 3, (rank + 1) * 3)
                            chunk = logits[..., q]
                            local_lse = torch.logsumexp(chunk, -1)
                            out = (torch.softmax(chunk, -1)[..., None] * values[..., q, :]).sum(-2)
                            counts = valid[:, q].sum(-1)
                            bad = counts == 0
                            out[bad] = torch.nan
                            local_lse[bad] = torch.inf
                            if base2:
                                local_lse = local_lse / torch.log(torch.tensor(2.0))
                            actual, lse = sink.apply_hyv4_sink(
                                out.to(dtype), local_lse, bias, counts,
                                dcp_size=dcp, lse_base2=base2,
                            )
                            outs.append(actual.float())
                            lses.append(lse * torch.log(torch.tensor(2.0)) if base2 else lse)
                        merged_lse = torch.logsumexp(torch.stack(lses), 0)
                        merged = (torch.stack(outs) * torch.exp(torch.stack(lses)-merged_lse)[..., None]).sum(0)
                        torch.testing.assert_close(merged_lse, ref_lse, atol=1e-6, rtol=1e-6)
                        torch.testing.assert_close(merged, ref, atol=0.006 if dtype==torch.bfloat16 else 1e-6, rtol=0.006 if dtype==torch.bfloat16 else 1e-6)
                        self.assertTrue(torch.isfinite(merged).all())

    def test_replicated_sink_loading_and_head_selection(self):
        for dcp in (2, 4, 8):
            for rank in range(dcp):
                layer = SimpleNamespace(num_local_heads=4, learnable_sink_param=torch.nn.Parameter(torch.empty(4)), attn_mqa=SimpleNamespace(), attn_mqa_for_dcp_decode=SimpleNamespace())
                parallel = SimpleNamespace(dcp_enabled=True, attn_tp_rank=rank, attn_dcp_size=dcp)
                model.configure_sink(layer, SimpleNamespace(num_attention_heads=dcp*4), parallel)
                weight = torch.arange(dcp*4, dtype=torch.bfloat16)
                layer.learnable_sink_param.weight_loader(layer.learnable_sink_param, weight)
                torch.testing.assert_close(layer.attn_mqa.hyv4_sink_getter(), weight[rank*4:(rank+1)*4].float())
                torch.testing.assert_close(layer.attn_mqa_for_dcp_decode.hyv4_sink_getter(), weight.float())
                with self.assertRaises(ValueError):
                    layer.learnable_sink_param.weight_loader(layer.learnable_sink_param, weight[:4])

    def test_non_dcp_retains_original_parameter(self):
        param = torch.nn.Parameter(torch.ones(4))
        layer = SimpleNamespace(learnable_sink_param=param)
        model.configure_sink(layer, None, SimpleNamespace(dcp_enabled=False))
        self.assertIs(layer.learnable_sink_param, param)


class BackendSinkTests(unittest.TestCase):
    def test_dcp_withholds_native_sink_and_corrects_output_and_lse_once(self):
        calls = []
        def native(**kwargs):
            calls.append(kwargs)
            return torch.ones(2,1,4,3), torch.zeros(2,4,1)
        ns = dict(torch=torch, _is_hcu=True, get_flashmla_op=lambda *a, **k: native,
                  get_flashmla_kv_valid_rows=lambda *a: None,
                  fixup_zero_kv_rows=lambda *a, **k: None)
        path = ROOT / "hcu_sglang/layers/attention/current_dsa/backend.py"
        tree = ast.parse(path.read_text())
        fn, = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_forward_flashmla_kv"]
        fn.returns = None
        for a in fn.args.args:
            a.annotation = None
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
        pool = SimpleNamespace(
            _flashmla_kv_target_q_heads_for_runtime=lambda n: n,
            _hcu_flashmla_decode_h16_enabled=False,
            physical_page_size=64, kv_cache_dim=2, dsa_kv_cache_store_fp8=True,
            dsa_index_topk=2, dcp_size=4,
            get_device_int32_arange=lambda n: torch.arange(n),
        )
        layer = SimpleNamespace(tp_q_head_num=4, head_dim=2, hyv4_sink_getter=lambda: None)
        metadata = SimpleNamespace(dsa_cache_seqlens_int32=torch.tensor([1,0]), flashmla_metadata=SimpleNamespace(flashmla_metadata=None,num_splits=None))
        from unittest.mock import patch
        import sys
        namespace = SimpleNamespace(apply_hyv4_sink=sink.apply_hyv4_sink)
        with patch.dict(sys.modules, {"hcu_sglang.layers.attention.current_dsa.hyv4_sink":namespace}):
            out, lse = ns["_forward_flashmla_kv"](
                pool, torch.zeros(2,4,2), torch.zeros(64,2), 3, 1.0,
                layer, metadata, torch.tensor([[0,-1],[-1,-1]]), None,
                attn_sink=torch.zeros(4), return_lse=True,
            )
        self.assertIsNone(calls[0]["attn_sink"])
        torch.testing.assert_close(out[0], torch.full((4,3),0.8))
        torch.testing.assert_close(lse[0], torch.full((4,), torch.log2(torch.tensor(1.25)).item()))
        torch.testing.assert_close(out[1], torch.zeros(4,3))
        torch.testing.assert_close(lse[1], torch.full((4,),-2.0))
