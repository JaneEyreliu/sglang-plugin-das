import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def extract(file, cls_name, name, scope):
    tree = ast.parse((ROOT / file).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    for arg in fn.args.args + fn.args.kwonlyargs:
        arg.annotation = None
    fn.returns = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), file, "exec"), scope)
    return scope[name]


class DFlyDPTests(unittest.TestCase):
    def test_logical_capture_refreshes_decode_rows_during_mixed_prefill(self):
        import sys
        import types
        from unittest.mock import patch
        output = torch.full((6, 3), -1, dtype=torch.int32)
        ids = torch.tensor([[1, 0, -1], [0, -1, -1]], dtype=torch.int32)
        table = torch.tensor([[20, 21], [70, 71]], dtype=torch.int32)
        mode = SimpleNamespace(is_target_verify=lambda: False, is_extend=lambda: False, is_decode=lambda: True)
        batch = SimpleNamespace(forward_mode=mode)
        module = types.ModuleType("sglang.srt.model_executor.forward_context")
        module.get_attn_backend = lambda: SimpleNamespace(forward_metadata=SimpleNamespace(page_table_1=table))
        with patch.dict(sys.modules, {module.__name__: module}):
            extract("hcu_sglang/speculative_dfly/utils.py", "DFlyLogicalTopkCapture", "record", dict(torch=torch))(SimpleNamespace(output=output), 1, ids, batch)
        self.assertEqual(output[:2].tolist(), [[21, 20, -1], [70, -1, -1]])

    def test_physical_capture_refreshes_decode_rows(self):
        calls = []
        ids = torch.tensor([[10, 11], [30, 31]], dtype=torch.int32)
        output = torch.full((8, 2), -1, dtype=torch.int32)
        mode = SimpleNamespace(is_target_verify=lambda: False, is_extend=lambda: False, is_decode=lambda: True)
        def seed(indices, incoming_mode):
            calls.append((indices, incoming_mode))
            return indices + 100
        scope = dict(torch=torch, get_attn_backend=lambda: SimpleNamespace(prepare_mtp_seed_indices=seed))
        extract("hcu_sglang/models/hyv4_flash/model.py", "FlashIndexPool", "_capture_dfly_topk", scope)(SimpleNamespace(dfly_topk_capture=output), ids, SimpleNamespace(forward_mode=mode))
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], ids)
        torch.testing.assert_close(output[:2], ids + 100)

    def test_idle_runs_empty_target_verify_before_publish(self):
        calls = []
        batch = object()
        class Verify:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.forward = SimpleNamespace(can_run_decode_cuda_graph=True)
            def prepare_for_verify(self, incoming, target):
                self.assert_batch = incoming
                calls.append(("prepare", incoming, self.kwargs))
                return self.forward, False
        def target_forward(**kwargs):
            calls.append(("target", kwargs))
        scope = dict(torch=torch, get_parallel=lambda: SimpleNamespace(attn_dp_enabled=True), DFlashVerifyInput=Verify,
                     CaptureHiddenMode=SimpleNamespace(FULL="full"), GenerationBatchResult=lambda **kw: SimpleNamespace(**kw),
                     make_draft_input_v2=lambda **kw: SimpleNamespace(**kw))
        obj = SimpleNamespace(device="cpu", block_size=7, _target_worker=SimpleNamespace(forward_batch_generation=target_forward))
        result = extract("hcu_sglang/speculative_dfly/worker.py", "DFlyWorkerV2", "_forward_idle", scope)(obj, batch, lambda x: calls.append(("publish", x)))
        self.assertEqual([x[0] for x in calls], ["prepare", "target", "publish"])
        self.assertIs(calls[0][1], batch)
        self.assertEqual(calls[0][2]["draft_token"].numel(), 0)
        self.assertEqual(calls[0][2]["draft_token_num"], 7)
        self.assertTrue(calls[1][1]["is_verify"])
        self.assertFalse(calls[1][1]["forward_batch"].can_run_decode_cuda_graph)
        self.assertEqual(result.next_token_ids.numel(), 0)
        self.assertEqual(result.accept_lens.numel(), 0)

    def test_draft_ffn_uses_attention_group_under_dp(self):
        calls = []
        class MLP(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                calls.append(kwargs)
                self.gate_up_proj = SimpleNamespace(output_sizes=[kwargs["intermediate_size"]])
        obj = SimpleNamespace(mlp=None, hidden_size=4, hidden_act="silu", quant_config=None, prefix="draft")
        scope = dict(torch=torch, DeepseekV2MLP=MLP, get_tensor_model_parallel_world_size=lambda: 8,
                     get_parallel=lambda: SimpleNamespace(attn_dp_enabled=True, attn_tp_size=1, tp_size=8))
        extract("hcu_sglang/speculative_dfly/model.py", "HYV4FlashDFlyLayer", "materialize_mlp", scope)(obj, 6, "cpu", torch.bfloat16)
        self.assertEqual(calls[0]["parallel_group"], "attn_tp")
        self.assertEqual(obj.mlp.gate_up_proj.output_sizes, [6])

    @unittest.skipUnless(os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") == "1", "HCU opt-in")
    def test_draft_greedy_uses_head_group_not_other_dp_requests(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("draft crossed into other attention-DP requests")
        shared = SimpleNamespace(world_size=8, all_gather_into_tensor=forbidden)
        head_group = SimpleNamespace(world_size=1)
        weight = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        hidden = torch.randn(3, 256, dtype=torch.bfloat16, device="cuda")
        head = SimpleNamespace(weight=weight, tp_group=head_group,
                               shard_indices=SimpleNamespace(num_org_elements=128, org_vocab_start_index=0))
        scope = dict(torch=torch, get_parallel=lambda: SimpleNamespace(tp_group=shared))
        actual = extract("hcu_sglang/speculative_dfly/model.py", "HYV4FlashDFly", "_greedy", scope)(SimpleNamespace(lm_head=head), hidden)
        torch.testing.assert_close(actual, (hidden.float() @ weight.float().T).argmax(-1))


if __name__ == "__main__":
    unittest.main()
