import unittest

import torch

from hcu_sglang.speculative_dfly import register_hooks
from hcu_sglang.speculative_dfly.utils import DFlyRequestState, build_draft_topk, commit_accept
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class DFlyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hcu_sglang.compat.aiter import bootstrap_aiter_compat
        bootstrap_aiter_compat()
        from sglang.srt.plugins import load_plugins
        load_plugins()

    def test_resolution_and_aux_capture(self):
        from sglang.srt.server_args import prepare_server_args
        from sglang.srt.arg_groups.overrides import resolving_view
        from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import resolve_spec_aux_hidden_state_config
        args = prepare_server_args(["--model-path", "/llm-models3/Hy4-flash-Channel-INT4-w4a8", "--trust-remote-code", "--tp", "8", "--quantization", "slimquant_w4a8_marlin", "--attention-backend", "dsa", "--dsa-prefill-backend", "flashmla_sparse", "--dsa-decode-backend", "flashmla_kv", "--language-model-only", "--disable-prefill-cuda-graph", "--context-length", "8192", "--speculative-algorithm", "DFLY"])
        args.resolve_once()
        cfg = resolving_view(args)
        self.assertEqual(cfg.speculative_num_draft_tokens, 6)
        self.assertEqual(cfg.speculative_num_steps, 1)
        self.assertEqual(cfg.speculative_eagle_topk, 1)
        aux = resolve_spec_aux_hidden_state_config(server_args=args, model_config=None, spec_algorithm=SpeculativeAlgorithm.from_string("DFLY"), is_draft_worker=False)
        self.assertFalse(aux.dflash_use_aux_hidden_state)
        self.assertIsNone(aux.draft_kv_num_layers)
        ordinary = resolve_spec_aux_hidden_state_config(server_args=args, model_config=None, spec_algorithm=SpeculativeAlgorithm.NONE, is_draft_worker=False)
        self.assertFalse(ordinary.dflash_use_aux_hidden_state)

    def test_logical_capture_request_order(self):
        import types
        from unittest.mock import patch
        from hcu_sglang.speculative_dfly.utils import DFlyLogicalTopkCapture
        output = torch.full((8, 3), -1, dtype=torch.int32)
        capture = DFlyLogicalTopkCapture(output)
        indices = torch.tensor([[0, -1, -1], [1, 0, -1], [0, -1, -1], [1, 0, -1], [2, 0, -1]], dtype=torch.int32)
        table = torch.tensor([[50, 51, 52], [70, 71, 72]], dtype=torch.int32)
        batch = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_target_verify=lambda: False, is_extend=lambda: True), extend_seq_lens=torch.tensor([2, 3]))
        backend = types.SimpleNamespace(forward_metadata=types.SimpleNamespace(page_table_1=table))
        with patch("sglang.srt.model_executor.forward_context.get_attn_backend", return_value=backend):
            capture.record(61, indices, batch)
        self.assertEqual(output[:2].tolist(), [[51, 50, -1], [72, 70, -1]])

    def test_normalized_greedy_request(self):
        import types
        from unittest.mock import patch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from hcu_sglang.speculative_dfly import validate_request
        sampling = SamplingParams(temperature=0)
        self.assertEqual(sampling.top_k, 1)
        self.assertEqual(sampling.temperature, 1.0)
        req = types.SimpleNamespace(sampling_params=sampling, return_logprob=False, grammar=None)
        seen = []
        def original(request, overlap):
            seen.append(request)
            return None
        with patch("sglang.srt.runtime_context.get_spec", return_value=types.SimpleNamespace(speculative_algorithm="DFLY")):
            self.assertIsNone(validate_request(original, req, True))
            req.sampling_params = SamplingParams(temperature=0.9)
            self.assertIn("greedy", validate_request(original, req, True))
        self.assertEqual(seen, [req])

    def test_public_dispatch(self):
        register_hooks()
        algo = SpeculativeAlgorithm.from_string("DFLY")
        self.assertTrue(algo.is_dflash_family())
        self.assertFalse(algo.need_topk())
        self.assertFalse(algo.carries_draft_hidden_states())
        self.assertEqual(algo.get_num_tokens_per_req_for_target_verify(7, False), 7)

    def test_checkpoint_declared_quantization(self):
        import json
        from pathlib import Path
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from hcu_sglang.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8LinearMethod
        from hcu_sglang.layers.quantization.slimquant_w4a8_marlin import SlimQuantW4A8Int8MarlinConfig
        layer = LinearBase.__new__(LinearBase)
        torch.nn.Module.__init__(layer)
        for model, formal in (("Hy4-flash-Channel-INT4-w4a8", False), ("Hy4-Channel-INT4-w4a8", True)):
            raw = json.loads((Path("/llm-models3") / model / "config.json").read_text())
            quant = dict(raw.get("quantization_config") or raw["text_config"]["quantization_config"])
            quant["hf_config"] = {"architectures": ["HYV4ForCausalLM"]}
            config = SlimQuantW4A8Int8MarlinConfig.from_config(quant)
            for suffix, quantized in (("0.self_attn.q_b_proj", True), ("0.self_attn.kv_b_proj", formal), ("0.self_attn.fused_qkv_a_proj_with_mqa", formal), ("0.mlp.gate_up_proj", True), ("hidden_correction.gate_up_proj", formal), ("mtp_fusion.eh_proj", False)):
                with self.subTest(model=model, suffix=suffix):
                    method = config.get_quant_method(layer, "model.dfly_layers." + suffix)
                    self.assertIsInstance(method, SlimQuantW4A8Int8LinearMethod if quantized else UnquantizedLinearMethod)

    def test_native_int8_loader_retains_codes_and_scales(self):
        from types import SimpleNamespace
        from hcu_sglang.speculative_dfly.model import HYV4FlashDFly
        from hcu_sglang.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8LinearMethod
        from sglang.srt.model_loader.weight_utils import default_weight_loader
        model = HYV4FlashDFly.__new__(HYV4FlashDFly)
        torch.nn.Module.__init__(model)
        model.quant_config = SimpleNamespace(checkpoint_format="hy4_compressed_tensors_w4a8")
        model.is_formal = False
        model.layers = torch.nn.ModuleList()
        model.norm = torch.nn.LayerNorm(2, bias=False, dtype=torch.bfloat16)
        model.x = torch.nn.Module()
        method = SlimQuantW4A8Int8LinearMethod(None)
        method.create_weights(model.x, 256, [128], 256, 128, torch.bfloat16, weight_loader=default_weight_loader)
        codes = torch.randint(-127, 128, (128, 256), dtype=torch.int8)
        scales = torch.linspace(.0001, .0077, 128).view(-1, 1)
        model.load_weights([("norm.weight", torch.ones(2, dtype=torch.bfloat16)), ("x.weight_scale", scales), ("x.weight", codes)])
        self.assertEqual(model.x.weight.dtype, torch.int8)
        self.assertEqual(model.x.weight_scale.dtype, torch.float32)
        torch.testing.assert_close(model.x.weight, codes, rtol=0, atol=0)
        torch.testing.assert_close(model.x.weight_scale, scales, rtol=0, atol=0)

    def test_formal_absorbed_kv_keeps_int8_parameter(self):
        from types import SimpleNamespace
        from hcu_sglang.speculative_dfly.model import _split_kv_b_proj
        codes = torch.tensor([[1, -2], [3, -4], [5, 6], [-7, 8]], dtype=torch.int8)
        scales = torch.tensor([[.0031], [.0073], [.0117], [.0179]], dtype=torch.float32)
        attn = SimpleNamespace(kv_b_proj=SimpleNamespace(weight=codes, weight_scale=scales), qk_nope_head_dim=1, v_head_dim=1)
        with self.assertRaisesRegex(ValueError, "must be bf16"):
            _split_kv_b_proj(attn)
        _split_kv_b_proj(attn, True)
        expected = (codes.to(torch.bfloat16) * scales.to(torch.bfloat16)).view(2, 2, 2)
        torch.testing.assert_close(attn.w_kc, expected[:, :1], rtol=0, atol=0)
        torch.testing.assert_close(attn.w_vc, expected[:, 1:].transpose(1, 2), rtol=0, atol=0)
        self.assertIs(attn.kv_b_proj.weight, codes)
        self.assertIs(attn.kv_b_proj.weight_scale, scales)

    @unittest.skipUnless(torch.cuda.is_available(), "requires HCU")
    def test_community_greedy_mm_and_graph(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from hcu_sglang.speculative_dfly.model import HYV4FlashDFly
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        hidden = torch.randn(6, 256, device="cuda", dtype=torch.bfloat16)
        model = SimpleNamespace(lm_head=SimpleNamespace(weight=weight, shard_indices=SimpleNamespace(num_org_elements=128, org_vocab_start_index=0)))
        parallel = SimpleNamespace(tp_group=SimpleNamespace(world_size=1))
        with patch("hcu_sglang.speculative_dfly.model.get_parallel", return_value=parallel):
            HYV4FlashDFly._greedy(model, hidden)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = HYV4FlashDFly._greedy(model, hidden)
            for _ in range(2):
                hidden.copy_(torch.randn_like(hidden))
                graph.replay()
                torch.testing.assert_close(actual, (hidden.float() @ weight.float().T).argmax(-1))

    def test_accept_boundary(self):
        candidates = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])
        out, lengths = commit_accept(candidates, torch.tensor([0, 2, 3]), torch.tensor([99, 98, 97]))
        self.assertEqual(lengths.tolist(), [1, 3, 4])
        self.assertEqual([row[:n].tolist() for row, n in zip(out, lengths)], [[99], [21, 22, 98], [31, 32, 33, 97]])

    @unittest.skipUnless(torch.cuda.is_available(), "requires HCU")
    def test_sparse_history_and_request_reuse(self):
        device = "cuda"
        state = DFlyRequestState(pool_size=4, hidden_size=2, num_layers=2, target_topk=8, dtype=torch.bfloat16, device=device)
        topk = torch.tensor([[[5, -1, 7, -1, -1, -1, -1, -1]], [[9, -1, 11, -1, -1, -1, -1, -1]]], dtype=torch.int32, device=device)
        req = torch.tensor([2], device=device)
        state.write(req_pool_indices=req, hidden=torch.ones((1, 2), device=device, dtype=torch.bfloat16), topk=topk)
        out = torch.full((2, 3, 128), 666, dtype=torch.int32, device=device)
        counts = torch.zeros(3, dtype=torch.int32, device=device)
        slots = torch.tensor([[40, 41, 42]], device=device)
        build_draft_topk(state_topk=state.topk, req_pool_indices=req, verify_slots=slots, out_rows=out, out_counts=counts)
        self.assertEqual(counts.tolist(), [3, 4, 5])
        for layer, history in enumerate(([5, 7], [9, 11])):
            for step in range(3):
                self.assertEqual(out[layer, step, :step+3].tolist(), history + [40, 41, 42][:step+1])
                self.assertTrue(bool((out[layer, step, step+3:] == -1).all()))
        state.write(req_pool_indices=req, hidden=torch.zeros((1, 2), device=device, dtype=torch.bfloat16), topk=torch.full_like(topk, -1))
        build_draft_topk(state_topk=state.topk, req_pool_indices=req, verify_slots=slots, out_rows=out, out_counts=counts)
        self.assertEqual(counts.tolist(), [1, 2, 3])
        self.assertEqual(out[0, 0, :3].tolist(), [40, -1, -1])


if __name__ == "__main__":
    unittest.main()
