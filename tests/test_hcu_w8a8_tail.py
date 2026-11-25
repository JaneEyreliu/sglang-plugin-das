import os
import unittest

import torch


@unittest.skipUnless(os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") == "1", "HCU integration opt-in")
class W8A8TailContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hcu_sglang.compat.aiter import bootstrap_aiter_compat

        bootstrap_aiter_compat()
        from sglang.srt.plugins import load_plugins

        load_plugins()

    def test_small_m_k_tail_and_graph(self):
        from hcu_sglang.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8LinearMethod
        from lightop.quant import per_token_quant_int8

        torch.manual_seed(1124)
        layer = torch.nn.Module()
        layer.register_parameter("weight", torch.nn.Parameter(torch.randint(-127, 128, (3072, 1728), device="cuda", dtype=torch.int8), requires_grad=False))
        layer.register_parameter("weight_scale", torch.nn.Parameter(torch.full((3072, 1), .003, device="cuda", dtype=torch.float32), requires_grad=False))
        method = SlimQuantW4A8Int8LinearMethod(None)
        self.assertEqual(method.w8a8_strategy, 1)
        method.process_weights_after_loading(layer)

        def reference(x):
            values, scales = per_token_quant_int8(x)
            return ((values.float() @ layer.weight.float()) * scales * layer.weight_scale.T).to(x.dtype)

        for count in (1, 6, 12, 18, 24, 28, 32, 64, 65):
            with self.subTest(count=count):
                x = torch.randn(count, 1728, device="cuda", dtype=torch.bfloat16)
                actual = method.apply(layer, x)
                torch.testing.assert_close(actual, reference(x), rtol=.02, atol=.05)

        x = torch.randn(24, 1728, device="cuda", dtype=torch.bfloat16)
        method.apply(layer, x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = method.apply(layer, x)
        for _ in range(2):
            x.copy_(torch.randn_like(x))
            graph.replay()
            torch.testing.assert_close(output, reference(x), rtol=.02, atol=.05)


if __name__ == "__main__":
    unittest.main()
