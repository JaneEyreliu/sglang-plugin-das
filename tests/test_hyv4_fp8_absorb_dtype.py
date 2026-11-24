import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch


class FP8AbsorbPostLoadTests(unittest.TestCase):
    def test_fp32_scale_keeps_bf16_bmm_weights(self):
        import torch
        from hcu_sglang.models.deepseek.deepseek_common.methods import weight_loader as m
        weight = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3).to(torch.float8_e4m3fn)
        scale = torch.tensor([0.125], dtype=torch.float32)
        for nextn in (False, True):
            with self.subTest(nextn=nextn):
                attn = NS(kv_b_proj=NS(weight=weight, weight_scale=torch.ones(4, 1)),
                          qk_nope_head_dim=1, v_head_dim=1, w_kc=None, w_vc=None, w_scale=1.0)
                model = NS(config=NS(num_hidden_layers=1, architectures=["HYV4ForCausalLM"]),
                           quant_config=NS(weight_block_size=None),
                           model=NS(start_layer=0, end_layer=1, layers=[NS(self_attn=attn)],
                                    decoder=NS(self_attn=attn)))
                with patch.multiple(m, _is_hcu=True, _is_fp8_fnuz=False,
                                    _use_aiter_gfx95=False, _is_npu=False, _is_musa=False), patch.object(
                    m, "channel_quant_to_tensor_quant", return_value=(weight, scale)
                ):
                    m.post_load_weights(model, is_nextn=nextn)
                expected = (weight.float() * scale).bfloat16().reshape(2, 2, 3)
                self.assertEqual(attn.w_kc.dtype, torch.bfloat16)
                self.assertEqual(attn.w_vc.dtype, torch.bfloat16)
                torch.testing.assert_close(attn.w_kc, expected[:, :1], rtol=0, atol=0)
                torch.testing.assert_close(attn.w_vc, expected[:, 1:].transpose(1, 2), rtol=0, atol=0)
                q = torch.ones(2, 1, 1, dtype=torch.bfloat16)
                torch.testing.assert_close(torch.bmm(q, attn.w_kc), expected[:, :1], rtol=0, atol=0)
                self.assertEqual(attn.w_scale, 1.0)
