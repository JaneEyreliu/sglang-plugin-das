from pathlib import Path
import importlib.util
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


spec = importlib.util.spec_from_file_location("hyv4_fp8_v_cpu_test", Path(__file__).resolve().parents[1] / "hcu_sglang/models/hyv4_fp8_v.py")
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)


def mod(name, **kw):
    m = ModuleType(name)
    m.__dict__.update(kw)
    return m


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.flag = SimpleNamespace(get=lambda: True)
        self.quantize = Mock(side_effect=lambda w, **kw: (w, torch.ones(1)))
        self.mods = patch.dict(sys.modules, {
            "sglang.srt.environ": mod("env", envs=SimpleNamespace(SGLANG_HY4_FP8_V_UP_PROJ=self.flag)),
            "sglang.kernels.ops.quantization.fp8_kernel": mod("fp8", fp8_dtype=torch.float8_e4m3fn),
            "sglang.srt.layers.quantization.fp8_utils": mod("utils", input_to_float8=self.quantize),
        })
        self.mods.start()
        self.addCleanup(self.mods.stop)

    def test_disabled_never_compiles_or_requires_weights(self):
        self.flag.get = lambda: False
        with patch.object(impl, "load_hipblaslt_fp8_bmm") as loader:
            attn = SimpleNamespace()
            impl.prepare(attn, is_hcu=True, use_aiter_gfx95=False)
            self.assertFalse(attn.use_hipblaslt_fp8_v_up_proj)
            loader.assert_not_called()

    def test_scaled_transposed_copy_preserves_original_bf16_weights(self):
        for scale in (1.0, 0.75, torch.tensor(0.5)):
            weight = torch.arange(24).reshape(2, 3, 4).to(torch.bfloat16)
            original = weight.clone()
            attn = SimpleNamespace(w_vc=weight, w_scale=scale, use_deep_gemm_bmm=False)
            with patch.object(impl, "load_hipblaslt_fp8_bmm") as loader:
                impl.prepare(attn, is_hcu=True, use_aiter_gfx95=False)
                loader.assert_called_once()
            self.assertIs(attn.w_vc, weight)
            torch.testing.assert_close(weight, original)
            torch.testing.assert_close(attn.w_vc_fp8, (weight * scale).transpose(1,2).contiguous())
            self.assertTrue(attn.w_vc_fp8.is_contiguous())

    def test_incompatible_profiles_fail_before_compilation(self):
        for is_hcu, aiter, deepgemm, dtype in ((False,False,False,torch.bfloat16), (True,True,False,torch.bfloat16), (True,False,True,torch.bfloat16), (True,False,False,torch.float32)):
            attn = SimpleNamespace(w_vc=torch.ones(2,3,4,dtype=dtype), use_deep_gemm_bmm=deepgemm)
            with patch.object(impl, "load_hipblaslt_fp8_bmm") as loader:
                with self.assertRaises(ValueError):
                    impl.prepare(attn, is_hcu=is_hcu, use_aiter_gfx95=aiter)
                loader.assert_not_called()

    def test_empty_projection_preserves_flattened_shape_without_native_load(self):
        with patch.object(impl, "load_hipblaslt_fp8_bmm") as loader:
            result = impl.project(SimpleNamespace(num_local_heads=4, v_head_dim=8), torch.empty(0,4,16,dtype=torch.bfloat16))
            self.assertEqual(result.shape, (0,32))
            loader.assert_not_called()
