from functools import cache
from pathlib import Path

import torch


@cache
def load_hipblaslt_fp8_bmm():
    from torch.utils.cpp_extension import load

    source = (
        Path(__file__).resolve().parents[1]
        / "layers/attention/current_dsa/csrc/hipblaslt_fp8_bmm.cu"
    )
    return load(
        name="hcu_sglang_hipblaslt_fp8_bmm",
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-DUSE_ROCM=1"],
        extra_ldflags=["-lhipblaslt"],
    )


def hipblaslt_fp8_bmm(x, weight, x_scale, weight_scale, out):
    load_hipblaslt_fp8_bmm().bmm(x, weight, x_scale, weight_scale, out)


def prepare(attn, *, is_hcu, use_aiter_gfx95):
    from sglang.srt.environ import envs

    attn.use_hipblaslt_fp8_v_up_proj = envs.SGLANG_HY4_FP8_V_UP_PROJ.get()
    if not attn.use_hipblaslt_fp8_v_up_proj:
        return
    if (
        not is_hcu or attn.w_vc.dtype != torch.bfloat16
        or attn.use_deep_gemm_bmm or use_aiter_gfx95
    ):
        raise ValueError(
            "HY4 FP8 V projection requires HCU BF16 MLA weights without AITER/DeepGEMM BMM"
        )
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
    from sglang.srt.layers.quantization.fp8_utils import input_to_float8

    load_hipblaslt_fp8_bmm()
    weight = attn.w_vc
    if not isinstance(attn.w_scale, (int, float)) or attn.w_scale != 1:
        weight = weight * attn.w_scale
    weight = weight.transpose(1, 2).contiguous()
    attn.w_vc_fp8, attn.w_vc_fp8_scale = input_to_float8(weight, dtype=fp8_dtype)


def project(attn, attn_output):
    if attn_output.dtype != torch.bfloat16:
        raise ValueError("HY4 FP8 V projection requires BF16 activations")
    output = attn_output.new_empty(
        (attn_output.shape[0], attn.num_local_heads, attn.v_head_dim)
    )
    if attn_output.shape[0] == 0:
        return output.flatten(1, 2)
    from sglang.kernels.ops.quantization.fp8_kernel import per_tensor_quant_mla_fp8

    scale = torch.zeros(1, dtype=torch.float32, device=attn_output.device)
    quant, scale = per_tensor_quant_mla_fp8(attn_output.contiguous(), scale)
    hipblaslt_fp8_bmm(
        quant, attn.w_vc_fp8, scale, attn.w_vc_fp8_scale, output,
    )
    return output.flatten(1, 2)
