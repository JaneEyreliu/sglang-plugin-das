from __future__ import annotations

_registered = False


def is_flash(config):
    return "HYV4FlashForConditionalGeneration" in (getattr(config, "architectures", None) or [])


def around_is_dsa(original, config):
    return is_flash(config) or original(config)


def around_shapes(original, self):
    if not is_flash(self.hf_config):
        return original(self)
    from copy import copy

    saved = self.hf_config
    self.hf_config = copy(saved)
    self.hf_config.architectures = ["HYV4ForCausalLM"]
    try:
        return original(self)
    finally:
        self.hf_config = saved


def flash_defaults(server_args, hf_config):
    from sglang.srt.arg_groups.model_override_base import resolving_view

    from sglang.srt.environ import envs

    cfg = resolving_view(server_args)
    if not envs.SGLANG_DSA_FUSE_TOPK.get():
        raise ValueError("HY4-Flash physical indices require SGLANG_DSA_FUSE_TOPK=1")
    if cfg.dsa_topk_backend != "sgl-kernel":
        raise ValueError("HY4-Flash HCU requires dsa_topk_backend=sgl-kernel")
    if cfg.enable_prefill_cp:
        from .cp import validate_profile
        validate_profile(cfg)
        envs.SGLANG_ENABLE_CP_V2.set(True)
    unsupported = {
        "speculative decoding": cfg.speculative_algorithm not in (None, "DFLY"),
        "DCP": cfg.dcp_size > 1,
        "DP attention": cfg.enable_dp_attention or cfg.attn_dp_size > 1,
        "HiCache": cfg.enable_hierarchical_cache,
        "HiSparse": cfg.enable_hisparse,
        "PD": cfg.disaggregation_mode != "null",
        "MoE A2A": cfg.moe_a2a_backend != "none" and not cfg.enable_prefill_cp,
        "PP": cfg.pp_size > 1,
        "waterfill": cfg.enable_waterfill,
        "two-batch overlap": cfg.enable_two_batch_overlap,
        "single-batch overlap": cfg.enable_single_batch_overlap,
    }
    rejected = [name for name, enabled in unsupported.items() if enabled]
    if rejected:
        raise ValueError(f"HY4-Flash HCU does not support {', '.join(rejected)}")
    if cfg.page_size not in (None, 64):
        raise ValueError("HY4-Flash HCU requires page_size=64")
    return {"page_size": 64}


def register_hooks():
    global _registered
    if _registered:
        return
    from hcu_sglang.speculative_dfly import register_hooks as register_dfly
    register_dfly()
    from transformers import AutoConfig
    from .config import HYV4FlashConfig, HYV4FlashTextConfig
    from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.arg_groups.model_override_base import register_model_override_predicate
    from sglang.srt.server_args import ServerArgs

    if "HYV4FlashForConditionalGeneration" not in ServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES:
        ServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES += ("HYV4FlashForConditionalGeneration",)

    for config in (HYV4FlashConfig, HYV4FlashTextConfig):
        AutoConfig.register(config.model_type, config, exist_ok=True)
        _CONFIG_REGISTRY[config.model_type] = config
    HookRegistry.register("sglang.srt.configs.model_config.is_deepseek_dsa", around_is_dsa, HookType.AROUND)
    HookRegistry.register("sglang.srt.configs.model_config.ModelConfig._derive_model_shapes", around_shapes, HookType.AROUND)
    register_model_override_predicate(lambda arch: arch == "HYV4FlashForConditionalGeneration")(flash_defaults)
    _registered = True
