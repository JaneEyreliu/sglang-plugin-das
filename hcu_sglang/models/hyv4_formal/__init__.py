from __future__ import annotations

_registered = False


def is_formal(config):
    return "HYV4VLForConditionalGeneration" in (getattr(config, "architectures", None) or [])


def around_is_dsa(original, config):
    return is_formal(config) or original(config)


def around_shapes(original, self):
    if not is_formal(self.hf_config):
        return original(self)
    from copy import copy

    saved = self.hf_config
    self.hf_config = copy(saved)
    self.hf_config.architectures = ["HYV4ForCausalLM"]
    try:
        return original(self)
    finally:
        self.hf_config = saved


def formal_defaults(server_args, hf_config):
    from sglang.srt.arg_groups.model_override_base import resolving_view

    from sglang.srt.environ import envs

    cfg = resolving_view(server_args)
    if not cfg.language_model_only:
        raise ValueError("HY4 formal HCU currently requires language_model_only")
    if getattr(hf_config.text_config, "attn_post_proj_norm", False):
        raise ValueError("HY4 formal post-projection norm is not ported")
    if cfg.enable_prefill_cp:
        from .cp import validate_profile
        validate_profile(cfg)
        envs.SGLANG_ENABLE_CP_V2.set(True)
    from hcu_sglang.models import hyv4_dp
    dp_enabled = hyv4_dp.requested(cfg)
    if dp_enabled:
        hyv4_dp.validate_profile(cfg, flash=False)
    unsupported = {
        "speculative decoding": cfg.speculative_algorithm not in (None, "DFLY"),
        "DCP": cfg.dcp_size > 1,
        "HiCache": cfg.enable_hierarchical_cache,
        "HiSparse": cfg.enable_hisparse,
        "PD": cfg.disaggregation_mode != "null",
        "MoE A2A": cfg.moe_a2a_backend != "none" and not (cfg.enable_prefill_cp or dp_enabled),
        "PP": cfg.pp_size > 1,
        "waterfill": cfg.enable_waterfill,
        "two-batch overlap": cfg.enable_two_batch_overlap,
        "single-batch overlap": cfg.enable_single_batch_overlap,
    }
    rejected = [name for name, enabled in unsupported.items() if enabled]
    if rejected:
        raise ValueError(f"HY4 formal HCU does not support {', '.join(rejected)}")
    if cfg.page_size not in (None, 64):
        raise ValueError("HY4 formal HCU requires page_size=64")
    return {"page_size": 64}


def register_hooks():
    global _registered
    if _registered:
        return
    from .cp import register_hooks as register_cp
    register_cp()
    from transformers import AutoConfig
    from .config import HYV4VLConfig
    from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.arg_groups.model_override_base import register_model_override_predicate
    from sglang.srt.server_args import ServerArgs

    if "HYV4VLForConditionalGeneration" not in ServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES:
        ServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES += ("HYV4VLForConditionalGeneration",)

    for config in (HYV4VLConfig,):
        AutoConfig.register(config.model_type, config, exist_ok=True)
        _CONFIG_REGISTRY[config.model_type] = config
    HookRegistry.register("sglang.srt.configs.model_config.is_deepseek_dsa", around_is_dsa, HookType.AROUND)
    HookRegistry.register("sglang.srt.configs.model_config.ModelConfig._derive_model_shapes", around_shapes, HookType.AROUND)
    register_model_override_predicate(lambda arch: arch == "HYV4VLForConditionalGeneration")(formal_defaults)
    _registered = True
