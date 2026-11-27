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
    
    # PD pair validated on HCU: CP8/EP8 prefill + a CP=1 decode worker.
    # Decode runs either the attention DP8 profile or TP-only MoE (--ep 1, no
    # A2A backend); plain TP8/EP8 DeepEP without DP reaches the terminal IHC
    # head as one stream (hidden) and cannot capture its decode graph.
    # DP8 decode supports the decode CUDA graph.
    pd_mode = cfg.disaggregation_mode in ("prefill", "decode")
    if pd_mode and cfg.disaggregation_transfer_backend != "mooncake":
        raise ValueError("HY4 formal HCU PD requires the mooncake transfer backend")
    if cfg.disaggregation_mode == "decode" and (
        not dp_enabled
        and (cfg.ep_size != 1 or cfg.moe_a2a_backend != "none")
    ):
        raise ValueError(
            "HY4 formal HCU PD decode requires either --ep 1 without a MoE A2A "
            "backend, or the attention DP8 profile (--attn-dp-size 8 --ep 8 "
            "--moe-a2a-backend deepep): the TP8/EP8 DeepEP profile without "
            "attention DP cannot capture its IHC head"
        )

    unsupported = {
        "speculative decoding": cfg.speculative_algorithm not in (None, "DFLY"),
        "DCP": cfg.dcp_size > 1,
        "HiCache": cfg.enable_hierarchical_cache,
        "HiSparse": cfg.enable_hisparse,
        "PD": cfg.disaggregation_mode not in ("null", "prefill", "decode"),
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

def around_language_model_only(original, server_args):
    """Run the upstream text-only checks with PD neutralized for formal HY4.

    The formal checkpoint is a VL checkpoint served text-only, so it needs
    --language-model-only. Upstream admits that flag together with PD only for
    the ``deepseek_v41`` text-only profile; the formal HY4 PD path has the same
    contract (text-only standard KV transfer), so keep the remaining upstream
    checks and let the real disaggregation mode stand.
    """
    from copy import copy

    from sglang.srt.arg_groups.model_override_base import (
        model_config_of,
        resolving_view,
    )

    cfg = resolving_view(server_args)
    if not (
        cfg.language_model_only
        and cfg.disaggregation_mode in ("prefill", "decode")
        and is_formal(model_config_of(server_args).hf_config)
    ):
        return original(server_args)

    proxy = copy(server_args)
    proxy.disaggregation_mode = "null"
    overrides = getattr(proxy, "_resolved_overrides", None)
    if overrides:
        proxy._resolved_overrides = [
            (
                source,
                {
                    key: value
                    for key, value in declared.items()
                    if key != "disaggregation_mode"
                },
            )
            for source, declared in overrides
        ]
    return original(proxy)

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
    HookRegistry.register("sglang.srt.arg_groups.model_hook.handle_language_model_only", around_language_model_only, HookType.AROUND)
    register_model_override_predicate(lambda arch: arch == "HYV4VLForConditionalGeneration")(formal_defaults)
    _registered = True
