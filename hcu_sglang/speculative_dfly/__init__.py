from sglang.srt.speculative.spec_info import CustomSpecAlgo, SpeculativeAlgorithm

_registered = False


class DFlyAlgorithm(CustomSpecAlgo):
    def need_topk(self):
        return False

    def is_dflash_family(self):
        return True

    def supports_target_verify_for_draft(self):
        return True

    def supports_prefill_shared_read_done(self):
        return True

    def carries_draft_hidden_states(self):
        return False

    def create_future_map(self, *args, **kwargs):
        return SpeculativeAlgorithm.create_future_map(self, *args, **kwargs)

    def handle_server_args(self, server_args):
        from sglang.srt.arg_groups.overrides import resolving_view
        from sglang.srt.utils.hf_transformers_utils import get_config

        cfg = resolving_view(server_args)
        if cfg.pp_size != 1 or cfg.attn_cp_size != 1 or cfg.dcp_size != 1:
            raise ValueError("DFLY requires PP=CP=DCP=1")
        hf = None
        if cfg.enable_dp_attention or cfg.attn_dp_size != 1:
            from hcu_sglang.models.hyv4_dp import validate_profile
            hf = get_config(cfg.model_path, trust_remote_code=cfg.trust_remote_code, revision=cfg.revision)
            architectures = getattr(hf, "architectures", None) or []
            if not any(arch in architectures for arch in ("HYV4VLForConditionalGeneration", "HYV4FlashForConditionalGeneration")):
                raise ValueError("DFLY attention DP requires HY4 or HY4-Flash")
            validate_profile(cfg, flash="HYV4FlashForConditionalGeneration" in architectures)
        if cfg.disaggregation_mode != "null" or cfg.enable_lora:
            raise ValueError("DFLY does not support PD or LoRA")
        if cfg.speculative_draft_model_path not in (None, cfg.model_path):
            raise ValueError("DFLY requires the draft bundled with the target checkpoint")
        if cfg.speculative_draft_attention_backend is not None:
            raise ValueError("DFLY uses the target attention backend")
        for name in ("speculative_num_steps", "speculative_eagle_topk"):
            if getattr(cfg, name) not in (None, 1):
                raise ValueError(f"DFLY requires {name}=1")
        width = cfg.speculative_num_draft_tokens
        if width is None:
            if hf is None:
                hf = get_config(cfg.model_path, trust_remote_code=cfg.trust_remote_code, revision=cfg.revision)
            width = getattr(hf, "dfly_block_size", 6)
        if not 2 <= int(width) <= 17:
            raise ValueError("DFLY block width must be between 2 and 17")
        server_args.speculative_num_steps = 1
        server_args.speculative_eagle_topk = 1
        server_args.speculative_num_draft_tokens = int(width)
        server_args.enable_mixed_chunk = False


def validate_request(original, req, enable_overlap, *args, **kwargs):
    from sglang.srt.runtime_context import get_spec
    if get_spec().speculative_algorithm == "DFLY":
        if req.return_logprob or req.sampling_params.top_k != 1 or req.grammar is not None:
            return "DFLY currently requires greedy decoding without grammar or logprobs"
    return original(req, enable_overlap, *args, **kwargs)


def resolve_aux_hidden_state(original, **kwargs):
    if getattr(kwargs["spec_algorithm"], "name", None) == "DFLY":
        from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import SpecAuxHiddenStateConfig
        return SpecAuxHiddenStateConfig()
    return original(**kwargs)


def register_hooks():
    global _registered
    if _registered:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    @SpeculativeAlgorithm.register("DFLY", supports_overlap=True, spec_class=DFlyAlgorithm)
    def worker_factory(server_args):
        from .worker import DFlyWorkerV2
        return DFlyWorkerV2

    HookRegistry.register("sglang.srt.speculative.dflash_utils.validate_dflash_request", validate_request, HookType.AROUND)
    HookRegistry.register("sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state.resolve_spec_aux_hidden_state_config", resolve_aux_hidden_state, HookType.AROUND)
    _registered = True
