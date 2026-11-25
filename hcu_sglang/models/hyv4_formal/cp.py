from sglang.srt.layers.cp.interleave import pad_local_rows

_registered = False


def validate_profile(cfg):
    if not (
        cfg.cp_strategy == "interleave"
        and cfg.tp_size == cfg.attn_cp_size == cfg.ep_size == 8
        and cfg.moe_dense_tp_size == 1
        and cfg.moe_a2a_backend == "deepep"
        and cfg.attn_dp_size == cfg.dp_size == cfg.pp_size == cfg.dcp_size == 1
        and cfg.speculative_algorithm is None
        and cfg.disaggregation_mode == "null"
        and not cfg.enable_dsa_cache_layer_split
    ):
        raise ValueError("HY4 formal prefill CP requires interleave TP8/CP8/EP8, DeepEP, dense TP1, without MTP, DP, PP, DCP, PD or cache sharding")


def shard_position_ids(original, self, positions, forward_batch):
    if positions.ndim != 2:
        return original(self, positions, forward_batch)
    metadata = forward_batch.attn_cp_metadata
    tokens_major = positions[:, :metadata.total_seq_lens].t()
    local_positions = self._interleave_shard(tokens_major).t().contiguous()
    return pad_local_rows(local_positions, metadata, dim=-1)


def register_hooks():
    global _registered
    if _registered:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.layers.cp.interleave.InterleaveCPStrategy.shard_position_ids", shard_position_ids, HookType.AROUND)
    _registered = True
