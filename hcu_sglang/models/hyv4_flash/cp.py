from types import SimpleNamespace


def validate_profile(cfg):
    if not (
        cfg.cp_strategy == "interleave"
        and cfg.tp_size == cfg.attn_cp_size == cfg.ep_size == 8
        and cfg.moe_a2a_backend == "deepep"
        and cfg.moe_dense_tp_size == 8
        and cfg.deepep_mode == "normal"
        and cfg.attn_dp_size == cfg.dp_size == cfg.pp_size == cfg.dcp_size == 1
        and cfg.speculative_algorithm is None
        and cfg.disaggregation_mode == "null"
        and cfg.language_model_only
        and not cfg.enable_dsa_cache_layer_split
    ):
        raise ValueError("HY4-Flash CP requires text-only interleave TP8/CP8/EP8, DeepEP normal, dense TP8, without MTP, DP, PP, DCP, PD or cache sharding")


def active(batch):
    if batch is None:
        return False
    from sglang.srt.layers.cp.utils import is_cp_active
    return is_cp_active(batch)


def gather(x, batch):
    from sglang.srt.layers.cp.utils import cp_materialize_global_token_order
    return cp_materialize_global_token_order(x, batch)


def shard(x, batch):
    from sglang.srt.layers.cp.utils import cp_shard_hidden_states
    return cp_shard_hidden_states(x, batch)


def full_indexer_metadata(batch, metadata):
    from dataclasses import replace
    import torch
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
    from sglang.srt.layers.cp.utils import get_cp_strategy
    from sglang.srt.model_executor.forward_context import get_req_to_token_pool
    table = get_req_to_token_pool().req_to_token[batch.req_pool_indices]
    attn_metadata = metadata.attn_metadata
    if metadata.topk_transform_method == TopkTransformMethod.RAGGED:
        row_to_batch = torch.repeat_interleave(
            torch.arange(batch.batch_size, device=batch.extend_seq_lens.device),
            batch.extend_seq_lens.to(torch.int64),
        )
        local_rows = get_cp_strategy().shard_local_tokens(row_to_batch)
        attn_metadata = replace(
            attn_metadata,
            topk_indices_offset=expand_ragged_offsets(
                attn_metadata.topk_indices_offset,
                row_to_batch, local_rows, batch.batch_size,
            ),
        )
    return SimpleNamespace(
        attn_metadata=attn_metadata,
        topk_transform_method=metadata.topk_transform_method,
        get_page_table_1=lambda:table,
        get_page_table_64=lambda:table[:,::64] // 64,
        get_seqlens_expanded=lambda:batch.positions + 1,
    )


def expand_ragged_offsets(offsets, full_rows, local_rows, batch_size):
    if offsets.numel() != local_rows.numel():
        raise ValueError("HY4-Flash CP ragged offsets do not match local query rows")
    per_request = offsets.new_zeros(batch_size)
    per_request.scatter_(0, local_rows, offsets)
    return per_request[full_rows]


def run_indexer(fn, x, q_lora, positions, batch, layer_id, return_indices):
    full_x = gather(x, batch)
    full_q = gather(q_lora, batch)
    previous = getattr(batch, "hy4f_cp_full_indexer", False)
    batch.hy4f_cp_full_indexer = True
    try:
        indices = fn(full_x, full_q, batch.positions, batch, layer_id, return_indices)
    finally:
        batch.hy4f_cp_full_indexer = previous
    return shard(indices, batch)
