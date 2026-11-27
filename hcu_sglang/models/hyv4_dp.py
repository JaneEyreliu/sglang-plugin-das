def requested(cfg):
    return cfg.enable_dp_attention or cfg.attn_dp_size > 1


def active():
    from sglang.srt.runtime_context import get_parallel
    return get_parallel().attn_dp_enabled


def validate_profile(cfg, *, flash=False):
    capacity = getattr(cfg, "max_running_requests", None)
    if capacity is not None and capacity < 8:
        raise ValueError("HY4 DP8 requires max_running_requests >= 8 before per-rank division")
    if not (
        cfg.attn_dp_size == 8
        and cfg.tp_size == cfg.ep_size == 8
        and cfg.dp_size == 1
        and cfg.moe_dense_tp_size == 1
        and cfg.moe_a2a_backend == "deepep"
        and cfg.attn_cp_size == cfg.dcp_size == cfg.pp_size == 1
        and not cfg.enable_prefill_cp
        and cfg.speculative_algorithm in (None, "DFLY")
        and cfg.disaggregation_mode == "null"
        and cfg.language_model_only
        and not cfg.enable_dsa_cache_layer_split
        and cfg.deepep_mode in (("normal",) if flash else ("normal", "low_latency"))
    ):
        raise ValueError("HY4 DP requires text-only TP8/DP8/EP8, DeepEP, dense TP1, CP1/DCP1/PP1, without PD or cache sharding; formal DP supports DeepEP normal/low_latency; Flash requires normal because native low-latency dispatch does not support its hidden size")
    if cfg.speculative_algorithm == "DFLY" and not (
        cfg.enable_dp_lm_head
        and cfg.kv_cache_dtype in ("auto", "bfloat16")
        and cfg.dtype in ("auto", "bfloat16")
        and cfg.disable_cuda_graph
        and cfg.disable_prefill_cuda_graph
    ):
        raise ValueError("HY4 DP8 DFLY requires DP lm_head, BF16 target KV and eager execution")


def prepare(batch, rows):
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel
    parallel = get_parallel()
    counts = [None] * parallel.tp_size
    dist.all_gather_object(counts, rows, group=parallel.tp_group.cpu_group)
    batch.hy4_dp_token_counts = tuple(counts)


def gather(x, batch):
    import torch
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel
    parallel = get_parallel()
    counts = batch.hy4_dp_token_counts
    if counts[parallel.tp_rank] != x.shape[0]:
        raise ValueError("HY4 Engram local rows do not match the published DP layout")
    width = max(counts)
    if width == 0:
        return x.new_empty((0, *x.shape[1:]))
    local = x.new_zeros((width, *x.shape[1:]))
    local[:x.shape[0]].copy_(x)
    output = x.new_empty((parallel.tp_size * width, *x.shape[1:]))
    dist.all_gather_into_tensor(output, local, group=parallel.tp_group.device_group)
    return torch.cat([output[i * width:i * width + n] for i, n in enumerate(counts)], dim=0)


def shard(x, rows, batch):
    from sglang.srt.runtime_context import get_parallel
    rank = get_parallel().tp_rank
    counts = batch.hy4_dp_token_counts
    if counts[rank] != rows or sum(counts) != x.shape[0]:
        raise ValueError("HY4 Engram projection rows do not match the published DP layout")
    return x.narrow(0, sum(counts[:rank]), rows).contiguous()
