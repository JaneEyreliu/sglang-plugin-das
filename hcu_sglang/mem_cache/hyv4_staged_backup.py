from hcu_sglang.utils.detection import is_hcu


def around_host_init(original, pool, *args, **kwargs):
    result = original(pool, *args, **kwargs)
    pool._layer_split_staged_src_ptrs = None
    if getattr(pool, "_is_dummy", False):
        return result
    from sglang.srt.environ import envs

    if not (is_hcu() and envs.SGLANG_OPT_HICACHE_STAGED_LAYER_SPLIT_BACKUP.get()):
        return result
    if not pool._is_device_layer_sharded(pool.device_pool):
        return result
    if not pool.can_use_write_back_jit:
        return result
    import torch

    buffers = [
        pool.device_pool.kv_buffer[layer]
        for layer in pool._owned_device_layer_ids(pool.device_pool)
    ]
    buffers += [draft.kv_buffer[0] for draft in pool.mtp_draft_device_pools]
    if len(buffers) != pool.layer_num:
        raise ValueError(
            f"Layer-split staged backup expects {pool.layer_num} layers, got {len(buffers)}"
        )
    pool._layer_split_staged_src_ptrs = torch.tensor(
        [buffer.data_ptr() for buffer in buffers],
        dtype=torch.uint64,
        device=pool.device_pool.device,
    )
    return result


def around_backup(original, pool, device_pool, host_indices, device_indices, io_backend):
    ptrs = getattr(pool, "_layer_split_staged_src_ptrs", None)
    if ptrs is None or io_backend != "kernel":
        return original(pool, device_pool, host_indices, device_indices, io_backend)
    if device_pool is not pool.device_pool:
        raise ValueError("Staged backup pointer table belongs to a different device pool")
    assert not getattr(pool, "_is_dummy", False), "backup on a dummy host pool"
    from sglang.srt.mem_cache.pool_host.mla import (
        jit_transfer_hicache_all_layer_mla_staged_lf_pf,
        maybe_dcp_kernel_indices,
    )

    host_indices = maybe_dcp_kernel_indices(host_indices, pool.dcp_size, pool.dcp_rank)
    device_indices = maybe_dcp_kernel_indices(device_indices, pool.dcp_size, pool.dcp_rank)
    return jit_transfer_hicache_all_layer_mla_staged_lf_pf(
        ptr_src=ptrs,
        src_indices=device_indices.to(pool.device_pool.device),
        dst_indices=host_indices.cpu(),
        staging=pool.staging_buffer,
        dst=pool.kv_buffer,
        page_size=pool.page_size,
    )


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    prefix = "sglang.srt.mem_cache.pool_host.mla.MLATokenToKVPoolHost."
    HookRegistry.register(prefix + "__init__", around_host_init, HookType.AROUND)
    HookRegistry.register(prefix + "backup_from_device_all_layer", around_backup, HookType.AROUND)
