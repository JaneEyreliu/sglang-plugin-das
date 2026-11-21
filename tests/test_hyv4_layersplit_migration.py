import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


def source_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node, = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class LayerSplitBroadcastTests(unittest.TestCase):
    def test_all_ranks_drop_cached_target_and_index_before_next_draft(self):
        fn = source_function(
            "hcu_sglang/mem_cache/dsa_cache_layer_split.py",
            "reset_layer_broadcast_cache", {},
        )
        for rank in range(8):
            with self.subTest(rank=rank):
                pool = SimpleNamespace(
                    index_key_cache=SimpleNamespace(remote_layer_id=78 if rank else None),
                    remote_kv_layer_id=78 if rank else None,
                    _drain_pending_layer_broadcasts=Mock(),
                )
                fn(pool)
                pool._drain_pending_layer_broadcasts.assert_called_once_with(
                    discard_index=True, discard_main=True,
                )
                self.assertIsNone(pool.remote_kv_layer_id)
                self.assertIsNone(pool.index_key_cache.remote_layer_id)

    def test_nextn_resets_before_forward_on_owner_and_nonowner(self):
        calls = []
        parallel = SimpleNamespace(enable_dsa_cache_layer_split=True)
        pool = SimpleNamespace(reset_layer_broadcast_cache=lambda: calls.append("reset"))
        env = {
            "sglang.srt.runtime_context": module("runtime", get_parallel=lambda: parallel),
            "sglang.srt.model_executor.forward_context": module("context", get_token_to_kv_pool=lambda: pool),
            "hcu_sglang.models.hyv4_env_hooks": module(
                "hooks", around_model_forward=lambda *a, **kw: calls.append("forward") or "result",
            ),
        }
        fn = source_function(
            "hcu_sglang/models/hyv4_nextn.py", "around_nextn_model_forward",
            {"is_hcu": lambda: True},
        )
        with patch.dict(sys.modules, env):
            for rank in range(8):
                calls.clear()
                self.assertEqual(fn(Mock(), SimpleNamespace(alt_stream=None), None, None, None), "result")
                self.assertEqual(calls, ["reset", "forward"])
            parallel.enable_dsa_cache_layer_split = False
            calls.clear()
            fn(Mock(), SimpleNamespace(alt_stream=None), None, None, None)
            self.assertEqual(calls, ["forward"])


class MooncakeNamespaceTests(unittest.TestCase):
    def test_cp_namespaces_preserve_pipeline_and_unsharded_keys(self):
        fn = source_function(
            "hcu_sglang/mem_cache/cpu_cache.py", "_set_mooncake_cp_namespace", {},
        )
        parallel = SimpleNamespace(enable_dsa_cache_layer_split=True)
        with patch.dict(sys.modules, {
            "sglang.srt.runtime_context": module("runtime", get_parallel=lambda: parallel),
        }):
            for pp in (False, True):
                suffixes = set()
                for rank in range(8):
                    store = SimpleNamespace(
                        attn_cp_size=8, attn_cp_rank=rank, enable_pp=pp,
                        mla_suffix="2" if pp else "",
                    )
                    fn(store)
                    self.assertTrue(store.cp_layer_split)
                    self.assertEqual(store.mla_suffix, f"2_cp{rank}" if pp else f"cp{rank}")
                    suffixes.add(store.mla_suffix)
                self.assertEqual(len(suffixes), 8)
            for size, enabled in ((1, True), (8, False)):
                parallel.enable_dsa_cache_layer_split = enabled
                store = SimpleNamespace(
                    attn_cp_size=size, attn_cp_rank=0, enable_pp=True, mla_suffix="2",
                )
                fn(store)
                self.assertFalse(store.cp_layer_split)
                self.assertEqual(store.mla_suffix, "2")


class StagedBackupTests(unittest.TestCase):
    def setUp(self):
        self.flag = SimpleNamespace(get=lambda: True)
        self.tensor = Mock(side_effect=lambda x, **kw: (x, kw))
        self.transfer = Mock()
        env = {
            "hcu_sglang.utils.detection": module("detection", is_hcu=lambda: True),
            "sglang.srt.environ": module("environ", envs=SimpleNamespace(
                SGLANG_OPT_HICACHE_STAGED_LAYER_SPLIT_BACKUP=self.flag,
            )),
            "torch": module("torch", tensor=self.tensor, uint64="uint64"),
            "sglang.srt.mem_cache.pool_host.mla": module(
                "mla", jit_transfer_hicache_all_layer_mla_staged_lf_pf=self.transfer,
                maybe_dcp_kernel_indices=lambda indices, size, rank: indices,
            ),
        }
        self.patch = patch.dict(sys.modules, env)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        spec = importlib.util.spec_from_file_location(
            "staged_backup_test", ROOT / "hcu_sglang/mem_cache/hyv4_staged_backup.py",
        )
        self.impl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.impl)
        buffers = [SimpleNamespace(data_ptr=lambda i=i: 1000 + i) for i in range(4)]
        self.device = SimpleNamespace(kv_buffer=buffers, device="cuda:0")
        self.pool = SimpleNamespace(
            device_pool=self.device,
            _is_device_layer_sharded=lambda p: True,
            _owned_device_layer_ids=lambda p: [1, 3],
            mtp_draft_device_pools=[SimpleNamespace(kv_buffer=[buffers[0]])],
            can_use_write_back_jit=True, layer_num=3,
            dcp_size=1, dcp_rank=0, staging_buffer=object(), kv_buffer=object(), page_size=64,
        )

    def test_pointer_order_preserves_owned_layers_then_packed_draft(self):
        original = Mock(return_value="initialized")
        self.assertEqual(self.impl.around_host_init(original, self.pool), "initialized")
        self.tensor.assert_called_once_with([1001, 1003, 1000], dtype="uint64", device="cuda:0")
        original.assert_called_once_with(self.pool)

    def test_disabled_and_unavailable_kernel_keep_original_backup(self):
        for enabled, kernel, sharded in [(False, True, True), (True, False, True), (True, True, False)]:
            with self.subTest(enabled=enabled, kernel=kernel, sharded=sharded):
                self.flag.get = lambda: enabled
                self.pool.can_use_write_back_jit = kernel
                self.pool._is_device_layer_sharded = lambda p: sharded
                self.impl.around_host_init(Mock(), self.pool)
                original = Mock(return_value="fallback")
                self.assertEqual(self.impl.around_backup(original, self.pool, self.device, "host", "device", "kernel"), "fallback")
                self.transfer.assert_not_called()
                original.assert_called_once_with(self.pool, self.device, "host", "device", "kernel")

    def test_backup_uses_host_destination_indices_and_device_source_indices(self):
        self.impl.around_host_init(Mock(), self.pool)
        host, device = Mock(), Mock()
        original = Mock()
        self.impl.around_backup(original, self.pool, self.device, host, device, "kernel")
        self.transfer.assert_called_once_with(
            ptr_src=self.pool._layer_split_staged_src_ptrs,
            src_indices=device.to.return_value, dst_indices=host.cpu.return_value,
            staging=self.pool.staging_buffer, dst=self.pool.kv_buffer, page_size=64,
        )
        device.to.assert_called_once_with("cuda:0")
        host.cpu.assert_called_once_with()
        original.assert_not_called()

    def test_rejects_mismatched_layer_axis_before_transfer(self):
        self.pool.layer_num = 4
        with self.assertRaisesRegex(ValueError, "expects 4 layers"):
            self.impl.around_host_init(Mock(), self.pool)
        self.transfer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
