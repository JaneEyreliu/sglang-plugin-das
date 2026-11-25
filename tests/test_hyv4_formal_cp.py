import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import unittest
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]


def load_policy():
    name = "sglang.srt.layers.cp.interleave"
    stub = ModuleType(name)
    stub.pad_local_rows = lambda x, metadata, dim: x
    spec = importlib.util.spec_from_file_location("hy4_cp_policy_test", _ROOT / "hcu_sglang/models/hyv4_formal/cp.py")
    mod = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: stub}):
        spec.loader.exec_module(mod)
    return mod


class FormalCPTests(unittest.TestCase):
    def test_profile_rejects_speculation_and_incompatible_layouts(self):
        mod = load_policy()
        cfg = dict(cp_strategy="interleave", tp_size=8, attn_cp_size=8, ep_size=8,
                   moe_dense_tp_size=1, moe_a2a_backend="deepep", attn_dp_size=1,
                   dp_size=1, pp_size=1, dcp_size=1, speculative_algorithm=None,
                   disaggregation_mode="null", enable_dsa_cache_layer_split=False)
        mod.validate_profile(SimpleNamespace(**cfg))
        for key, value in [("cp_strategy", "zigzag"), ("speculative_algorithm", "DFLY"),
                           ("ep_size", 4), ("moe_dense_tp_size", 8), ("dcp_size", 2),
                           ("moe_a2a_backend", "none")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                mod.validate_profile(SimpleNamespace(**dict(cfg, **{key:value})))

    def test_mrope_shards_token_axis_and_retains_all_three_rows(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is required for tensor layout validation")
        mod = load_policy()
        positions = torch.arange(3 * 17).reshape(3, 17)
        batch = SimpleNamespace(attn_cp_metadata=SimpleNamespace(total_seq_lens=17))
        for rank in range(8):
            strategy = SimpleNamespace(_interleave_shard=lambda x, rank=rank:x[rank::8])
            actual = mod.shard_position_ids(lambda *a:None, strategy, positions, batch)
            torch.testing.assert_close(actual, positions[:,rank::8])
        marker = object()
        self.assertIs(mod.shard_position_ids(lambda *a:marker, None, positions[0], batch), marker)

    def test_wrapper_exposes_original_modules_without_registering_duplicates(self):
        tree = ast.parse((_ROOT / "hcu_sglang/models/hyv4_formal/model.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=="HYV4VLForConditionalGeneration")
        methods = [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ("model","lm_head","logits_processor")]
        isolated = ast.ClassDef(name="Wrapper", bases=[], keywords=[], body=methods, decorator_list=[])
        scope={}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[isolated], type_ignores=[])),"wrapper","exec"),scope)
        obj=scope["Wrapper"](); obj.language_model=SimpleNamespace(model=object(),lm_head=object(),logits_processor=object())
        for name in ("model","lm_head","logits_processor"):
            self.assertIs(getattr(obj,name),getattr(obj.language_model,name))


if __name__ == "__main__":
    unittest.main()
