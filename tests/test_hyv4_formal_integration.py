import os
import types
import unittest

MODEL = '/llm-models3/Hy4-Channel-INT4-w4a8'


def arguments(extra=()):
    from sglang.srt.server_args import prepare_server_args
    return prepare_server_args(['--model-path', MODEL, '--tp', '8', '--ep', '1', '--quantization', 'slimquant_w4a8_marlin', '--attention-backend', 'dsa', '--dsa-prefill-backend', 'flashmla_sparse', '--dsa-decode-backend', 'flashmla_kv', '--language-model-only', '--disable-prefill-cuda-graph', '--disable-cuda-graph', '--context-length', '8192', *extra])


@unittest.skipUnless(os.environ.get('HCU_PLUGIN_RUN_GPU_TESTS') == '1', 'HCU integration opt-in')
class FormalIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hcu_sglang.compat.aiter import bootstrap_aiter_compat
        bootstrap_aiter_compat()
        from sglang.srt.plugins import load_plugins
        load_plugins()

    def test_public_resolution_registry_and_quantization(self):
        from sglang.srt.configs.model_config import ModelConfig, is_deepseek_dsa
        from sglang.srt.models.registry import ModelRegistry
        from sglang.srt.arg_groups.model_override_base import resolved_view
        args = arguments()
        args.resolve_once()
        config = ModelConfig.from_server_args(args)
        self.assertEqual(config.hf_config.architectures, ['HYV4VLForConditionalGeneration'])
        self.assertEqual((config.hidden_size, config.num_hidden_layers, config.kv_lora_rank, config.qk_rope_head_dim, config.index_head_dim), (6144, 78, 512, 64, 128))
        self.assertTrue(config.is_lm_only)
        self.assertTrue(is_deepseek_dsa(config.hf_config))
        self.assertTrue(is_deepseek_dsa(config.hf_text_config))
        self.assertEqual(config.attention_arch.name, 'MLA')
        self.assertEqual(config.quantization, 'slimquant_w4a8_marlin')
        self.assertEqual(resolved_view(args).page_size, 64)
        cls, arch = ModelRegistry.resolve_model_cls(config.hf_config.architectures)
        self.assertEqual(cls.__module__, 'hcu_sglang.models.hyv4_formal.model')

    def test_scope_and_effective_fallback(self):
        from hcu_sglang.models.hyv4_formal import around_shapes, around_is_dsa
        seen = []
        original_config = types.SimpleNamespace(architectures=['HYV4VLForConditionalGeneration'])
        owner = types.SimpleNamespace(hf_config=original_config)
        def fail(obj):
            self.assertEqual(obj.hf_config.architectures, ['HYV4ForCausalLM'])
            raise RuntimeError('sentinel')
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            around_shapes(fail, owner)
        self.assertIs(owner.hf_config, original_config)
        other = types.SimpleNamespace(architectures=['HYV4FlashForConditionalGeneration'])
        self.assertEqual(around_is_dsa(lambda cfg: seen.append(cfg) or 'baseline', other), 'baseline')
        self.assertEqual(seen, [other])
        for flags in [('--pp-size', '2'), ('--dcp-size', '2'), ('--enable-hisparse',), ('--enable-two-batch-overlap',), ('--page-size', '1')]:
            with self.subTest(flags=flags), self.assertRaises((ValueError, AssertionError)):
                arguments(flags).resolve_once()

    def test_weight_mapping_and_indexer_layout(self):
        import torch
        from hcu_sglang.models.hyv4_formal.config import HYV4VLConfig
        from hcu_sglang.models.hyv4_formal.model import HYV4VLForConditionalGeneration
        from hcu_sglang.models.hyv4_formal.text import permute_hyv4_indexer_weight, normalize_hyv4_weight_scale
        from hcu_sglang.layers.quantization.slimquant_w4a8_marlin import SlimQuantW4A8Int8MarlinConfig
        cfg = HYV4VLConfig.from_pretrained(MODEL)
        quant_dict = dict(cfg.quantization_config, hf_config=cfg)
        quant = SlimQuantW4A8Int8MarlinConfig.from_config(quant_dict)
        self.assertEqual(quant.checkpoint_format, 'hy4_compressed_tensors_w4a8')
        weight = torch.arange(128, dtype=torch.float32)
        actual = permute_hyv4_indexer_weight('model.layers.0.self_attn.indexer.k_norm.weight', weight, cfg.text_config)
        torch.testing.assert_close(actual, torch.cat((weight[64:], weight[:64])), rtol=0, atol=0)
        scale = torch.ones(128, 1)
        name = 'model.layers.0.self_attn.indexer.wk.weight_scale'
        self.assertEqual(normalize_hyv4_weight_scale(name, scale, quant)[0], name)
        mapper = HYV4VLForConditionalGeneration.hf_to_sglang_mapper
        pairs = list(mapper.apply([('model.layers.1.mlp.gate.wg.weight', weight), ('lm_head.weight', weight), ('vit.vit.patch.weight', weight)]))
        self.assertEqual([p[0] for p in pairs], ['language_model.model.layers.1.mlp.gate.weight', 'language_model.lm_head.weight', 'visual.patch.weight'])

    def test_formal_ihc_real_forward(self):
        import torch
        from hcu_sglang.models.hyv4_formal.config import HYV4Config
        from hcu_sglang.models.hyv4_formal.text import HYV4HCPreLayer, HYV4HCLayer
        torch.cuda.set_device(0)
        torch.manual_seed(1124)
        cfg = HYV4Config(hidden_size=6144, hc_mult=4)
        layer = HYV4HCPreLayer(cfg, 'test').cuda()
        with torch.no_grad():
            layer.hc_fn.weight.normal_(0, 0.003)
            layer.hc_scale.fill_(0.2)
            layer.hc_base.normal_(0, 0.2)
        for count in [0, 1, 4, 65]:
            x = torch.randn(count, 4, 6144, device='cuda', dtype=torch.bfloat16)
            actual, post = layer(x)
            flat = x.float().flatten(1)
            mix = (flat @ layer.hc_fn.weight.T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
            pre = torch.sigmoid(mix[:, :4] * layer.hc_scale[0] + layer.hc_base[:4]) + cfg.hc_eps
            expected = (x.float() * pre[:, :, None]).sum(1).to(x.dtype)
            post_ref = cfg.hc_magnitude * torch.sigmoid(mix[:, 4:] * layer.hc_scale[1] + layer.hc_base[4:]) + cfg.hc_eps
            torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.008)
            torch.testing.assert_close(post, post_ref, rtol=3e-5, atol=3e-5)
            self.assertFalse(layer._fused_ihc_pre_disabled)
            residual = HYV4HCLayer(cfg, 'post').cuda()
            out = torch.randn(count, 6144, device='cuda', dtype=torch.bfloat16)
            result = residual.post(out, x, post)
            torch.testing.assert_close(result, (x.float() + out.float()[:, None] * post[:, :, None]).to(x.dtype), rtol=0, atol=0)
            self.assertFalse(residual._fused_ihc_post_disabled)


if __name__ == '__main__':
    unittest.main()
