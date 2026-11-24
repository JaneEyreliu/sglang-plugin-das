import types
import unittest

import os

MODEL=os.environ.get('HY4_FLASH_MODEL','/llm-models3/Hy4-flash-Channel-INT4-w4a8')


def arguments(extra=()):
    from sglang.srt.server_args import prepare_server_args
    return prepare_server_args(['--model-path',MODEL,'--tp','8','--quantization','slimquant_w4a8_marlin','--attention-backend','dsa','--dsa-prefill-backend','flashmla_sparse','--dsa-decode-backend','flashmla_kv','--language-model-only','--disable-prefill-cuda-graph','--context-length','8192',*extra])


@unittest.skipUnless(os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") == "1", "HCU integration opt-in")
class FlashIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global ModelConfig, is_deepseek_dsa, ModelRegistry
        from hcu_sglang.compat.aiter import bootstrap_aiter_compat
        bootstrap_aiter_compat()
        from sglang.srt.plugins import load_plugins
        load_plugins()
        from sglang.srt.configs.model_config import ModelConfig, is_deepseek_dsa
        from sglang.srt.models.registry import ModelRegistry

    def test_public_resolution_and_geometry(self):
        args=arguments()
        args.resolve_once()
        config=ModelConfig.from_server_args(args)
        from sglang.srt.arg_groups.model_override_base import resolved_view
        self.assertEqual(resolved_view(args).page_size,64)
        self.assertEqual(config.kv_lora_rank,512)
        self.assertEqual(config.qk_rope_head_dim,64)
        self.assertEqual(config.index_head_dim,128)
        self.assertEqual(config.attention_arch.name,'MLA')
        self.assertTrue(config.is_lm_only)
        self.assertEqual(config.hf_config.architectures,['HYV4FlashForConditionalGeneration'])
        self.assertTrue(is_deepseek_dsa(config.hf_config))
        self.assertTrue(is_deepseek_dsa(config.hf_text_config))
        cls,arch=ModelRegistry.resolve_model_cls(config.hf_config.architectures)
        self.assertEqual(cls.__module__,'hcu_sglang.models.hyv4_flash.model')

    def test_fp8_index_cache_allocation_and_budget(self):
        import torch
        from sglang.srt.runtime_context import get_context
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
        from hcu_sglang.mem_cache.glm5_index_cache import around_build_pool, fp8_index_cache_active, uses_fp8_index_cache
        from hcu_sglang.mem_cache.dsa import around_compute_dsa_indexer_cell_size
        args=arguments()
        args.resolve_once()
        get_context().set_server_args(args)
        config=ModelConfig.from_server_args(args)
        owner=types.SimpleNamespace(model_config=config)
        self.assertFalse(fp8_index_cache_active())
        def construct(configurator):
            self.assertTrue(fp8_index_cache_active())
            return DSATokenToKVPool(size=128,page_size=64,kv_lora_rank=512,dtype=torch.bfloat16,qk_rope_head_dim=64,layer_num=1,device='cuda',index_head_dim=128,enable_memory_saver=False,kv_cache_dim=576)
        pool=around_build_pool(construct,owner)
        self.assertFalse(fp8_index_cache_active())
        self.assertTrue(pool.use_fp8_index_k_cache)
        buf=pool.get_index_k_with_scale_buffer(0)
        self.assertEqual(buf.dtype,torch.uint8)
        self.assertEqual(buf.shape[-1],64*132)
        self.assertEqual(pool.get_key_buffer(0).dtype,torch.bfloat16)
        kvc=types.SimpleNamespace(kv_cache_dtype=torch.bfloat16,page_size=64,model_config=config)
        seen=[]
        def budget(configurator,**kwargs):
            seen.append(kwargs)
            return 132*kwargs['num_layers']
        self.assertEqual(around_compute_dsa_indexer_cell_size(budget,owner,kvc=kvc,num_layers=42),132*42)
        self.assertEqual(len(seen),1)
        def fail(configurator):
            self.assertTrue(fp8_index_cache_active())
            raise RuntimeError('allocation sentinel')
        with self.assertRaisesRegex(RuntimeError,'allocation sentinel'):
            around_build_pool(fail,owner)
        self.assertFalse(fp8_index_cache_active())
        self.assertFalse(uses_fp8_index_cache(types.SimpleNamespace(architectures=['DeepseekV3ForCausalLM'])))

    def test_reject_unsupported(self):
        for flags in [('--dcp-size','2'),('--pp-size','2'),('--attn-dp-size','2'),('--enable-prefill-cp',),('--enable-hisparse',),('--enable-two-batch-overlap',),('--page-size','1')]:
            with self.subTest(flags=flags),self.assertRaises((ValueError,AssertionError)):
                arguments(flags).resolve_once()

    def test_unrelated_dsa(self):
        self.assertTrue(is_deepseek_dsa(types.SimpleNamespace(architectures=['DeepseekV3ForCausalLM'],index_topk=2048)))
        self.assertFalse(is_deepseek_dsa(types.SimpleNamespace(architectures=['LlamaForCausalLM'],index_topk=None)))

    def test_physical_indices_reach_decode_consumer(self):
        import torch
        from unittest.mock import patch
        from hcu_sglang.layers.attention.current_dsa.backend import DeepseekSparseAttnBackend
        backend=DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.forward_metadata=types.SimpleNamespace(page_table_1=torch.tensor([[192,64]],device='cuda'),dsa_cache_seqlens_int32=torch.tensor([2],device='cuda'))
        backend.dsa_decode_impl='flashmla_sparse'
        backend.dcp_size=1
        backend.hisparse_coordinator=None
        backend.use_fused_topk=True
        backend.dsa_topk_backend=types.SimpleNamespace(is_sgl_kernel=lambda:True)
        backend.token_to_kv_pool=types.SimpleNamespace(get_key_buffer=lambda _:torch.zeros((256,1,4),device='cuda'))
        backend._resolve_kpool_tail_backend=lambda indices,impl:impl
        backend._check_kpool_tail_backend=lambda *args:None
        indices=torch.full((1,2048),-1,dtype=torch.int32,device='cuda')
        indices[0,:2]=torch.tensor([193,65],device='cuda')
        q=torch.randn(1,1,4,device='cuda')
        layer=types.SimpleNamespace(is_cross_attention=False,layer_id=0,tp_q_head_num=1,head_dim=4,v_head_dim=2,scaling=0.5)
        captured=[]
        backend._forward_flashmla_sparse=lambda **kwargs:captured.append(kwargs) or q
        with patch('hcu_sglang.layers.attention.current_dsa.backend.transform_index_page_table_decode',side_effect=AssertionError('physical indices remapped')):
            actual=backend.forward_decode(q,None,None,layer,types.SimpleNamespace(),save_kv_cache=False,topk_indices=indices)
        self.assertIs(actual,q)
        self.assertIs(captured[0]['page_table_1'],indices)
        self.assertEqual(captured[0]['page_table_1'][0,:2].tolist(),[193,65])

    def test_reject_logical_index_consumer(self):
        from sglang.srt.environ import envs
        with envs.SGLANG_DSA_FUSE_TOPK.override(False):
            with self.assertRaisesRegex(ValueError,'physical indices'):
                arguments().resolve_once()


if __name__=='__main__':
    unittest.main()

