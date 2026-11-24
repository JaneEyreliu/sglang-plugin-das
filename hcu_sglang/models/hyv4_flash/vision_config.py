from transformers.configuration_utils import PreTrainedConfig

class HYV4VLVisionConfig(PreTrainedConfig):
    model_type = "hy_v4_vl_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        spatial_patch_size=1,
        temporal_patch_size=1,
        out_hidden_size=None,
        hidden_act="gelu",
        rms_norm_eps=1e-5,
        max_image_size=2048,
        interpolate_mode="bilinear",
        cat_extra_token=0,
        perceive_pre_norm=True,
        perceive_post_norm=True,
        projector_type=None,
        perceive_intermediate_size=None,
        use_image_newline_token=True,
        use_2d_rope=False,
        rope_2d_theta=10000.0,
        rope_2d_percent=1.0,
        img_max_token_num=None,
        max_vit_seq_len=None,
        add_patchemb_bias=True,
        video_spatial_compression=False,
        video_temporal_compression=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_image_size = max_image_size
        self.interpolate_mode = interpolate_mode
        self.cat_extra_token = cat_extra_token
        self.perceive_pre_norm = perceive_pre_norm
        self.perceive_post_norm = perceive_post_norm
        self.projector_type = projector_type
        self.perceive_intermediate_size = perceive_intermediate_size
        self.use_image_newline_token = use_image_newline_token
        self.use_2d_rope = use_2d_rope
        self.rope_2d_theta = rope_2d_theta
        self.rope_2d_percent = rope_2d_percent
        self.img_max_token_num = img_max_token_num
        self.max_vit_seq_len = max_vit_seq_len
        self.add_patchemb_bias = add_patchemb_bias
        self.video_spatial_compression = video_spatial_compression
        self.video_temporal_compression = video_temporal_compression
        self._validate_hy_v4_vl_vision()

    def _validate_hy_v4_vl_vision(self):
        if self.projector_type not in (None, "rmsnorm_mlp_adapter"):
            raise ValueError(
                f"Unsupported HYV4 projector_type: {self.projector_type!r}"
            )
        if self.use_2d_rope and self.rope_2d_percent != 1.0:
            raise ValueError("HYV4 vision supports only full 2D RoPE")
        if not self.add_patchemb_bias:
            raise ValueError("HYV4 vision supports only a biased patch embedding")
        if self.video_spatial_compression or self.video_temporal_compression:
            raise ValueError("HYV4 vision does not support video token compression")
