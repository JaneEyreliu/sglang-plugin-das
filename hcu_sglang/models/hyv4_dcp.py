import torch


def load_sink(param, loaded_weight):
    if param.shape != loaded_weight.shape:
        raise ValueError("HYV4 sink weights must contain one logit per head")
    param.data.copy_(loaded_weight.float())


def configure_sink(layer, config, parallel):
    if not parallel.dcp_enabled:
        return
    layer.learnable_sink_param = torch.nn.Parameter(
        torch.empty(
            config.num_attention_heads,
            dtype=torch.float32,
            device=layer.learnable_sink_param.device,
        )
    )
    layer.learnable_sink_param.weight_loader = load_sink

    def local_sink():
        start = parallel.attn_tp_rank * layer.num_local_heads
        return layer.learnable_sink_param[start : start + layer.num_local_heads]

    def dcp_sink():
        heads = layer.num_local_heads * parallel.attn_dcp_size
        start = (parallel.attn_tp_rank // parallel.attn_dcp_size) * heads
        return layer.learnable_sink_param[start : start + heads]

    layer.attn_mqa.hyv4_sink_getter = local_sink
    layer.attn_mqa_for_dcp_decode.hyv4_sink_getter = dcp_sink
