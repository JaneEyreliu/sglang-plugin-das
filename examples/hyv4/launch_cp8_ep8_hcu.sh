#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || ( "$1" != flash && "$1" != formal ) ]]; then
  echo "Usage: bash $0 {flash|formal} MODEL_PATH [sglang arguments...]" >&2
  exit 2
fi
variant="$1"
model_path="$2"
shift 2
plugin_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
workspace_root="${HY4_WORKSPACE:-$(dirname -- "$plugin_root")}"
adapter_root="${ADAPTER_DAS_ROOT:-$workspace_root/adapter-das}"
sglang_root="${SGLANG_ROOT:-$workspace_root/sglang}"
export PYTHONPATH="$adapter_root:$sglang_root/python:$plugin_root${PYTHONPATH:+:$PYTHONPATH}"
export HCU_SLIMQUANT=0
export HCU_DEEPEP_BRIDGE=0
export SGLANG_PLUGINS=hcu
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export SGLANG_ENABLE_CP_V2=1
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export SGLANG_USE_W4A8_MASKED_HIPC=1
unset SGLANG_W4A8_TPMOE_BACKEND
if [[ "$variant" == flash ]]; then
  export SGLANG_EXTERNAL_MODEL_PACKAGE=hcu_sglang.models.hyv4_flash.external
  export SGLANG_DSA_FUSE_TOPK=1
  mode=normal
  graph_args=(--disable-cuda-graph)
  max_requests=16
  dense_tp=8
  max_total_tokens=65536
  seed=342649703
else
  export SGLANG_EXTERNAL_MODEL_PACKAGE=hcu_sglang.models.hyv4_formal.external
  export SGLANG_DSA_FUSE_TOPK=0
  mode=auto
  graph_args=(--cuda-graph-max-bs 8)
  max_requests=4
  dense_tp=1
  max_total_tokens=16384
  seed=156798804
fi
exec python3 -m hcu_sglang.launch_server \
  --model-path "$model_path" --trust-remote-code \
  --tp 8 --ep 8 --attn-cp-size 8 --enable-prefill-cp --cp-strategy interleave \
  --moe-dense-tp-size "$dense_tp" --moe-a2a-backend deepep --deepep-mode "$mode" \
  --quantization slimquant_w4a8_marlin --kv-cache-dtype fp8_e4m3 \
  --attention-backend dsa --dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_kv \
  --language-model-only --disable-prefill-cuda-graph "${graph_args[@]}" \
  --context-length 8192 --chunked-prefill-size 2048 --max-running-requests "$max_requests" \
  --mem-fraction-static 0.80 --disable-custom-all-reduce --max-total-tokens "$max_total_tokens" \
  --host "${HOST_IP:-0.0.0.0}" --port "${PORT:-30225}" --random-seed "$seed" "$@"
