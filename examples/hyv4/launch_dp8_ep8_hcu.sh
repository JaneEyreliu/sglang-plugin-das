#!/usr/bin/env bash
set -euo pipefail
model=${1:?Usage: launch_dp8_ep8_hcu.sh formal|flash [server arguments]}
shift
WORKSPACE=${WORKSPACE:?Set WORKSPACE to the parent directory of adapter-das, sglang and sglang-plugin-das}
MODEL_ROOT=${MODEL_ROOT:?Set MODEL_ROOT to the model mount directory}
export PYTHONPATH="$WORKSPACE/adapter-das:$WORKSPACE/sglang/python:$WORKSPACE/sglang-plugin-das"
export SGLANG_PLUGINS=hcu
export HCU_SLIMQUANT=0
export HCU_DEEPEP_BRIDGE=0
unset SGLANG_USE_W4A8_MASKED_HIPC SGLANG_USE_W4A8_CONTIGUOUS_HIPC SGLANG_W4A8_TPMOE_BACKEND SGLANG_ENABLE_CP_V2
case "$model" in
  formal)
    MODEL_PATH="$MODEL_ROOT/Hy4-Channel-INT4-w4a8"
    export SGLANG_EXTERNAL_MODEL_PACKAGE=hcu_sglang.models.hyv4_formal.external
    export SGLANG_DSA_FUSE_TOPK=0
    ;;
  flash)
    MODEL_PATH="$MODEL_ROOT/Hy4-flash-Channel-INT4-w4a8"
    export SGLANG_EXTERNAL_MODEL_PACKAGE=hcu_sglang.models.hyv4_flash.external
    export SGLANG_DSA_FUSE_TOPK=1
    ;;
  *) exit 2 ;;
esac
DEEPEP_MODE=${DEEPEP_MODE:-low_latency}
case "$DEEPEP_MODE" in
  low_latency)
    if [[ "$model" == flash ]]; then
      echo "Flash hidden size 1536 is unsupported by current native DeepEP low_latency; use DEEPEP_MODE=normal" >&2
      exit 2
    fi
    export SGLANG_USE_W4A8_MASKED_HIPC=1
    ;;
  normal) export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1 ;;
  *) echo "DEEPEP_MODE must be normal or low_latency" >&2; exit 2 ;;
esac
exec python3 -m hcu_sglang.launch_server \
  --model-path "$MODEL_PATH" --trust-remote-code --language-model-only \
  --tp 8 --attn-dp-size 8 --enable-dp-lm-head --ep 8 \
  --moe-dense-tp-size 1 --moe-a2a-backend deepep --deepep-mode "$DEEPEP_MODE" \
  --quantization slimquant_w4a8_marlin --kv-cache-dtype fp8_e4m3 \
  --attention-backend dsa \
  --dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_kv \
  --disable-prefill-cuda-graph --disable-cuda-graph \
  --context-length 8192 --chunked-prefill-size 16384 \
  --max-running-requests 64 --max-total-tokens 16384 \
  --mem-fraction-static 0.90 --host 0.0.0.0 --port 18026 "$@"
