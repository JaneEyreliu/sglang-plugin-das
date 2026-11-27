#!/usr/bin/env bash
set -euo pipefail
model=${1:?Usage: launch_dp8_ep8_dfly_hcu.sh formal|flash [server arguments]}
shift
WORKSPACE=${WORKSPACE:?Set WORKSPACE to the deployment workspace}
case "$model" in
  formal)
    export DEEPEP_MODE=${DEEPEP_MODE:-low_latency}
    width=7
    memory=0.94
    ;;
  flash)
    export DEEPEP_MODE=${DEEPEP_MODE:-normal}
    width=6
    memory=0.90
    ;;
  *) exit 2 ;;
esac
chunk=16384
if [[ "$DEEPEP_MODE" == low_latency ]]; then
  chunk=1024
fi
exec bash "$WORKSPACE/sglang-plugin-das/examples/hyv4/launch_dp8_ep8_hcu.sh" "$model" \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --speculative-algorithm DFLY --speculative-num-draft-tokens "$width" \
  --chunked-prefill-size "$chunk" --mem-fraction-static "$memory" "$@"
