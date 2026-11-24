# HY4-Flash HCU adaptation

This port adds the community HY4-Flash model to official SGLang v0.5.22 through the HCU adapter and plugin. The reference is community main commit `2544c9ac34be4447bdffa612c37015a3f8da9e84`; exact inputs and destination hashes are recorded in `hy4-flash-source-manifest.json`.

## Scope and ownership

The plugin owns `hcu_sglang.models.hyv4_flash`, its external model registration, config classes, mixed CT weight mapping, and scoped DSA/model-shape compatibility hooks. The adapter owns the three community Triton operator modules in `hcu_adapter.hy4_flash_ops`. Official SGLang source is preserved. The model is imported from the installed plugin, with no runtime import from a DAS fork.

The initial deployment uses TP8, EP1, full KV storage, language-model-only serving, W4 experts with dynamic A8 and channel W8 dense/shared projections, and the existing HCU Triton MoE runner. Flash's window selection is implemented by its indexer over the full sparse KV pool. It does not opt into a separate hybrid SWA pool. Logical pages are 64 tokens. Flash requires scaled FP8 Index-K independently of the BF16 MLA KV dtype. The existing index-cache construction ContextVar is enabled only for Flash or the previously supported GLM compressed-cache predicate, and the memory estimator uses the same policy; the context is reset on success and failure.

Main's NVIDIA FlashInfer top-k is replaced with Torch top-k. CT expert packed INT32 tensors use the existing plugin packed-byte conversion; CT INT8 scales keep their original scale names. Dense and shared FFNs receive the checkpoint's mixed quantization configuration. The 0.5.22 TopK interface is supplied an explicit STANDARD output format, equivalent to main's bound standard runner for this deployment.

Flash indexers emit physical KV slots for paged decode and ragged offsets for prefill. The environment explicitly sets `SGLANG_DSA_FUSE_TOPK=1` and the model hook rejects a logical-index consumer or a Top-K backend other than `sgl-kernel`. Without this requirement the existing HCU backend translates physical slots again, producing incorrect output and, with a smaller pool, FlashMLA VMFault. The sliding-window output is padded with invalid entries to index_topk=2048 for the HCU decode ABI; only the causal 512-token window remains valid.

Configuration hooks are scoped to `HYV4FlashForConditionalGeneration`; other models call the current wrapped baseline. AutoConfig and SGLang config registration precede external model discovery. The standard language-model-only architecture allowlist is extended to include Flash. No global architecture substitution is retained: the shape hook uses a temporary instance config copy and restores the original config in `finally`.

Speculation, CP, DCP > 1, attention DP, PP > 1, MoE A2A, waterfill, HiCache, HiSparse, PD and batch overlap are rejected. VL is not part of this deployment's acceptance criteria; imported vision source is retained but its execution is unvalidated.

## Run and verify

Set `HY4_FLASH_ROOT` and `HY4_FLASH_MODEL` for the deployment. The accompanying `py_scr/hy4_flash_env.sh` selects the three source roots and `hcu_sglang.models.hyv4_flash.external`. Start with `bash py_scr/hy4_flash_tp8.sh`. Its initial limits are context 8192, chunked prefill 2048, four running requests, and decode Graph batches up to four; prefill Graph is disabled. The default uses NCCL collectives (`--disable-custom-all-reduce`) and a 16384-token total KV budget to avoid quantizing an unnecessarily large BF16 pool on each decode step. This budget is shared across requests.

The adapter requires its native extension. On gfx938 with the matching DTK/Torch environment, build in the adapter directory:

```bash
AMDGPU_TARGETS=gfx938 MAX_JOBS=8 DAS_PLATFORM_TAG=dtk2610.torch2110 python3 setup_hip.py build_ext --inplace
```

Keep the native gate enabled. An extension compiled for another Torch/DTK ABI must be rebuilt. The dedicated container also required IPython for the official public API import.

Run the opt-in integration check from the deployment root after sourcing the environment:

```bash
HCU_PLUGIN_RUN_GPU_TESTS=1 python3 -m unittest discover -s sglang-plugin-das/tests -p test_hyv4_flash_integration.py -v
python3 py_scr/test_hy4_flash_kernels.py
python3 py_scr/test_hy4_flash_indexpool.py
python3 py_scr/test_hy4_flash_pool_update.py
python3 py_scr/test_hy4_flash_fused_norm.py
python3 py_scr/test_hy4_flash_expert.py
python3 py_scr/test_hy4_flash_service.py
```

The kernel check includes iHC empty/small/prefill-sized reference comparisons, paged/ragged window indices and changed-position Graph replay, and BF16/FP8 Engram lookup comparisons. Independent IndexPool checks cover cross-page causal logits for 1/4/65 rows, changed-query/position Graph replay, query RoPE/Hadamard against an FP32 oracle within FP8 rounding bounds, gated pool writes at page boundaries, and decode update versus prefill cache with changed-input Graph replay. The service check saves raw sequential/concurrent/repeated responses and requires complete expected answers with normal stops.

## Validation record

Baseline: official `baa9befa381511cd4d56f83980951e5a45fffc6d`, adapter `94911d1`, plugin `6df16b7`. Input content was clean when file-mode noise was excluded; baseline snapshots were retained before editing.

On 2026-11-23, native gfx938 compilation and six public integration tests passed. These cover config/geometry/registry, real BF16 MLA plus scaled FP8 Index-K allocation and capacity accounting, unsupported modes, unrelated DSA fallback, actual decode consumption of nonidentity physical indices, and rejection of logical-index mode. Component checks passed for iHC, fused post/pre norm, window-selection, Engram, IndexPool logits, query, pool writes and decode updates, including changed-input Graph replay. Actual checkpoint TP8 W4 expert and W8 attention/dense projection tests passed against independent quantized Torch references.

An initial full-model Graph run loaded all 148 shards and served HTTP successfully but failed all nine semantic checks. Small-KV eager attempts also exposed FlashMLA VMFault, including with custom all-reduce disabled. The physical-index consumer mismatch was then fixed; eager NCCL mode passed all nine semantic checks with correct complete answers and normal stops. This includes sequential/repeated and concurrent requests, 2352/2356-token chunked prefill and shared-prefix cache hits. These are smoke checks, not a benchmark accuracy claim. Final decode Graph capture (batch sizes 1/2/4) and all nine semantic checks also passed. A separate 60-token response enumerated integers 1 through 30 exactly, ended normally, and the scheduler log confirmed `cuda graph: True` during decode. The acceptance service used port 30223 with PID 108002; that Flash service was subsequently stopped before formal-model validation.

VL, long-context accuracy beyond these smoke inputs, MTP, CP/DCP, PD and performance tuning have not been validated. Logs and raw responses are retained under `flash_validation_20261123/` including failed attempts and `eager-pass-generation-smoke.json`.

## GSM8K-200 follow-up

On 2026-11-23 the first 200 questions from the existing official GSM8K test.jsonl (SHA256 3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14) were evaluated with the same running service. Historical chat/no_think 0-shot, 4096-token protocol scored 193/200 (96.5%), all 200 normal stops, zero invalid and zero truncated. Historical raw completion 5-shot, 512-token bench scored 93/200 (46.5%), with 44 invalid and 46 truncated; 43 invalid responses hit the Question stop string. Normal-stop-correct scoring for raw completion was 79/200. These protocols differ and scores must remain separate. Both were rescored offline, raw responses retained, and model identity and final health 200 checked. This is a 200-question subset measurement without a same-checkpoint upstream-service baseline. Detailed protocol, data location, scripts and artifacts are recorded in the deployment root under flash_validation_20261123/GSM8K_REPORT.md and gsm8k_default_200 / gsm8k_chat_200.

## 2026-11-24 proj/hy4 follow-up

The same runtime used for the formal model was revalidated with Flash. TP8 decode Graph and nine generation checks passed. Same-protocol GSM8K first 200 scored 190/200 versus 193/200 on the old baseline; normal stops 200, invalid 0, truncated 0. See the dated [development record](hcu-sglang-development.md) for fixed versions, independent rescoring, changed questions and limitations. Raw evidence is retained outside Git under `hy4_validation_20261124/proj_hy4_flash/`.

## 2026-11-24 repeat accuracy result

The unchanged runtime repeated the same first-200 GSM8K chat/no_think protocol and scored **193/200 (96.5%)**, compared with 190/200 in the preceding run and 193/200 historically. All 200 requests stopped normally; independent scoring agreed, with no invalid or truncated outputs. Relative to the preceding run, indices 30, 125 and 187 improved and none regressed. This is consistent with the reproduced batch-sensitive behavior; the small number of runs does not establish a normal variation range or identify the responsible operator. See the dated [development record](hcu-sglang-development.md) for versions, evidence and limitations.
