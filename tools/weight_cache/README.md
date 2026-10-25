# Native EP weight-cache IPC validation

`validate_ep_ipc.py` loads a real model through the persistent daemon using
`ipc_cache`, `zero_copy`, and `fallback:false`. It requires eight visible ROCm
GPUs with TP1/PCP8/DCP1/EP8 and the actual HCU Triton MoE runtime. It performs no
forward pass, allocates no KV cache, and establishes no numerical correctness,
throughput, or TTFT result. Separate model/operator correctness gates still apply.

## Setup

Use a compatible installed runtime, the reviewed core/plugin sources, and a real
supported checkpoint. Verify staged source hashes against the declared commits
before running; the receipt records supplied commit labels without verifying Git.
Start the actual EP weight daemon and wait for all ranks to become ready. Keep
it alive throughout both clients and comparison. Never issue its release command.

Run both client lifetimes **inside the private producer container**, sequentially,
using its actual Python, model path, source imports, device visibility, and socket
mount. Separate containers are not required. Root/operator owns the producer
lifecycle and an outer **900-second timeout for each client lifetime**. Each
probe sets a 180-second distributed initialization timeout. Terminate only owned
client processes if the outer deadline expires; never stop an unrelated service.

The plan JSON requires `source_commits` and `engine_python_argv`. Use the same
model, quantization, backend, and relevant engine configuration as the producer.
This minimal schema illustrates the shape; a real checkpoint may require more
flags, native libraries, plugin environment, or model-specific configuration:

```json
{
  "source_commits": {"core": "<verified-commit>", "plugin": "<verified-commit>"},
  "engine_python_argv": [
    "/opt/runtime/bin/python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "/model", "--load-format", "ipc_cache",
    "--model-loader-extra-config", "{\"socket_dir\":\"/run/ep-cache\",\"mode\":\"zero_copy\",\"fallback\":false}",
    "--tensor-parallel-size", "1", "--pipeline-parallel-size", "1",
    "--prefill-context-parallel-size", "8", "--decode-context-parallel-size", "1",
    "--enable-expert-parallel", "--all2all-backend", "allgather_reducescatter",
    "--moe-backend", "triton", "--enforce-eager", "--dtype", "bfloat16"
  ]
}
```

The tool removes API-only `--host`, `--port`, and
`--enable-prompt-tokens-details` options and replaces `--master-port`. All remaining
options pass through the real `EngineArgs` parser. It does not change platform or
version guards, supply a custom model loader, or permit disk fallback.

## Two independent client lifetimes

Choose unused ports distinct from the producer rendezvous. Use fresh output
paths; receipts are created exclusively to prevent stale-result reuse. Within
the private producer container, with `PY` pointing to the real runtime Python:

```sh
PY=/opt/runtime/bin/python
SCRIPT=/src/core/tools/weight_cache/validate_ep_ipc.py
PLAN=/run/validation/plan.json

timeout 900s "$PY" -m torch.distributed.run --nnodes=1 --nproc-per-node=8 \
  --master-addr=127.0.0.1 --master-port=30723 "$SCRIPT" \
  --plan "$PLAN" --master-port 30723 --output /run/validation/ipc-a
# Require exit 0 and all owned client processes gone before launching B.
timeout 900s "$PY" -m torch.distributed.run --nnodes=1 --nproc-per-node=8 \
  --master-addr=127.0.0.1 --master-port=30724 "$SCRIPT" \
  --plan "$PLAN" --master-port 30724 --output /run/validation/ipc-b
"$PY" "$SCRIPT" --compare /run/validation/ipc-a /run/validation/ipc-b \
  --output /run/validation/comparison.json
```

Capture both OS exit codes and verify the producer stays alive. Proceed between
commands only after the preceding checks succeed; a passing JSON cannot override
a nonzero process exit. Comparison requires all eight rank receipts, eight unique
physical GPU UUIDs, and matching identities across both lifetimes.

## Evidence and limits

The production request validates schema, cache key, and physical UUID. Pass-through
observers retain the actual imported tensors without changing responses. The probe
checks every final registered parameter/buffer against its imported storage and
layout, requires tied aliases to retain object identity, checks restored expert
maps/masks/routing tables and EP ownership, and requires the real `FusedMoEKernel`
with exact `TritonExperts`.

Physical allocation fingerprints use the installed Torch rebuild signature by
field name. They include the exported storage handle, offsets, dtype, shape, and
stride, while excluding per-consumer reference counters/events. Raw handles are
never written. Cross-process pointer equality is not used as allocation evidence.
A `None` handle is accepted only for zero storage bytes and zero tensor elements;
these entries carry an explicit empty-allocation marker and separate names/counts.
Each rank must also import a nonempty tensor with positive storage bytes and a
binary handle. Empty views of nonempty storage still require a binary handle.
Client allocator peaks are not total device-memory measurements.

The initial full-checkpoint hardware run passed two eight-rank lifetimes and
cross-run comparison: 3,797 named tensors including aliases, 1,375 nonempty IPC
entries, one empty RoPE buffer, and 42 routed MoE layers per rank. Both process
exits were zero. This result establishes loading/restart identity only; the
separate Indexer numerical gate was failing and is not waived by this probe.

## AMD/FLA Prefill serving follow-up, 2026-10-25

The companion [Plugin PR #57](https://github.com/tai-core/vllm-plugin-das-glm53-bw1000/pull/57)
adds a separate final-token quality study using the same resident producer and
zero-copy clients. Core's `vllm` runtime tree was unchanged for those runs. The measured
configuration is TP1/PCP8/EP8, eager text Prefill, with one generated token per
request; it does not establish Decode or production P-D behavior.

The current acceptance allows ordinary floating-point differences. Twenty
constructed natural-language multiple-choice prompts scored 20/20 under both
C1 and C4. Full-vocabulary total variation reached 0.36% for 2K and 1.38% for
12K. Logits and some distribution screening alerts remain disclosed; this small
smoke set cannot certify overall model quality. The 64-request study includes
actual mixed continuation/fresh-request steps, complete ordered chunk coverage,
and short requests repeated after long requests. Quality observation and forced
four-request intake are excluded from performance timing.

Default fast BF16 projections remain enabled. The maintained FlashKDA-DCU
backend is not connected in these runs; recurrence uses core's AMD/FLA-derived
implementation. The companion quality tools compare the same token's
probability and full unmasked distributions, rather than requiring arbitrary
random-prompt generated tokens to match exactly.

See the companion [quality report](https://github.com/tai-core/vllm-plugin-das-glm53-bw1000/blob/5f04900f7b4930aabd48461f0d3973b60bd9ea97/docs/ep-prefill-quality-20261025.md)
for source hashes, the frozen screening policy, preserved alerts, and limits.
Loading/storage identity, model quality, and throughput are separate pieces of
evidence; this IPC tool continues to validate loading and restart identity only.

Separate uninstrumented ON/OFF and reverse-order follow-ups contain 832 valid
requests, including 640 measurements. The secondary combined 2K/c4 result is
6268.39 -> 7253.42 input tokens/s (+15.71%), with SSE TTFT p95
1.4530 -> 1.1482 seconds (-20.98%). No ON window fell below the historical
6000-token/s slow-band threshold (0/12 versus OFF7/12).

The initial single-window C2 regression (-4.66% throughput) remains in the
report. Reverse six-window confirmation measured +1.94%; the combined C2
summary is +0.95% throughput and -3.22% p95. This does not prove zero overhead:
C1 p95 rose 3.16% in a small sample, and long-control throughput varied
-0.71% to +0.33%. The companion [performance report](https://github.com/tai-core/vllm-plugin-das-glm53-bw1000/blob/041074ce6762d779e9577c1392e859ddf3d574a4/docs/ep-async-prefill-recheck-20261025.md)
preserves each pair, raw-source hashes, the excluded failed harness run, and
the measurement limits. The producer and core runtime were unchanged in that study.

## Optional FlashKDA client recurrence, 2026-10-25

`Glm5NextLinearAttention.prefill_core` now delegates raw beta and convolved
head-local Q/K/V to `prefill_recurrence`. Its default implementation preserves
the existing AMD/FLA call and applies sigmoid exactly once. Convolution, request
state gathering/scattering, cache shapes and ownership stay in `prefill_core`.
The companion plugin can select the maintained FlashKDA implementation with
`VLLM_HCU_GLM_KDA_BACKEND=flashkda`; the default remains `fla`. This selector is
for the IPC client experiment, with the existing producer kept resident.

FlashKDA receives raw BF16 beta and normalizes Q/K internally. The plugin passes
the rank's head interval, contiguous tensors and int64 sequence boundaries,
validates the native ABI before layer initialization, and records selection
and first actual execution on every worker. Explicit unsupported selections or
missing extensions fail instead of falling back. FlashKDA accepts FP32 cache
buffers but rounds recurrent state through BF16 internally.

The initial source-pinned native study passed 92 numerical checks against an independent
FP64 recurrence, including C1/C4, head slicing and next-chunk continuity. It
showed faster local H8 recurrence for 2K layouts, slower single-sequence 8K, and
slightly higher FP64 error than AMD/FLA. The 5% relative-L2 component screen is
not a model-quality tolerance or a serving-performance claim. See the companion
`docs/ep-flashkda-native-20261025.{md,json}` report for full measurements and
preserved failed build/harness attempts.

The matched full-model study completed 128 quality requests: both backends
scored 20/20 under C1 and C4, with no answer flips and stable repeat/post-long
checks. Cross-backend full-vocabulary TV reached 0.48% on 2K and 1.35% on 12K;
all numerical screening alerts remain disclosed. This is a small constructed
task smoke test, not general model-quality certification.

The initial uninstrumented 300-request comparison measured 2K/C2 throughput
4252.67 -> 5144.16 input tokens/s (+20.96%), with SSE TTFT p95
1.03995 -> 0.81952 seconds (-21.20%). 2K/C4 throughput rose 5.40% and C1 rose
16.76%, while 8K and 32767-token controls lost 5.87%-6.39% throughput. FLA
therefore remains the default; FlashKDA is an explicit opt-in for evaluating
short-request workloads. One FLA-then-FlashKDA pair and small long-control
samples do not establish statistical non-inferiority or production tail bounds.
See the companion [full report](https://github.com/tai-core/vllm-plugin-das-glm53-bw1000/blob/ea662c41a0e832ccb2c68cffb5ade1b897602c91/docs/ep-flashkda-prefill-20261025.md)
and JSON for matching hashes, per-window results, quality limits and cleanup.
The same resident producer survived every run; no daemon restart, checkpoint
fallback or changes to node08 were needed.

### FlashKDA long-request optimization follow-up

The final native candidate `25c67a311ed2645feb3aeb59912dc350b598dcf9`
adds value-column partitioning, DPP layout exchanges, operand prefetch and
overlapping loader reads. AUTO uses one loader wave for narrow value partitions;
the core and plugin runtime remain the same as the earlier client study.

A new matched 486-request run (360 measured, 102 warmup, 24 ingress audit)
measured 8K/32767-token throughput 5.57%-5.85% above the original FlashKDA,
with TTFT p95 5.16%-5.42% lower. The optimized long controls remain
0.51%-1.28% below FLA throughput. 2K/C1/C2/C4 throughput is respectively
15.92%/16.84%/9.18% above FLA, preserving the short-request gains.

The candidate passed 153 CPU contracts, 83 exact native output/state and
continuation checks, and 92 FP64/AMD/Flash checks. In 128 model-quality requests,
C1/C4 remained 20/20; 64 matched full-vocabulary captures were identical to
original FlashKDA. These are bounded smoke results, with existing distribution
alerts retained. FLA remains the default backend. The original producer stayed
resident and all test clients were cleaned up; node08 was not changed.

See the updated companion
[optimization report](https://github.com/tai-core/vllm-plugin-das-glm53-bw1000/blob/codex/weight-cache-static-ep/docs/ep-flashkda-long-optimization-20261025.md).
The completed run is the final acceptance scope; no additional reverse-order
run or final-candidate model profile was performed.
