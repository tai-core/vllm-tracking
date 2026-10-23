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
