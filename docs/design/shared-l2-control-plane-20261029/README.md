# Shared L2 control-plane optimization

## Scope

This branch starts at dev `b43f70d69c`. CP development remains paused and is
not included. The changes reduce repeated metadata work in the existing
same-host shared pool; the mmap backing, CUDA registration and DMA layout
remain compatible. Deploy the broker and scheduler overlay together because
the scheduler now uses the `admit` operation.

## Changes

- Eager stores query only confirmed positional candidates beyond each request's
  store cursor, including resumed requests, boundary handoffs and finished tails.
  Unscheduled requests are not scanned. An empty candidate set performs no
  admission RPC.
- `admit` combines existence lookup and bounded slot reservation in one broker
  critical section and one RPC. Already cached blocks advance the store cursor
  even when no slots were reserved.
- Read lookup excludes the GPU-computed prefix. Request-level futures return
  `(None, False)` while lookup is pending, using the scheduler's existing pending
  connector protocol. Allocation retries retain their acquired lease. Request
  cancellation releases results that arrive after cancellation.
- `async_lookup: false` in the connector extra configuration retains synchronous
  suffix lookup for diagnosis. The default is asynchronous.
- Broker stats include `rpc_<operation>`, `admission_keys`, `lookup_keys`, and
  `acquire_keys`. Internal lookup/reservation within `admit` do not count as extra
  wire RPCs. These counters are cumulative; compare deltas during a run.

Async lookup uses one executor thread per scheduler and the existing client
connection/owner. Other synchronous operations on that connection can still wait
behind an active lookup. The broker remains Python/JSON with a shared lock;
this change does not claim to remove its capacity limit.

## Validation and acceptance

The isolated TY18 test container overlays frozen source files onto the existing
vLLM image. The host model service and CP snapshot are untouched.

- Shared-pool tests cover incremental queries, zero-reservation cursor progress,
  nonblocking lookup, allocation retries, cancellation, preemption, confirmed
  token boundaries, finished stores and skipping reads for complete GPU hits.
- A CUDA smoke test verifies WRITING invisibility, cross-client lookup, shared
  D2H/H2D byte equality and zero held leases after completion.
- The existing lazy-reset test contained a stale assertion on `_cursor`, which
  dev no longer defines after the prior lazy eviction fix. Only that assertion
  is removed; completion, reset and cache invalidation assertions remain.
- Full scheduler and shared-pool suites: **97 passed** (22.58 seconds).
  All applicable pre-commit checks, including mypy, passed.
- The isolated harness supplies the original two log-capture fixtures explicitly
  and the official OPT-125m config locally; it runs without network or model
  weights. Deprecation and unregistered marker warnings remain.

No end-to-end throughput improvement is claimed from these tests. Performance
acceptance requires the existing four DEP2 Prefill / DEP8 Decode topology,
approximately 90% token-cache-hit workload, identical request length distribution
and concurrency, and separate GPU/HOST hit counts. Compare completed input TPM,
output TPS, TTFT, errors, queue growth, broker CPU and the new RPC/key counters.
Start at 300M offered input TPM; increase only once the queue is stable. The old
400M offered / approximately 323M completed result is a historical reference,
not a controlled same-machine baseline.

If broker CPU remains saturated after redundant key traffic drops, profile the
remaining operations before adding sharding or changing the storage backend.

## End-to-end benchmark checkpoint

The completed 2026-10-29 development-node measurements are recorded in
[BENCHMARK.md](BENCHMARK.md), with per-run summaries and the final runtime
state alongside it. Completed input throughput was approximately 293–296M TPM
at approximately 90.014% cache hit; higher offered load increased Prefill queues.
This did not demonstrate sustained 350M or 400M capacity.

The 350M and 400M stages ended by killing clients without waiting for drain,
as requested. Their unfinished requests are reported separately. The final
snapshot, taken after evidence collection, shows no active clients, zero P/D
queues, and zero held cache leases. Model services remain running. CP source
development remains paused, with its former Prefill service stopped.

`evidence-locations.json` identifies the external raw results and frozen control
scripts by SHA256. Credentials and bulk request logs are excluded from Git.
