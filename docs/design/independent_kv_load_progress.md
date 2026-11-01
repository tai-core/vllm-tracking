# Independent lookup and KV receive progress

A nonblocking query only helps when the engine reaches it promptly. Previously,
lookup completion was checked while admitting requests, so a full model queue or
an allocation failure ahead of a request delayed its lookup. Worker receive
submission and completion checks also ran through the serial model RPC loop.

With async-load prefetch, async scheduling, MRV2, PP1/DP1, no speculative decoding,
and a polling executor, EngineCore now progresses pending lookups before checking
model queue capacity. This polls existing lookups; it neither allocates pages nor
admits requests. Completed lookup results remain owned by the connector until
normal admission or cancellation consumes them.

For a single-node MultiprocExecutor and an explicitly supporting connector:

```text
Scheduler allocates private async destination pages
  -> split_async_loads(metadata)
  -> per-rank control Pipe -> Worker receive thread -> stream restore
  -> event query succeeds -> independent control reply
  -> Core gathers every rank -> Scheduler marks ready -> normal model scheduling
```

The normal model RPC queue and its FIFO replies are unchanged. The control
channel carries receive completions and invalid block IDs, not model results or
store completions. Each worker has one receive-state owner. A rank is counted
once per pending request, and the request is not ready until every worker reports
completion. Errors propagate independently of model RPC replies.

## Connector contract

- `split_async_loads` returns ordinary metadata and an optional independent load
  payload. It must preserve ordinary bookkeeping and remove operations it moves
  to the independent path, preventing duplicate submission.
- `initialize_async_loads` explicitly opts into independent receive ownership.
- `start_async_loads` accepts explicit metadata without binding or clearing the
  forward connector metadata or changing the global forward context.
- `poll_async_loads` performs a nonblocking completion query and returns `None`
  while there is nothing to report. Invalid block IDs can precede completion;
  they must not be reported later than the corresponding completion.

MultiConnector keeps child indices in the independent payload. Unsupported
children continue through the ordinary path. The current LMCache integration
opts in only for pure consumers without lazy offload or controlled-receipt
namespaces. Its control thread owns retrieval submission, result observation,
error draining, and event release; the forward thread no longer drains them.

The worker records a cache-initialization event on the normal stream before
starting loads on a separate stream. Destinations are allocator-owned async
pages. Cancelled or failed requests retain these pages until all ranks finish
receiving; control threads stop before connector/device shutdown. No additional
GPU collectives or per-layer host waits are introduced.

## Validation

CPU tests cover lookup progress behind a blocked allocation/full model queue,
independent restore completion while model work is pending, repeated incomplete
queries, all-rank completion aggregation, errors, and control-thread shutdown.

On node07 with the existing weights daemon, six C1/C4 requests passed a cold ->
8192-token cached prefix -> 24575-token continuation comparison. All 1632 KDA
state pairs (6528 components) were exact, all 48 request/rank continuation traces
were contiguous and agreed across ranks, and all 48 physical page aliases were
verified. Full-vocabulary total variation was at most 0.007588; all six top1
outputs agreed. These synchronized numerical runs are excluded from performance
measurements. Node08 was not used.

## Matched timing and profile

The primary comparison reuses `continuousrestore20261030b` at 9 RPS. Ordered
input payloads match after excluding request IDs: 945 issued requests and 812
measurement requests. Both use 9102 input tokens (8192 cached, 910 new), GPU L1
plus 96GiB Host L2, 236 automatically sized KV blocks/rank, max batch tokens
12288, CP8+EP8, and the same persistent weights daemon.

| Metric | Prior candidate | Independent progress |
| --- | ---: | ---: |
| Actual offered RPS | 9.480953 | 9.480941 |
| Logical completion tokens/s | 85,546 | 85,971 |
| TTFT P50 | 2306.3ms | 1715.5ms |
| TTFT P90 | 3154.7ms | 2198.4ms |
| TTFT P99 | 3725.7ms | 2537.7ms |
| Mean server waiting | 18.22 | 9.53 |

These are single finite windows. P90 improves 30.31%; throughput is essentially
unchanged at the offered-load ceiling, not a peak-capacity result. The candidate
has 18 more complete-prefix equivalents served from GPU L1, so this is not a
fixed-L2-hit-ratio restore microbenchmark. Both runs pass source and queue-trend
checks with zero preemptions.

A separate 30-second profile observes the following P50 delays:

| Interval | Before | After |
| --- | ---: | ---: |
| Admission to lookup actually ready | 959.61ms | 8.96ms |
| KV allocation to first Worker restore submit | 361.53ms | 10.56ms |
| Last sampled restore GPU phase to all-rank observation | 367.30ms | 5.82ms |
| All-rank observation to Core receive | 13.78ms | 3.71ms |

The candidate has 234 complete measurement restore chains and 56 native GPU
samples. Lookup completion uses its own marker; its later scheduling lookup is
not treated as the first completion. Sampled copy/scatter union is 3.88ms P50,
with 98.5% temporal overlap with other-request forward stream spans. Intervals
are not additive, GPU anchors are approximate, and temporal overlap is not a
causal savings estimate.

Remaining waits are visible: lookup-ready to allocation is 348ms P50, and ready
requests can still wait behind model batches. The new control send costs 12.35ms
wall / 7.81ms thread CPU P50 per control batch. This reduces waiting; it does not
claim zero cache overhead or 180k logical tokens/s capacity.

## Open-loop SLA capacity sweep

A subsequent AIPerf request-rate sweep uses strict TTFT P90 < 12s plus a
service-queue stability check. Profiles are excluded. Coarse/refined rates use
the same qualification as 300-second confirmations; an SLA-pass window with
growing waiting queues is not accepted as sustained capacity.

| Batch token budget | Auto KV blocks/rank | Highest confirmed nominal RPS | Logical tokens/s | TTFT P90 |
| --- | ---: | ---: | ---: | ---: |
| 12288 | 236 | 10 | 90,268 | 4.218s |
| 8192 | 389 | 13 | 116,498 | 3.550s |

Both retain GPU L1, 96GiB Host L2, stream restore, max_num_seqs=64, and the same
model/daemon at gpu-memory-utilization=.152. Lowering the batch token cap
lets native profiling increase KV from 1.71 to 2.82GiB/rank. The 1134MiB
increase comprises 1058MiB less retained non-KV consumption and 76MiB less
transient headroom; it is not solely a reduction in peak activations. This is
a configuration-level gain, not a new kernel implementation. It should not be
assumed optimal for uncached requests or other lengths.

The final 8192 confirmation has 4095 successful requests, actual arrival
12.7926 RPS, completion 12.7992 RPS, and a 305.097-second offered window.
The adjacent 14 RPS trial has P90=9.295s but waiting rises (slope .1034 req/s),
so its 123,017 completion tokens/s are not accepted. The highest validated
point reaches 64.72% of the 180k target, leaving 63,502 tokens/s (35.28%).
Resolution is one nominal RPS; this is finite-window evidence, not a global
hardware peak or indefinite stability guarantee.

The initial pooled-connection 8192 confirmation had two immediate HTTP transport
errors and was excluded. The complete confirmation was repeated with AIPerf's
`--connection-reuse-strategy never`, with zero errors and connection setup left
in TTFT. The mechanism behind the earlier disconnects was not proven. The
12 RPS coarse configuration comparison used pooled connections in both runs
and identical ordered payloads except request IDs: P90 changed 20.706s -> 2.764s.

At the final qualified point, mean batch size is 6054 tokens (73.90% of budget),
P90/max 8190, and 32.71% of batches exceed 95% occupancy. The 12288 setting had
mean 4065 and no such full batches even under overload. Thus input pressure was
not the only constraint, and average batches are still not full.

With nonshared prefixes, each request needs 13 restore blocks plus 5 continuation
blocks. Those budgets are shared across all in-flight cohorts. Reservations are
absent from the reported block-usage gauge. The original profile had 81 external
allocation rejections; all had fewer than 13 free-after-reservation blocks.
High-free local-hit rejections also point, by source exclusion rather than a
direct branch trace, to Mamba's same-step cached-state guard. This does not prove
that the guard can safely be removed.

The final profile has roughly 4ms sampled copy/scatter intervals, all temporally
covered by other forwards. Representative sampled model-stream shares are EP
dispatch/combine 18.82%, Indexer 15.79%, MoE experts 14.82%, MLA 13.13%, KDA
head exchange 8.00%, FlashKDA recurrence 3.46%, and 25.98% uninstrumented remainder.
EP includes peer waits, expert work includes activation/quant/permutation, and
these are stream intervals, not pure kernel utilization. Operator sampling
raises CPU time in geometry-matched observations, so its timings only localize
work; unprofiled windows determine capacity.

## Fixed-KV capacity at an 8192-token batch cap

The November 1 trials retain the same model/cache bundle, CP8+EP8, GPU L1,
96GiB Host L2, stream restore, HTTP connection policy and existing node07
weights daemon. Node08 is not accessed. Explicit KV sizing keeps the startup
profile forward but bypasses automatic KV sizing. AIPerf uses Poisson arrivals,
without a client concurrency cap, and qualification requires TTFT P90 < 12s
and no sustained waiting/backlog growth. Input is 9102 tokens: 8192 cached and
910 newly computed; the reported logical throughput includes the cached input.

| KV setting/rank | Blocks | Highest qualified nominal RPS observed | Logical tokens/s | Input tokens/min | TTFT P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Auto baseline, 3,025,017,816 bytes | 389 | 13 | 116,498 | 6,989,901 | 3.550s |
| Explicit +512MiB, 3,561,888,728 bytes | 458 | 14 | 124,503 | 7,470,204 | 4.133s |
| Explicit +1GiB, 4,098,759,640 bytes | 528 | 15 | 133,083 | 7,985,010 | 6.589s |

The +512MiB 14RPS window has 4410/4410 successful requests. Its adjacent
15RPS trial fails with P90=13.292s and growing waiting queues. At matched
14RPS, full-batch share improves from the auto baseline's 43.52% to 52.33%.
The minimum sampled free HBM in the +512MiB run is 1.761627GiB.

The initial +1GiB pressure run completes 700/700 requests and 72 full
8190-token batches. It was stopped by an experimental 1GiB free-HBM floor:
two sampling streams observe minima of 0.964802 and 0.929420GiB. That floor
was an experiment policy, not a hardware limit or an observed OOM. The
operator then removes the floor and reuses this full-batch evidence to run
the same +1GiB configuration through the SLA sweep; HBM monitoring remains.

In that continuation, 14RPS qualifies at 124,593 logical tokens/s and P90
3.338s. The completed 15RPS window has 4725/4725 successful requests, actual
arrival/completion 14.631025/14.621346 RPS over 307.427s, and P90=6.588838s.
Waiting decreases between window halves, with slope -0.10055 requests/s.
This is 6.89% above the +512MiB qualified throughput and 14.24% above the
auto baseline. Computed throughput is 13,305 tokens/s; logical throughput
reaches 73.94% of the 180k target.

The operator stops the higher-rate scan to resize Host L2. Therefore 133,083
logical tokens/s is a qualified finite-window lower bound, not a located
peak. The interrupted point contributes no capacity result. The raw runner
summary retains its interrupted RUNNING label; the explicit outcome is
recorded in `experiments/kv-plus1g-openloop-20261101/interrupted-analysis.json`.
The stopped owner cleans up its test API, load generator and old L2 servers,
while retaining the existing weights daemon.

The source code is unchanged. Comparisons are finite windows with the same
request geometry and unique suffixes, not exact-payload causal experiments.
The functional numerical and next-chunk evidence remains the previously
validated bundle. Profiles are excluded from capacity. A separate +512MiB
profile still points to allocation/admission and ready-to-schedule waits;
restore intervals mostly overlap other forwards and are not additive idle.

## Host L2 at 80% of available memory

After stopping the sweep, eight persistent Host-L2 servers replace the old
96GiB test pool. Their capacity is computed once at startup:

```text
effective_available = min(host MemAvailable, cgroup memory.max - memory.current)
per_server = floor(0.8 * effective_available / 8 / 64MiB) * 64MiB
```

The container limit is 768GiB. Its startup current usage is
43,468,255,232 bytes, leaving 781,165,465,600 bytes (about 727.52GiB), which
is tighter than host MemAvailable. Rounding each server down to the allocator's
64MiB unit gives **72.75GiB per server, 582GiB total**, or 6.0625 times the
old Host-L2 capacity. GPU L1 continues to share the model's GPU KV pool;
Host L2 is separate system memory.

The server's internal flag is `--l1-size-gb 72.75`; despite that name it
configures this Host-L2 tier. The same functional bundle, 8192-token cache
chunks, separate object groups and rank/port mapping are retained. The
allocator's lazy mode registers memory in the background up to its configured
capacity; it is not growth driven only by cache requests.

The L2-only owner is `/runstate/l2available80-20261101/controller.py`; its
receipt and per-server logs are in the same directory. It keeps only the
cache services running, without starting a model API or benchmark. The
table above predates this enlargement and used 96GiB Host L2. The following
section records the subsequent 582GiB validation.


All eight new L2 servers complete registration of 74496MiB each with zero
pin-memory failures, and all eight PING requests succeed. After registration,
container usage is about 624.23GiB of 768GiB, while host MemAvailable is about
327.58GiB. The original weights daemon PID and start time are unchanged.
The L2-only services are intentionally left running; no benchmark is resumed.


## Open-loop validation with the persistent 582GiB L2

A subsequent sweep borrows the running eight L2 servers instead of restarting
or owning them. Pre/postflight validates the original daemon processes and the
specific L2 owner/server identities; cleanup covers only this run's API and
FakeD processes. The L2 pool remains 582GiB and all existing data is retained.

HBM inspection shows 10.33–10.52GiB free per card before the API starts, about
0.9GiB less than with the old 96GiB L2. This run therefore uses 3,561,888,728
KV bytes per rank (458 blocks), with batch cap 8192. The explicit-KV startup
utilization setting is .14. No 1GiB headroom gate is imposed. The minimum
sampled free HBM over pressure and measured loads is 0.827728GiB; no OOM or
illegal-access error is observed.

The first attempt is invalid: rank 6 briefly enters degraded mode during seed
requests and skips stores, although HTTP seeding succeeds. This leaves some
prefixes absent only on server 30734, causing cold fallback during the pressure
window. That run contributes no capacity result. The retry reads all 64 prefixes
on all 8 servers while the seed API's layouts remain registered, repairs only
missing slots 1–11, then verifies all 512 entries before measurements. One repair
round suffices. This is a benchmark preparation fix, not a change to LMCache's
fail-open availability policy. Probe locks and sessions are released.

| Host L2 | GPU KV blocks/rank | Qualified nominal RPS | Input TPM | TTFT P90 |
| --- | ---: | ---: | ---: | ---: |
| 96GiB, matched KV baseline | 458 | 14 | 7,470,204 | 4.133s |
| 582GiB, persistent pool | 458 | 14 | 7,473,773 | 6.297s |

The 582GiB point completes 4410/4410 requests over 306.170 seconds, with actual
arrival/completion 13.717817/13.685221RPS. Every request accounts for 8192 cached
and 910 computed tokens. Waiting decreases between window halves (slope
-0.20715 requests/s). The adjacent 15RPS point completes 4725/4725 requests but
fails P90=14.167825s and has growing waiting (slope+0.35703 requests/s).
Its throughput is excluded. Thus the measured qualified result is 124,563
logical tokens/s, or 7,473,773 input tokens/min; computed throughput is 747,213
new tokens/min.

At the qualified point, mean batch size is 6917 tokens and 56.70% of batches
reach 8190 tokens. The matched-KV throughput differs only +0.0478% from the old
96GiB result: this 64-prefix working set shows no throughput gain from the
larger Host-L2 capacity. P90 is higher in this single window; that alone does
not establish a causal regression. The earlier 7,985,010TPM used 528 GPU blocks
and is not a matched-KV comparison.

This remains a finite-window, one-nominal-RPS-grid result, not a global hardware
maximum. Thirty qualification, memory-gate and prefix-readiness tests pass;
independent review checks source counts, the 512-entry matrix, batch data,
HBM and cleanup. The immutable functional bundle remains unchanged. Persistent
L2 servers have no per-run profile overlay, so absent server-side events must
not be interpreted as zero overhead. Both failed and successful owners retain
the original weights daemon and L2 servers; node08 is untouched.

Artifacts: `experiments/l2-582-openloop-retry-20261101/result.json`,
`sweep-analysis.json`, `memory-readiness.json` and `output/`; the invalid first
attempt is retained separately in `experiments/l2-582-openloop-20261101/`.


The validated model-start settings for this persistent L2 are:

```text
--max-num-batched-tokens 8192
--kv-cache-memory-bytes 3561888728
--gpu-memory-utilization 0.14
```

Reuse the existing eight L2 endpoints and retain GPU prefix caching and stream
restore. The earlier explicit +1GiB KV setting was measured with 96GiB L2;
it is not the memory configuration validated in this 582GiB sweep.
