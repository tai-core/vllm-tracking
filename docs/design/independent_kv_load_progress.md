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
model/daemon at gpu-memory-utilization=.152. Lowering the activation token
budget lets native profiling increase KV from 1.71 to 2.82GiB/rank. This is a
configuration-level gain, not a new kernel implementation. It should not be
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
