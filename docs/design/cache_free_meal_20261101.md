# Cache critical-path optimization

The changes remove idle Core model RPCs, reuse Indexer metadata within a batch,
and bypass MLA replicated-row filtering for pure Prefill. Stream restore stays
enabled. The model-service runs use the original node07 weights daemon, CP8+EP8, GPU L1,
582 GiB Host L2, 3561888728 KV bytes/rank and an 8192 batch-token cap. Node08 was
not accessed.

## Core admission

The same offered 14 RPS workload improved TTFT P90 from 6.297 to 2.841 seconds.
At 15 RPS, P90 improved from 14.168 to 6.336 seconds and the queue qualification
changed from fail to pass. The qualified point was 133590 logical input tokens/s
(8.015 million TPM), 7.25% above the prior highest qualified point. This is an
earlier finite-window result for the Core-only candidate, not a measurement of
the final combined code.

The first Core numerical run had two cold/restore TV values above the original
1% gate (maximum 1.15148%). Its direct comparison with a matched deployed-code
restore stayed within that same tolerance (maximum 0.78518%, matching top-1).
Performance testing proceeded under the user's allowance for small output
differences; the original cold/restore failure was retained. The later combined
candidate independently passed the original cold/restore gate, as recorded below.

The separate 14 RPS event profile showed zero-token model executions fall from
25 to zero. Lookup completion to first KV allocation P50 fell from 796.44 to
1.716 ms. Ready-to-schedule P50 remained about 504 ms. The changes therefore
removed an admission delay without proving that all cache waiting disappeared.

## Indexer and MLA

Indexer row uploads and local metadata are reused within `Glm53PCPStep`; CPU
searchsorted replaces GPU row-mask selection. In six sampled batches, Indexer
HtoD calls fell from 264 to 36, and the new path had four nonzero operations per
batch. Its single cumsum per batch was not the large wait observed in profiling.

The initial cross-run Indexer long test failed at 14 RPS, including 639 timeouts.
The complete ordered payloads matched the Core run after excluding request IDs.
A later same-process old/new/old Torch comparison showed the candidate's CPU
wall time improve from 99.996/100.161 to 91.262 microseconds per new token. GPU
activity stayed in the same range. This did not justify attributing the entire
cross-run slowdown to Indexer, or treating profiler throughput as capacity.

MLA's six boolean-index operations exposed long synchronization waits. The pure
Prefill fast path uses the authoritative CPU ownership flags, preserves padding
and contiguous inputs/slots, and retains the old path for mixed or unsupported
cases. Its real trace contained 66 sampled MLA calls with zero nonzero operations
and zero 4-byte DtoH copies. The measured CPU issue time became shorter than GPU
activity; that alone is not a throughput result.

## Ordinary, unprofiled comparison

One API process switched implementations while idle. Every variant passed the
same six ordinary API output checks before receiving load; GPU L1 was reset and
the payload sequence was identical apart from request IDs. All eight ranks
confirmed the selected source and that Torch/event profiling was off.

| Variant | Completion rate inside offered window | TTFT P90 |
|---|---:|---:|
| Core-only reference, first | 10.479 req/s | 13.453 s |
| Row/local-metadata reuse, old pool path | 10.648 req/s | 12.168 s |
| Full Indexer reuse | 11.155 req/s | 11.054 s |
| Indexer + MLA fast path | 11.459 req/s | 9.836 s |
| Core-only reference, last | 10.580 req/s | 13.352 s |

The combined variant was about 8.8% faster than the mean of the two reference
points. These were short 14 RPS diagnostic windows with queue growth; none is a
capacity claim. Absolute results differ from the earlier Core long run. The
current evidence establishes the same-process ordering, not the cause of that
cross-run baseline shift.

## Correctness and source scope

The combined candidate passed C1/C4 full-vocabulary checks: maximum cold/restore
TV was 0.62117%, and all six top-1 outputs agreed. All 1632 restored KDA state
pairs / 6528 components matched their stored values exactly. All 48 request/rank
continuation traces were contiguous and agreed, with four or five suffix chunks.
It also passed the six-case ordinary API token/logprob check without per-layer
numeric synchronization.

Targeted tests passed: 29 Core, seven LMCache ownership, nine sink, 108 Indexer,
and 51 MLA/step tests. One strict xfail preserves a pre-existing mixed ownership
counterexample; this work does not claim general mixed decode correctness.

LMCache staging views are also cached at registration. Four CPU correctness
tests and a BW1000 buffer test passed. A real-device accessor microbench reduced
one kernel-view plus one object-view lookup from 14.702 to 0.725 microseconds of
host CPU time. This is a preparation-cost result. The ordinary L2 servers still
use their previous bundle, so no service-level gain is attributed to that change.

## Final qualification

The final same-process comparison used 300-second offered windows, identical
12 RPS payloads, the same eight Worker PIDs and verified disabled profiling.
Each implementation passed its ordinary six-case output check.

| Implementation / target | Actual arrival | Completion rate | Logical TPS | P90 | Qualification |
|---|---:|---:|---:|---:|---|
| Core-only / 12 RPS | 11.895 | 10.517 | 95722 | 40.531 s | rejected |
| Combined / 12 RPS | 11.892 | 11.403 | 103792 | 15.621 s | rejected |
| Combined / 11 RPS | 10.916 | 10.880 | **99026** | **4.173 s** | passed |

At the matched 12 RPS point, the combined implementation improved in-window
completion rate by 8.43% and reduced P90 by 61.46%, but both points failed the
SLA/queue qualification. The qualified combined point delivered **5.942 million
logical TPM**; all requests succeeded, and waiting/capacity/deferred trends were
not growing. This is the highest qualified point in the bounded final test,
not a proven global maximum or an indefinitely sustainable service rate.

This result is about 55.0% of the requested 180k logical TPS target. The target
was not reached. The earlier Core-only 133590 TPS result was not reproduced as
a current absolute baseline; its cause remains unresolved, so it must not be
merged with the final combined capacity claim. The historical 18k uncached peak
was not remeasured. Two separate final Torch captures compare current cached
and cold forward efficiency; they are not capacity measurements.

## Final cached/cold diagnostic

Both captures ran in the same API process and performed almost the same total
new-token work: 318500 cached-suffix tokens and 318570 cold tokens. They did not
remeasure a cold peak. For batches at least 95% of 8192 tokens, excluding the six
operator-probe batches, cached work used 28 batches of 8190 tokens; cold work used
12 batches of 8192 and 21 of 8078. The shapes are close but not identical.

| Per global new token | Cached suffix | Cold |
|---|---:|---:|
| CPU forward wall |68.504µs|60.819µs|
| CPU thread time |57.759µs|56.847µs|
| Actual model GPU activity union |91.048µs|81.981µs|

GPU activity includes collective waits, not just useful FLOPs. Summed kernel
categories per token were dominated by collectives (39.326/33.863µs), GEMM
(23.843/21.968µs), and MLA (8.245/4.816µs); FlashKDA was 2.309/5.889µs. Category
durations can overlap and must not be added to the GPU-union metric. MLA nonzero
and 4-byte DtoH counts were zero in both captures.

The hot profile linked all 284 measurement requests to HTTP, with 203 complete
external-restore chains; 81 requests lacked external stages and were not assigned
zero restore time. Submit→all-rank-ready wall P50 was 113.814ms; overlap with real
forward GPU activity had P50 92.409ms and uncovered time P50 15.806ms. Ready→first
actual forward had P50 1472.605ms, overlap P50 1381.495ms and uncovered P50 90.361ms.
These are separate percentiles, so they do not add. Uncovered time is not an
estimate of causal cache tax. The hot 14 RPS profile was above the final qualified
11 RPS point and includes overload queueing; its overlap cannot be generalized to
all loads.

The evidence supports further attention to forward and collective efficiency,
not another claim that lookup/read-lock dominates. It does not establish Free
Meal, a current 18k cold peak, or the cause of the cross-run absolute baseline shift.

All test API/FakeD processes were cleaned. The original daemon PID383180 and its
start time were unchanged, all eight ordinary L2 servers remained registered
at 72.75 GiB each, and node08 was untouched. LMCache's staging-view change remains
code/microbench validated but was not loaded into those live L2 processes.

Raw artifacts are retained in the shared BW1000 workspace under
`experiments/cache-free-meal-20261101/`,
`experiments/cache-free-meal-indexer-20261101/`, and
`experiments/cache-free-meal-mla-20261101/`. These are workspace paths, not files
in this repository. The final summary is
`experiments/cache-free-meal-mla-20261101/qualification-summary.json`.
