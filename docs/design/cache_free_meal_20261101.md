# Cache admission and forward overhead

## Remove idle model RPCs

Independent restores previously left a zero-token model RPC in the normal batch
queue even when splitting receive metadata removed all Worker work. These replies
could delay admission behind an in-flight forward.

Core now asks connectors whether normal metadata still requires a Worker step.
The default is conservative. MultiConnector retains the step when any child needs
it or asynchronous saves remain. Worker state updates, finished/preempted requests,
block zeroing/copies, encoder work, and flush/probe metadata still execute. Only an
empty step in independent-load mode is skipped. Core continues polling the oldest
model result and preserves scheduler-local output processing.

CP8 skips the command at the Core dispatcher, so all eight Workers retain the same
model RPC order. Stream restore and all-rank completion remain enabled.

### Validation

The new targeted checks passed: 29 Core engine/MultiConnector tests, seven LMCache
ownership/metadata tests, and nine Plugin sink tests. These include pending and
completed model results, local error/output delivery, repeated empty steps, and
retention of real Worker work. Relevant Ruff and diff checks passed.

Node07 used the original weights daemon (PID 383180, started
2026-10-25T16:34:29.147352108Z). C1/C4 covered six 32767-token requests with an
8192-token restored prefix. All 1632 restored KDA state pairs / 6528 components
matched their stored values exactly. All 48 request/rank continuation sequences
were contiguous and agreed across ranks, with four or five restored suffix steps.

All six top-1 outputs matched. The candidate's cold-versus-restored full-vocabulary
TV reached 1.15148%; two cases exceeded the existing 1% cold/restore gate. A second
run used the deployed bundle with the exact same prepared inputs, salts and stored
prefixes. Its cold/restore maximum was 0.77329%. Direct candidate-versus-deployed
restore TV was at most 0.78518%, with matching top-1 and the existing selected-token
logprob check passing. Performance testing proceeded under the user's allowance
for small final-output differences. This is a comparative acceptance result,
not a claim that the original cold/restore gate passed or logits were bitwise equal.

### Matched open-loop results

Both runs used GPU L1, 582 GiB Host L2, stream restore, CP8+EP8, 3561888728 KV bytes
per rank, an 8192 batch-token cap, 64 sequence slots, and the same 250 ms maximum
restore coalescing setting. Each request contained 9102 input tokens, including
8192 cached tokens, and produced one token through FakeD. The 300-second AIPerf
windows used the same Poisson workload and 15-second warmup. Qualification requires
TTFT P90 below 12 seconds and the existing server-queue checks.

| Target RPS | Actual arrival RPS, old / new | P90, old / new | New logical TPM | Qualification, old / new |
|---|---|---|---:|---|
| 14 | 13.71782 / 13.71783 | 6.297 / 2.841 s | 7,477,349 | pass / pass |
| 15 | 14.63790 / 14.63792 | 14.168 / 6.336 s | 8,015,429 | fail / pass |

At 15 RPS, completion rate increased from 14.17261 to 14.67705 requests/s. The
measured waiting-queue trend changed from +0.3570 to -0.1163 requests/s. The highest
qualified point in this candidate run is 133590 logical tokens/s, 7.25% above the
previous highest qualified point. Higher rates were not tested in this stage;
these finite windows do not establish an indefinite service capacity.

The separate full-batch check recorded 72 full batches, up to 8190 actual tokens.
The minimum sampled free HBM was about 0.849 GiB; all requests succeeded. Sampling
was device-wide every 200 ms and does not prove an instantaneous worst-case bound.

### What the profile establishes

The separate event profiles used almost identical actual arrival rates. At nominal
14 RPS, zero-token model executions inside the offered interval fell from 25 to 0.
Lookup completion to the first KV allocation attempt fell from 796.44 to 1.716 ms
at P50, and from 1456.02 to 380.63 ms at P90. The earlier report's count of 27 zero
steps covered a wider capture interval and is not the matching denominator.

Restore-ready to scheduling remained: P50 was 498.66 ms before and 504.38 ms after;
P90 was 878.03 and 805.37 ms. The complete external-restore cohorts contain 266 and
301 requests respectively, because L1/external path membership differs. These are
cohort distributions, not paired per-request deltas; stage percentiles cannot be
added. The change removed a major admission delay, but has not made cache free.

The run cleaned only its test API/FakeD processes. The original daemon and ordinary
582 GiB L2 servers survived; node08 was untouched. Raw evidence is retained in the
workspace experiment `experiments/cache-free-meal-20261101/`, including
`performance-analysis.json`, `numeric-baseline-comparison.json`, and the original
profiles. Profiling windows are excluded from capacity results.
