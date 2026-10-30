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
