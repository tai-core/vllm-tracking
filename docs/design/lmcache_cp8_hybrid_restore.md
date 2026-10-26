# CP8/EP8 LMCache support: Core contracts and acceptance

The paired Plugin connector restores replicated MLA/Indexer cache and
head-sharded KDA state using the existing LMCache MP transport. Core contributes
two small changes; the resident weight producer and IPC protocol remain in use.

- `KVCacheGroupSpec.kv_cache_spec_for_connector` retains the original copied
  `UniformTypeKVCacheSpecs` before scheduler configuration selects a representative
  spec. This preserves distinct MLA/Indexer byte layouts for connector identity
  and validation without changing scheduler planning or source worker specs.
- The explicit `kv_cache_memory_bytes` startup branch releases inactive allocator
  memory after `profile_run()`, before KV allocation/export. The automatic
  profiling path already performs this cleanup. Native logs identified an RCCL
  P2P allocation failure requesting 2,822,144 bytes at the first long KDA A2A;
  each worker released 4.25 GiB of cached profile allocations after this fix.

The existing producer successfully served six fresh zero-copy client lifetimes
across numeric and performance tests, with 1,418 tensors mapped on every rank and
checkpoint fallback disabled. Normal nonblocking runs passed after the startup
cleanup; the producer was not restarted, and node08 was not accessed.

Validation on the paired TP1/PCP8/EP8, eager BF16, DeepGEMM/FlashKDA configuration:

- Cold input: 32,767 tokens; saved external prefix: 8,704 tokens; suffix: 24,063.
  Fresh engines reported exactly 8,704 external-hit tokens per request and zero
  local APC hits. All eight ranks covered the suffix in three continuous steps.
- 544 KDA state pairs / 2,176 conv-Q/K/V and recurrent component hashes matched
  exactly before continuation. Two model cases preserved top1 and correct answers.
- Full-vocabulary maximum TV was 0.2125%; raw logits were not identical (maximum
  absolute difference 1.484375 and relative L2 10.83%). These finite cases do not
  establish general accuracy or arbitrary mixed-workload equivalence.
- Matched C1 logical throughput was 12,019.54 → 15,165.00 input tokens/s, with
  TTFT p95 3.0891 → 2.4660 seconds. C2 throughput regressed 4.55%, and p95 rose
  4.20%; these negative results are retained. Logical throughput includes cached
  tokens and is not hardware compute throughput.

Targeted CPU validation: two scheduler-layout tests, nine worker-memory tests,
and 94 paired Plugin tests passed. The worker tests used
`--confcutdir=tests/v1/worker` to avoid an unrelated macOS accelerator teardown
fixture. GPU model, state, cleanup and performance evidence is recorded in the
[paired acceptance report](https://github.com/tai-core/vllm-plugin-das-glm53-bw1000/blob/392dfa983753967e9989a0e5d954b2e8db804247/docs/ep-lmcache-pcp-20261026.md).

This is bounded Prefill plus one output token, aligned scratch-empty recovery and
same-topology reuse. Complete Decode, arbitrary nonempty KPool tail handoff,
cross-topology P/D and production SLOs were not evaluated. Implementation and
testing were AI-assisted; the paired PRs remain Draft for human review.
