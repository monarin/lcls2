# Perlmutter Sim10 10k: two-stream bottleneck trace

The matched clean run reproduces the slowdown. Host timing attributes nearly
all of the additional two-stream time to input-group bookkeeping on the BD
thread. This is a staging-only workload: no expansion, calibration, cube,
dense raw gather, or user payload D2H.

## Matched results

Job 61072191, nid001129, one BD and one physical A100; same CPU affinity and
NUMA interleave policy for every case. Frozen install-B, batch 100, 512 KiB
bulk threshold, 6 GiB budget, KvikIO fallback ON, 8 workers, 1 MiB task,
16 MiB bounce buffer, auto-direct OFF. Each case reads 10,000 events,
50,000 datagrams, 33,582,000,000 bytes (31.275675 GiB), and invokes 100
callbacks of 100 events. The five files are unchanged.

| Run | One stream | Two streams |
|---|---:|---:|
| Clean loop seconds | 15.1488 | 24.8319 |
| Clean GiB/s | 2.0646 | 1.2595 |
| Host-timed loop seconds | 16.3067 | 26.6205 |
| Host timing overhead vs clean | 7.64% | 7.20% |

These are one matched pair per mode, not repeated medians. The previous accepted
matrix independently reproduced the same direction on another node.
Local page residency was zero immediately before each loop after DataSource
setup. This establishes local-page-cache-cold Lustre; backend cache state and
remote traffic are unverified. No NVMe staging was used.

## Where the extra time goes

Whole-loop exclusive BD-thread seconds, including startup; nested phases are
subtracted so these rows do not double count. They are wall time on the host
thread, not GPU duration or pure CPU execution time.

| Phase | One stream | Two streams | Increase |
|---|---:|---:|---:|
| InputGroupPool.poll | 0.6404 | 9.2928 | +8.6523 |
| InputGroupPool.issue, excluding children | 1.1665 | 2.7888 | +1.6223 |
| Read completion wait | 1.3397 | 0.9162 | -0.4235 |
| KvikIO pread submission loop | 1.4579 | 1.4939 | +0.0360 |
| Read-group wrapper, excluding children | 3.2908 | 3.1931 | -0.0978 |
| User descriptor-check callback | 1.7086 | 1.6633 | -0.0453 |

The first two increases sum to 10.2746 s, or 99.6% of the 10.3138 s
host-timed loop increase. This is a net share: the remaining phases net to
about +0.04 s, with offsetting changes of up to a few tenths of a second.
The clean increase is 9.6831 s. Instrumentation
perturbs the workload, so the 99.6% attribution applies to the timed pair;
it is not an exact decomposition of the clean pair. Explicit BD-thread timed CUDA
synchronization phases total about 0.0021 s in each run. Separately, the trace
sums 1.91 / 1.76 s of `cuStreamSynchronize` across all recorded host threads.
Its 50,000 calls match the read count and are consistent with KvikIO fallback
worker synchronization; that thread attribution is inferred, since the summary
does not group by thread. This sum is not a main-thread critical-path measurement
and does not increase with two streams.

## Mechanism

Each batch has 100 events across five physical XTC files: 500 input groups.
`GpuEventManager._issue_group_reads` plans slots once, then calls
`InputGroupPool.issue` once for every group. Each `issue` calls `poll`, which
iterates every retained group. Its supplied-slot check also scans retained
groups with `any(g.slot == slot ...)`.

With two execution streams, the previous batch's parsed windows remain
retained while the next 500 groups are issued. The source path indicates that
repeated polling visits those windows even when live references make them
ineligible for retirement, entering `InputWindow._try_retire` to take its lock
and check state. This is a code-path inference; `visited_states` counts all
pool entries, including window-less groups, not individual `_try_retire` entries.
Both traces record 150,000 `cudaEventQuery` calls, so the extra state visits do
not produce an increase in the total recorded CUDA event queries.
This consumes BD-thread time before more reads or parser work can be submitted;
it does not require a CUDA wait on the other stream.

| Poll counter | One stream | Two streams |
|---|---:|---:|
| Calls | 50,101 | 50,101 |
| Group-state visits | 12,524,500 | 37,323,500 |
| Largest retained group set | 500 | 1,000 |

That is 24,799,000 additional visits. Fresh groups without parsed windows are
cheap to skip; retained windows incur the retirement checks. The extra
`input.issue` self time is consistent with the supplied-slot linear scan,
but that expression was not separately timed, so its exact share is not proven.

Runtime source snapshots in `frozen/` are byte-identical to this checkout.
The timer records source hashes for the functions it instruments.

## Nsight timeline cross-check

The captures contain 100 `walk_xtc`, 100 `init_locators`, and 100 `locate_fields`
kernels in each case. The one-stream case uses Nsight stream 13; two streams
split those kernels 50/50 across streams 13 and 14. Total kernel activity is
only 7.05 / 7.14 milliseconds. The union of traced GPU kernel and copy activity
is 1.552 / 1.464 seconds within the captured BD loops. These are traced activity
unions, not hardware utilization or end-to-end I/O time.

For a concrete comparison, zoom to zero-based batch 20:

| Capture | `psana.group.issue` interval | Duration | Nested polling | GPU copy/kernel union |
|---|---|---:|---:|---:|
| One stream | 3.787267–3.869402 s | 82.14 ms | 7.29 ms | 15.54 ms |
| Two streams | 6.014671–6.202366 s | 187.69 ms | 94.19 ms | 14.64 ms |

Each interval contains 501 polls (one at planning plus 500 issue calls), and
neither contains GPU kernel execution. In the two-stream trace, the next
`walk_xtc` starts on stream 13 at 6.218748 s. Batch 21 issues reads at
6.278648–6.470008 s, spends 95.10 ms polling, then launches `walk_xtc` on stream
14 at 6.486114 s. Use the BD process's NVTX row and CUDA stream rows in Nsight.
The gaps are consistent with host submission overhead, not long-running GPU
kernels holding the other stream back.

The Nsight receipts report 35.35 / 45.53 s elapsed, but the `bd.loop` ranges are
16.91 / 28.03 s. The accepted outer timer includes profiler start/stop and the
final MPI barrier. This additional elapsed time lies outside the captured
`bd.loop`; start, stop, and barrier costs were not timed separately. Do not use the profiled receipt's GiB/s as workload
throughput. Even the captured body is instrumented; clean runs remain the
performance reference. `native/timeline-analysis.json` and `analyze_trace.py`
record the SQL queries, activity unions, and example coordinates.

Viewable reports (Nsight Systems 2026.2.1):

- `04-nsys-s1.nsys-rep`
- `05-nsys-s2.nsys-rep`

Copies also reside on SDF under
`/lscratch/monarin/tmp/sim10-trace-artifacts-20261110/`.

## Next scoped fix

Use a maintained busy-slot set for constant-time supplied-slot membership, and
avoid a full-pool poll for each member of an already planned issue batch.
Poll at explicit scheduling/retirement boundaries while preserving input holds,
small-stream credits, deferred CUDA retirement, and error cleanup. Validate
ownership/backpressure tests, then repeat this exact clean 10k pair. This trace
does not implement that optimization or claim a measured speedup from it.

## Artifacts and review

Implementation: ec0941a31222045e31e5a8ea8fb49c02c4a63a28.
Independent implementation review: PASS, no blocking findings. All six native cases and Slurm steps passed (job elapsed 5:43). Independent
results/analysis review: PASS on
`3b1e9b3ac2d4e0d45c660be8097b1c9c3111c76b`, no blocking findings.
The final evidence-only commit records that verdict and applies four requested
documentation clarifications; runtime and offline analysis code are unchanged. Large report root on Perlmutter:

```
/pscratch/sd/m/monarin/psana2-gpu/sim10-trace-20261110/attempt01/results
```

`native/` contains the exact compact JSON receipts; raw case logs and viewable
Nsight reports remain at the artifact root. The profile runs are diagnostic;
use clean runs for throughput. See `scope.md`, `status.json`, and the review
folders for the bounded review-loop record.
