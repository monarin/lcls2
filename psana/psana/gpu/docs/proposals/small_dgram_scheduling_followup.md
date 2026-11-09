# Small-datagram scheduling: deferred follow-up

Recorded November 9, 2026 at the user's request. Keep sim10 performance work at
**gpu_bulk_target_bytes=512*1024** until this scheduling behavior is addressed.
This note proposes future work; it does not change the framework.

Two issues are coupled to the current small-datagram path:

- One live small input group per physical XTC file can constrain read-ahead and
  overlap between CUDA execution streams. This ownership restriction applies
  even with one CUDA stream. It is distinct from read-dominated timelines where
  independent streams are permitted to overlap but their compute does not.
- Group boundaries force changes to GPU execution size. An EB batch_size=100
  does not ensure a 100-event GpuTask invocation: the scheduler cuts at every
  next small-group boundary in any contributing file.

In the [reviewed bulk-target sweep](../../../../../notes/sim10_bulk_target_20261109/README.md),
all cases used 10,000 sim10 events, batch 100, one BD/GPU/CUDA stream:

| Target | Reader calls | Task invocations | Events per invocation | GiB/s |
|---|---:|---:|---:|---:|
| 512 KiB | 50,000 | 100 | 100 | 2.12–2.19 |
| 4 MiB | 9,454 | 4,356 | 1–5 | 1.56–1.59 |
| 8 MiB | 4,599 | 2,233 | 1–11 | 1.65–1.79 |

At 512 KiB the 0.5–0.7 MiB datagrams are classified as large: the pipeline can
submit 500 reads for a 100-event/five-file batch, then invoke the task once.
These are asynchronously submitted requests, not 500 simultaneous storage I/Os.
At larger targets they coalesce as small groups; peak submitted-but-not-retired
requests falls to five. Different per-file panel counts produce different group
boundaries (e.g. 5,6,7 events), whose union splits execution into shorter ranges.
The larger target changes ownership/scheduling as well as read size, so this
experiment does not isolate storage read-size effects or prove a sole bottleneck.

Future design should separate physical read coalescing, read-ahead allowance,
and GPU execution size. Allow useful read/compute overlap without forcing tasks
to execute every small read group. Preserve byte-bounded admission, transition
fences, per-event identity, group ownership until every consumer finishes, CUDA
completion-based reuse, and bounded memory. Validate one and two CUDA streams,
unequal per-file datagram sizes, tails/gaps/transitions, and multiple consumers.
Do not infer buffer release from Python iteration or weaken lifetime guarantees.
Report actual reader counts, execution sizes, pending bytes, CPU scheduling and
CUDA timelines when evaluating a fix. This work is deferred by the user.

The next authorized dataset is [100,000 repeated sim10 events](../../../../../notes/mfx_expand10_100k_20261109/README.md).
Its JF payload is approximately 313 GiB, close to the earlier 10k uncompressed
JF test. At batch 100 the per-batch input remains approximately 320 MiB, so this
increases total file working set and measurement duration, not the per-batch
read-ahead window. Maintain 512 KiB for its subsequent performance baseline.
