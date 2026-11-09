# Sample10 psana pipeline versus cold read baseline

Completed measurements on sdfampere023. Independent Claude review passed at
`118dcb5d3d563a4996b6669fd702bd6d51727fd8`, with four non-blocking findings
addressed as documented limitations. [Review](review-01/review.json),
[responses](review-01/response.md), [scope](scope.md).

This harness uses the accepted 10k synthetic MFX fixture and the existing
expansion, calibration and 100-bin/99-active cube kernels unchanged. Each run
is a fresh MPI process group: one SMD0, one EB and N BDs. All BDs see exactly
one physical GPU selected by UUID before MPI initialization. Each BD is pinned
to four distinct physical CPU cores; each helper has one. BDs increase total
CPU allocation and KvikIO workers (four per BD), not just process count.
Device PCI/mask, CPU affinity, runtime hashes and effective settings are saved.
Each four-core BD budget hosts its main thread plus four KvikIO workers and
runtime helper threads. Most measured host categories have nearly equal wall
and main-thread CPU time, indicating little main-thread descheduling there;
background-worker contention was not separately measured. A 5–6 core/BD or
fixed-total-core comparison remains a possible follow-up.

Both cold and warm cases have fresh GPU contexts. Whole-loop time includes the
start barrier, event delivery, lazy first-call allocations/calibration setup,
and all BD GPU drains, but excludes DataSource setup, explicit cache conditioning,
NIC sampling, acceptance readback and cleanup. There is no warm GPU timer,
per-event D2H, full-cube export, or cross-BD cube reduction in these rates.
Scaling is aggregate input/processing, not a complete distributed-cube product.

Before cold runs, only the ten owner-controlled synthetic bigdata files are
fsync'd, FADV_DONTNEED-evicted and checked with mincore. All must have zero pages
resident. SMD buffers may already be primed during DataSource setup. Warm cases
explicitly read the ten files outside timing and require >=99.9% residency.
Weka backend caches are uncontrolled. No global cache flush or data mutation.
Physical ethtool counters bracket the loop, with their own timing interval;
idle counter endpoints are also retained. Ports remain separate; NIC includes
all detectors and other host traffic. Do not infer exact cache-hit percentages.

Every event has 3,358,200 JF datagram bytes. The 10k JF numerator is
33,582,000,000 bytes (31.275675 GiB); the ten bigdata streams also contain
4,090,281,368 event bytes for CPU-routed ancillary detectors. Physical NIC
bytes and file sizes (including transitions/SMD) are distinct denominators.

A read/parse-only control exercises field descriptors and XTC parsing but skips
user expansion/calibration/cube kernels and their constants/banks. This changes
both compute and memory demand; it is a bottleneck diagnostic, not scientific
output or a pure single-factor CPU/GPU attribution.

The optional diagnostic wraps existing framework hooks with nested exclusive
main-thread wall and thread-CPU timers and CUDA events. It is separate from
clean throughput cases. CUDA intervals can overlap and include submission gaps;
their sum is not device busy time. Host exclusive categories leave uninstrumented
Python/framework work explicitly outside attribution.

1/2 BDs compare one and two streams; 4 BDs use one stream. Each stream's cube
bank is 6.25 GiB, so tested configurations have at most four banks (~25 GiB)
with an aggregate >=8 GiB non-bank reserve gate. Snapshots are not continuous
memory peaks. The pre-loop reserve check uses a per-BD free-memory snapshot
while other contexts may still initialize, so it is order-dependent rather than
a synchronized reservation. prepare() compiles kernels; cube banks allocate
later during callbacks. Recorded post-loop snapshots also exceeded the 8 GiB
reserve, but do not establish the minimum free memory throughout execution.
The 4 BD/2 stream configuration would need 50 GiB of banks alone
and is outside this A100-40GiB test. Framework budget remains 4 GiB per BD,
nominal EB batch 7, SMD grouping 20, bulk read target 512 KiB, forced KvikIO CPU
fallback, four workers/4 MiB tasks/16 MiB bounce/no auto-direct per BD.

Native evidence is under
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/pipeline_scaling_20261109_r3`.
Job 43977007, exclusive sdfampere023. Pilot maximum is 400 events. The campaign
runs two pilots, a same-node standalone read control, 18 clean full-10k cases,
and three diagnostics. There are two cold/warm repetitions for full 1BD/2stream,
2BD/2stream and 4BD/1stream; other controls have one cold/warm pair. This is a
bounded scaling screen, not an exhaustive sweep or confidence-interval study.

Preserved failed attempts: job 43976569 (`..._r1`) had insufficient advertised
MPI slots (no read); job 43976777 (`..._r2`) lacked the process-local synthetic
reader registration before DataSource setup (no event loop). Corrected by
requesting six Slurm task slots and registering the existing adapter. No core
psana or detector kernel changes were made.

## Result and recommendation

**Use two BDs per GPU with two execution streams each as the next baseline for
this 100-bin workload.** It improves cold input processing by **57.1%** over one
BD/two streams, with eight physical worker cores instead of four. Four BDs/one
stream gain only another **8.3% cold** and provide no warm-cache benefit, while
using sixteen worker cores. These are measured configurations, not an exhaustive
optimum. Both multi-BD choices hold ~25 GiB of persistent cube banks. More bins
require revisiting the memory budget before increasing concurrency.

Exact JF datagram bytes per completed whole loop determine all GiB/s below.
The two helpers use two additional physical cores, independent of worker cores.

| Task | BDs | Streams/BD | Worker cores | Cold GiB/s | Warm GiB/s | Repeats per cache mode |
|---|---:|---:|---:|---:|---:|---:|
| Full | 1 | 1 | 4 | 1.058 | 1.280 | 1 |
| Full | 1 | 2 | 4 | **1.231** | **1.537** | 2 |
| Full | 2 | 1 | 8 | 1.540 | 1.593 | 1 |
| Full | 2 | 2 | 8 | **1.934** | **2.248** | 2 |
| Full | 4 | 1 | 16 | 2.095 | 2.235 | 2 |
| Read/parse only | 1 | 2 | 4 | 1.255 | 1.625 | 1 |

For repeated cases, entries are medians (two samples); raw ranges and rates are
in `results.json`. The recommended configuration processes **618 events/s cold**
and **719 events/s warm**. The 4BD/1stream cold range is 2.087–2.103 GiB/s;
2BD/2stream is 1.918–1.949 GiB/s. Warm ranges overlap (2.225–2.245 versus
2.244–2.251 GiB/s); do not claim a meaningful warm advantage for four BDs.
The shared ~13.9 s warm-loop plateau is not attributed by this investigation.
GPU sharing, single-core SMD0/EB helpers and host bandwidth are possible limits.
A 4BD diagnostic or helper-core variation would help distinguish them.

The same-node standalone four-worker KvikIO control measured **4.120 GiB/s cold**
and **12.843–12.868 GiB/s warm**. This control keeps a continuously replenished
35-request ring, reads only JF files and does no psana event handling or user
compute. It is not an allocation of 4 GiB/s per GPU or a multi-GPU/node ceiling.
All clean full-pipeline cold cases received **36.805–37.385 GiB** on the physical
NIC, consistent with 31.276 GiB JF plus 3.809 GiB ancillary event bytes and
protocol/readahead overhead. Warm cases received only **14–24 MiB**. The network
counter evidence supports the intended local cache distinction.

## Why the read-only rate falls in psana

The strongest evidence is the ablation: omitting expansion, calibration and cube
kernels improves the 1BD/2stream rate only ~2% cold and ~6% warm relative to the
full-case medians. This supports **host input/event preparation as the main
1BD/2stream bottleneck**, with the GPU computation largely hidden behind that
work. This conclusion does not imply user compute is irrelevant after adding BDs.

A focused cold 1BD/2stream diagnostic attributes 27.189 of its 27.429 seconds,
leaving only 0.240 s outside instrumented sections. Exclusive main-thread CPU
time totals **23.758 s**, about **86.6%** of wall time. That includes CPU work in
Python, libraries and driver calls, not only Python execution. It does not count
KvikIO background worker CPU time. The diagnostic is slower than the clean
25.40 s median, so these measurements identify costs rather than provide exact
percentages of an uninstrumented run.

| Exclusive host category | Seconds |
|---|---:|
| Ancillary event initialization, datagram construction and CPU reads | **7.153** |
| Subbatch/input-group planning, issue bookkeeping, parser/descriptor preparation and group submission bookkeeping | **8.478** |
| KvikIO submission + future waits | **6.309** |
| Remaining measured controller/task/lifecycle work | 5.249 |
| Outside instrumented sections | 0.240 |

The KvikIO row is 1.785 s submission plus 4.523 s future waits. Future waits
include 1.421 s of main-thread CPU time; they are not pure disk/network waiting.
MPI batch receive/unpack/request is only **0.199 s** (within the remaining-work
row), while producer execution-stream retirement waits are only **0.013 s**.
There are zero pressure-triggered drains in every accepted case. This refines the
earlier broad 'read dominated' description: **the host input pipeline dominates,
not simply time blocked waiting for Weka or for another GPU stream to finish**.

The focused diagnostic has a review-discovered instrumentation caveat:
the inherited generator wrapper discards StopIteration.value from _handle_steps.
This defers the EndRun-triggered final flush to finish(), changing its attribution
while retaining the final drain inside the measured loop. Event/count acceptance
passed, and both diagnostics recorded five slot flush calls. Clean throughput
cases and the main campaign diagnostic hooks are unaffected. The native focused
report note claiming unchanged controller behavior is too broad; it is preserved
as recorded evidence with this correction. A future reuse of the focused wrapper
should forward generator return values before collecting new timings.

[host-timing.md](host-timing.md) contains all disjoint host categories. GPU CUDA
intervals are separately retained; do not add them to host time or sum them across
BDs to infer GPU busy time. No new Nsight timeline or utilization union was
collected here. Input request high-water counts are submitted-but-not-yet-joined
requests, not proof that all requests execute physically at once.

The current code issues the first GPU read before CPU ancillary handling, then
waits for inputs, prepares parsed field descriptors and calls the task before
advancing. The standalone ring can refill as each read completes. Relevant
unchanged code: [batch controller](../../psana/psana/gpu/gpu_events.py#L945),
[input-group issue](../../psana/psana/gpu/gpu_events.py#L813),
[subbatch planning](../../psana/psana/gpu/gpu_events.py#L578), and
[CPU EventManager](../../psana/psana/psexp/event_manager.py).
The installed `gpu_events.py` and `event_manager.py` were SHA-verified identical
to these checkout files. In this simulated decompression path the task consumes
field descriptors directly; there is no separate contiguous-seed/raw gather
before expansion. Two GPU streams permit overlap, but do not parallelize the
Python preparation thread. Multiple BDs provide independent host processes and
additional worker cores, which explains the measured improvement in conjunction
with the stage timings; a fixed-total-core process-count comparison was not run.

The next optimization experiments should reduce the repeated CPU work around
1,500 small callbacks: batch/subbatch sizing and avoiding ancillary detector
materialization that the application does not need. In this fixture the CPU-routed
streams contribute 4.09 GB per 10k events; one stream alone contributes 3.96 GB.
These are proposed next experiments, not implemented speedups.

## Bridge to the earlier sub-1 GiB/s result

The original runner used serial psana. A separate same-node follow-up ran that
unchanged runner with cache/NIC brackets, then MPI 1BD/2stream with the same
four-core worker affinity, task and explicit KvikIO settings:

| Mode | Cold GiB/s | Warm GiB/s |
|---|---:|---:|
| Original serial runner, whole loop | **1.192** | **1.498** |
| MPI 1BD/2stream, whole loop | **1.242** | **1.540** |

This does **not reproduce the earlier sub-1 GiB/s absolute rate**. Serial versus
MPI accounts for only a small part of the difference on this controlled node.
The older node/cache/CPU placement conditions were not controlled identically;
we cannot assign that historical difference to one cause. Also, the old ~0.995
GiB/s conversion used all stored fixture bytes per event, while this report uses
exact JF datagram bytes (the old 283.58 Hz corresponds to ~0.887 GiB/s JF).
The large read-only versus full-pipeline gap remains in the controlled tests.
The serial steady timer is retained in native evidence but is not compared with
the MPI whole-loop timer. Fresh GPU startup/lazy initialization is included in
both whole-loop modes, and compiler-cache history is not identical between the
main campaign and follow-up allocations.

## Validation and evidence

All 23 main MPI cases (two pilots, 18 clean cases and three diagnostics), the
standalone three-pass read control, four serial/MPI bridge cases and focused
10k diagnostic passed. Every full 10k MPI case has the exact global timestamp
partition, exact accepted per-bank/bin counts, 50,000 completed JF requests,
33,582,000,000 JF bytes and zero pressure drains. Main-source/runtime hashes
are stable across cases. Fourteen CPU audit tests pass, including negative tests
for duplicate events, false cache/rate claims, wrong banks, missing read bytes,
NIC resets and overlapping CPU affinity. No new full-pixel cube comparison or
cross-BD cube export was performed; numerical kernel acceptance is inherited
from the earlier reviewed prototype.

Independent summary artifacts: [results.json](results.json),
[followup-results.json](followup-results.json), [tests.log](tests.log), and
[slurm-accounting.txt](slurm-accounting.txt). Some MPI logs contain UCX
'unexpected tag-receive descriptor' warnings at shutdown; all accepted jobs exit
0, and exact event/count checks pass. These warnings are retained, not treated
as evidence of missing data or repaired by this benchmark. The existing generic
fallback warning says NVMe; these actual input files are read from Weka without
NVMe staging.

Additional native artifacts:
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/pipeline_serial_20261109_r1`
(job 43977453, 3m23s) and
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/pipeline_detailed_20261109_r1`
(job 43979257, 1m01s). Main job 43977007 completed in 19m23s. All three ran on
exclusive sdfampere023 allocations, using GPU UUID
`GPU-52ed4d03-f40c-877b-7455-4abb19f74cf3` only. The node's other GPUs were reserved by exclusivity
but were not used by the benchmark.
