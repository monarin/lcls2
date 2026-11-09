# Sim10 bulk-read target sweep

Results pending. Scope: vary only `gpu_bulk_target_bytes` between 512 KiB, 4 MiB,
and 8 MiB in the same one-node, one-BD, one-GPU, one-stream staging benchmark.
The expected request counts from fixture offsets are 50,000, 9,454, and 4,599
respectively. All full cases read exactly 33,582,000,000 payload bytes
(31.275675 GiB), covering 10,000 events across five JF XTC streams.

Settings: EB batch 100, SMD batch 1000, eight KvikIO workers, 1 MiB tasks,
16 MiB bounce buffer, fallback ON, automatic direct I/O OFF, 6 GiB GPU budget.
All MPI ranks share all available physical cores, with NUMA interleaving;
exclusive A100 node with 700 GiB requested memory. The public GpuTask requests
sim10 seed descriptors, with no dense gather, calibration, expansion, cube,
user kernels, or timed payload D2H. This is framework staging throughput.

Three 200-event preflights precede two 10k-event runs per target, in order:
512 KiB, 4 MiB, 8 MiB, 8 MiB, 4 MiB, 512 KiB. Preflights warm compiler caches;
lazy allocations in each new process remain timed. Timing covers the MPI event
loop and final GPU drain. Setup, input hashing/copying, cache eviction, idle NIC
samples, and final cache measurements are excluded from the loop timer.

Input is the accepted sample10_10k fixture, copied with SHA verification into
this campaign's unique Weka FFB directory. No input NVMe staging. Only the five
bigdata payload files are evicted after DataSource setup and required to show
zero mincore residency; SMD and server caches are uncontrolled. Physical NIC
counters bracket the loop, with separate idle samples. NIC bytes include SMD,
protocol/background traffic, and the slightly wider measurement window; they
are not an exact cache-hit fraction. The temporary FFB copy is removed afterward.

The bulk target controls small-datagram classification and adjacent coalescing.
Raising it changes read size/count, small-group ownership, and GPU execution
subbatch sizes. Therefore this measures the net effect of bulk reads within the
current pipeline, not an isolated storage-read-size effect. Logical XTC streams
(files) are distinct from the one CUDA execution stream.

Acceptance compares exact event/callback timestamps to the frozen fixture,
read counts and size histograms to an independent offset-based prediction,
complete byte counts, no CPU payload reads or admission drains, drained resource
state, cold payload pages, NIC deltas, and identical runtime/settings/device.
Read counts mean GPU-reader requests to KvikIO, not KvikIO worker tasks or OS
syscalls. The independent prediction is bounded to batch 100, uses exact
adjacency, and breaks on intervening records/gaps. No production planner imports.
Scientific accuracy of the intentionally lossy seed representation is outside
this staging benchmark; this sweep does not inspect pixel values.

Job **44009313** is queued for an exclusive ampere node. Slurm's current estimate
is **2026-11-09 14:02:44 Pacific**, subject to change. It may use a backfill window
of at least 10 minutes, up to the requested 20-minute limit. This queue delay is
not a failed benchmark. No measured rates or call counts are available yet.

Implementation: `f849540905a19cc2c18d620702088ade120740fe`. Syntax checks passed;
hardware checks and post-run audit tests remain pending. Independent Claude code-only review passed on that implementation commit,
with no blocking findings. Hardware acceptance and review of native results
remain pending; see [review](review-01/review.json) and [responses](review-01/response.md).

Native output root:
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sim10_bulk_target_20261109_r1`.
Slurm logs use the same path with `_logs` appended. Scripts and frozen expected
request counts are committed under this notes directory.

After job completion, run with Python3.11 from xpp_drp_gpu_311:

```bash
python notes/sim10_bulk_target_20261109/audit.py /sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sim10_bulk_target_20261109_r1
python notes/sim10_bulk_target_20261109/test_audit.py /sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sim10_bulk_target_20261109_r1
```

Then inspect actual counts/rates/NIC/subbatch sizes, freeze the native evidence
in the detached review tree, and complete the results-review round. Do not infer
hardware acceptance from the code-review verdict.

An offline cross-check against the frozen B runtime's CPU read planner passed
for both 200 and 10,000 events at all three targets. It exactly reproduced the
independent expected read-size histograms and counts. This provides no measured
GPU I/O or timing evidence. See cpu-plan-check.json and the executed Python
source preserved in cpu-plan-check-command.txt.

The hash manifest also retains the old instrument.py as provenance for the
lineage of bulk_accounting.py; the active counter wrapper is bulk_accounting.py.
The inherited --preflight flag is unused by this sim10 campaign; its short
preflight cases are selected solely by --events 200.
