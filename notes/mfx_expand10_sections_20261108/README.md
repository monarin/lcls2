# One-/two-stream section timing

These are new diagnostic profiles, not a retrospective timing breakdown of the
uninstrumented [clean baseline](../mfx_expand10_baseline10k_20261108/README.md).
Job **43863315**, sdfampere026, A100-SXM4-40GB, completed 0:0 in 2m14s. One then
two execution streams ran sequentially in fresh serial processes. The existing
reviewed runner/profile hooks and frozen kernels/runtime were unchanged.

Same 10k sample10 fixture on Weka, no NVMe/DRAM preload, batch 7, 100 bins,
4 GiB framework budget, 512 KiB group target, KvikIO 26.10 CPU fallback/four
workers/four allocated CPUs. Each run passed 10k ordered event and bin/count
checks, 1000 warmup + 9000 measured events, 1500 callbacks (1000 x 7 + 500 x 6),
expected bank ownership, and zero admission drains. Source/runtime hashes were
rechecked after execution. No new full-pixel numerical validation was performed.

## What dominates

The single-stream case spends substantial host wall time waiting for input
completion (6.30 s), waiting to reuse its execution slot (5.95 s), and preparing
CPU events (5.79 s). With two streams, slot-reuse wait almost disappears
(0.025 s), but input-completion wait is the largest measured section (10.03 s).
CPU event preparation remains substantial (6.40 s). This supports a pipeline
limited by input/preparation work rather than sustained GPU kernel execution;
it does not isolate Weka as the sole bottleneck or establish storage saturation.

The [host timing table](cpu-table.md) partitions the main thread's wall time
using the innermost recognized NVTX span. Nested sections are subtracted, so its
rows sum to the diagnostic envelope. These are elapsed sections, including
blocking, not CPU utilization. CPU ancillary reads are separated from their
parent event-construction span before regrouping. Other instrumented work
includes task submission, consumer release, uploads/enqueues and final flush.

| Main comparison | 1 stream | 2 streams |
|---|---:|---:|
| Diagnostic envelope | 34.013 s | 34.360 s |
| Input completion wait | 6.299 s (18.5%) | 10.028 s (29.2%) |
| Execution-slot reuse wait | 5.945 s (17.5%) | 0.025 s (0.1%) |
| CPU ancillary reads + event construction | 5.788 s (17.0%) | 6.402 s (18.6%) |
| Outside instrumented sections | 6.595 s (19.4%) | 7.433 s (21.6%) |

The uninstrumented remainder includes Python/framework work and profiling
overhead; no CPU sampling was enabled, so it cannot be assigned more narrowly.
Input wait is a KvikIO future wait encompassing unfinished worker reads/H2D and
scheduling; it is not a measurement of physical Weka read latency. The increased
wait in this one two-stream capture is not proof that adding a stream slows I/O.

## GPU execution, separately

Actual Nsight device durations within the same envelopes:

| GPU section | 1 stream | 2 streams |
|---|---:|---:|
| Calibration | 6.242 s | 6.314 s |
| Sim10 expansion | 1.752 s | 1.856 s |
| Cube accumulation | 1.343 s | 1.350 s |
| Event acceptance/count | 0.032 s | 0.032 s |
| XTC walk + locator kernels | 0.060 s | 0.064 s |
| All kernels, interval union | 9.429 s (27.7%) | 9.616 s (28.0%) |
| All H2D copies, interval union | 1.758 s | 1.833 s |

Calibration is about two thirds of user-kernel time. The sim10 path has no
separate dense raw gather: descriptor-based seed access is part of expansion.
GPU parser kernels themselves are short; CPU parser preparation/enqueue costs
1.99/2.40 s and descriptor build/upload calls cost another 1.99/2.11 s on the host.
Neither is the duration of its GPU kernels alone.

No task-kernel overlap was observed in either steady envelope (the sum of
individual task durations equals their interval union). H2D totals include
metadata copies as well as input payloads. No D2H occurs in these envelopes;
validation copies happen afterward, with no per-event publication or full cube
export. Worker `Buffered pread` interval unions are 11.92/16.31 s, including
filesystem cache/OS effects. They are not exclusive wall-time contributions.
**Do not add GPU, copy, worker-read or host totals together.** They can overlap.

## Timing limits and reproducibility

The envelope starts at the first steady `expand_calibrate_call` NVTX range and
ends at the last instrumented main-thread range. This excludes 3.7/2.9 ms from
the native callback-window head/tail (34.0169/34.3627 s); it includes exactly
1350 launches of each user kernel for 9000 events. Parser counts are 1349 because
input preparation for the first measured callback already occurred.

Diagnostic warmed rates are 264.6/261.9 events/s. These are single instrumented
captures on another node; use the clean 257.3/283.6 events/s medians for the
previous baseline. CPU/CUDA/NVTX timing hooks perturb execution and this pair
does not measure a causal scaling factor. Cache state and external node/storage
activity were uncontrolled; no cache flush, cold-read or device-exclusivity claim.
The initial GPU snapshot shows 6373 MiB/100% activity; both native before_setup
samples show 474087424 bytes used (~452 MiB). Final snapshot is 0 MiB/0%.
These spot checks cannot establish continuous isolation.

Native artifacts:
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sections_20261108_r1/`

- `s1.nsys-rep`, `s2.nsys-rep`: view in Nsight Systems.
- `s1.sqlite`, `s2.sqlite`: read-only analysis inputs.
- `s1-report/`, `s2-report/`: native report and profile JSON.
- Logs: adjacent `sections_20261108_r1_logs/43863315.{out,err}`.

`run.sbatch SOURCE_DIRECTORY NEW_OUTPUT_DIRECTORY` reproduces the capture.
`python analyze.py NATIVE_ROOT --output NEW_SUMMARY_JSON` verifies acceptance,
partitions nested NVTX ranges, clips GPU/worker intervals and binds inputs by SHA.
Five host interval tests cover overlap, nesting, clipping, gaps and invalid
crossing ranges. `summary.json` records all detailed CPU/GPU timings and artifact
hashes; `source-verification.json` binds the unchanged source/runtime.

Next targeted measurement: distinguish remaining host work and input issue/wait
costs before tuning worker count or batching. This task makes no framework or
kernel changes and does not start the broader batch sweep.
