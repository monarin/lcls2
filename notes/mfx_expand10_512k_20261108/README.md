# 512 KiB sim10 rerun: admission stalls removed, kernel overlap still absent

The user requested a 512 KiB read target and another concurrency check.
The sim10 performance runner now defaults to `gpu_bulk_target_bytes=524288`.
The framework default stays 1 MiB. This changes read grouping only: the five JF
files still contain the same sampled pixels, calibration and event identities.
The earlier diagnostic wrapper retains its explicit 8 MiB/512 KiB controls.

A datagram smaller than the target enters the small-group path. Each sim10 JF
datagram is 524,724–734,604 bytes, so at 512 KiB all five files use large groups.
This bypasses the one-outstanding-small-group credit and increases the group
count from 790 coalesced groups to 5,000 per-file/event groups in this 1k fixture.
This is a workload setting, not a general repair of the small-group policy.

## Native rerun

Job **43823340**, sdfampere034, A100-SXM4-40GB, completed exit 0 in 1m57s.
All nine native cases passed exact timestamp/event/bin checks for 1,000 events.
Nine existing host tests and four controlled device cases passed. Three host
tests also verify the trace overlap calculation (distinct streams, half-open
interval boundaries, idle gaps, and invalid/empty input rejection).

Same frozen runtime/calibration, 100 bins, nominal batch 7, 4 GiB framework
budget, 100 warmup + 900 measured events, KvikIO compatibility ON (CPU fallback).
Pilot verification warms file cache. Three fresh-process repeats per stream
count run alternately on the same GPU. The 8 MiB control has one clean repeat.

| Target | CUDA streams | Clean repeats | Warmup/steady pressure drains | Clean events/s |
|---|---|---|---|---|
| 512 KiB | 1 | 3 | 0 / 0, every repeat | 386.7–392.2; median 387.1 |
| 512 KiB | 2 | 3 | 0 / 0, every repeat | 486.1–489.0; median 487.4 |
| 8 MiB | 2 | 1 | 15 / 142 | 438.1 |

Two-stream 512 KiB cases distribute 500 events to each bank, with 150 callbacks.
The control uses 158 callbacks and 501/499 bank events. The local median is about
11% above the one 8 MiB control and 26% above the one-stream median, but these
short warmed tests do not establish sustained performance or explain the gain.
Do not infer concurrent GPU kernels from the rates or the zero-drain counters.

## Actual CUDA overlap

Separate Nsight Systems 2025.3.1 captures use throughput mode with no additional
Python timing patches, at two streams for both targets. Traced rates are excluded
from the clean table. The analysis reads CUPTI device kernel start/end intervals
and counts union time with at least two distinct active stream IDs. It covers the
**full trace including warmup**, not a separate steady window. All four task
kernel counts match the report callback count on the two execution streams.
The CUDA-producing process has no Nsight warning/error diagnostics; recorded
no-CUDA/no-NVTX warnings belong to other child processes. See
[trace-diagnostics.json](trace-diagnostics.json) and each summary's `global_pid`.

| Target | Task kernels | Task streams | Cross-stream task-kernel overlap | All-kernel overlap |
|---|---|---|---|---|
| 8 MiB | 632 (158 × 4) | 13, 14 | 0 ns | 0 ns |
| 512 KiB | 600 (150 × 4) | 13, 14 | 0 ns | 0 ns |

For each execution, expansion and accumulation kernels are paired in FIFO order
within its CUDA stream; the expansion launch API is matched by correlation ID.
In the 512 KiB trace, **all 149 next expansion launches begin after the previous
cube-sum kernel finishes**. The minimum gap is 4,114,750 ns (~4.11 ms).
Thus the next task is not queued early enough to overlap in this capture; absence
of overlap cannot be attributed solely to both kernels competing for GPU resources.
This does not isolate which host/I/O/preparation stage causes the delay. The
trace includes many KvikIO stream synchronizations and other host work; their
inclusive API totals are not causal or critical-path attribution. Copy/compute
or CPU/GPU overlap was not quantified by this kernel analysis.

The requested setting removes admission drains, **but does not establish or
restore concurrent task-kernel execution in this test**. Next investigate the
interval between one sum and the next expansion: read completion, parsing,
metadata upload and host synchronization. No core or kernel fix is included.

## Reproduction and evidence

`run.sbatch` runs host/device checks, six clean default-target cases, the 8 MiB
control, and two traces. `analyze_trace.py` verifies the trace has one GPU context
and process, every expected task kernel, and both execution streams; absent
trace activity is an error rather than a zero-overlap result. The overlap
summaries and the three-test log are committed here. Full reports, native traces
and SQLite exports are under:

`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/target512_20261108_r1/`

[artifacts.json](artifacts.json) records paths, sizes and SHA256.
[validation-summary.json](validation-summary.json) records all nine cases;
[execution-source-hashes.json](execution-source-hashes.json) was rechecked against
the files actually executed. Scheduler logs here have terminal progress lines
and trailing whitespace normalized; originals remain in the sibling `_logs`
directory. See [status.json](status.json) for independent review status.

For this runner, the default is sufficient; the explicit equivalent is:

```bash
python runner.py ... --bulk-target-bytes 524288
```

The framework's small-group policy, its byte budget, stored data, calibration DB,
and GPU expansion/calibration/cube kernels are unchanged. No 10k campaign ran.
