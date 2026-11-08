# Small real-JF pipeline with twelve calibration passes

Diagnostic derived from `../jf_small_timeline_20261108/`: same 64 original MFX
run-10 events and ten source files, actual MPI SMD0 + EB + one BD, one A100,
execution depth 2, batch 16, four-event subbatches, 1.25 GiB framework budget,
512 KiB read-group target, two cube bins, and KvikIO CPU fallback. No dataset,
core framework or installed runtime change. This deliberately inflated compute
workload is not a throughput benchmark.

Each task calls the original calibration once, then replays its same kernel
11 times into the same retained float32 output, reading original uint16 raw and
fixed constants each time. Cube count/sum executes once. Additional passes
allocate no full images and add no host synchronization. The first callback
copies one final calibrated image (64 MiB) for pixel validation after the final
pipeline join. The inherited first-two raw snapshots and separately labeled
single-pass resident control are retained. The wrapper intentionally uses the
example calibration's private kernel cache and exact ABI for this experiment;
it is not a proposed user-facing API.

Look at `PIPELINE_64_EVENTS`, after its first four warmup callbacks. The new
`calibration_repeat12_enqueue` NVTX range marks the CPU submissions. Kernel names
remain `calibrate`; expect 192 calibration kernels and 16 each of gather, count,
and cube sum. The resident control remains two single-pass batches and must not
be mistaken for pipeline evidence. Warmup includes the original callback-boundary
join after 16 events; no intentional join is introduced between steady callbacks.

Acceptance uses CUDA correlation IDs to pair each gather CPU API with its device
kernel, then matches the following cube sum in the same stream. A next-stream
launch **returning before** the previous sum ends demonstrates concurrent
submission. This is stronger than just overlapping host call entry/exit and is
independent of actual device kernel overlap, which is reported separately.
The analyzer requires such a handoff after warmup. All 64 event identities,
16 four-event callbacks, 4+4+4+4 splits, zero admission drains, 32/32 bank totals,
per-bin acceptance and the first final calibrated image's CPU-v3 SHA must pass.
No new full-pixel cube-sum validation is claimed.

Launch: `sbatch --output=EXISTING_LOGDIR/%j.out --error=EXISTING_LOGDIR/%j.err
run.sbatch ABSOLUTE_SOURCE_DIR NEW_OUTPUT_DIR`. Use the provided frozen runtime.
Analyze with Python 3.9+: `analyze.py SQLITE REPORT_JSON --output NEW_JSON`.
Native reports, compiler cache and job logs belong on scratch. The maintained
scripts, compact results and review records belong in this directory.

## Result and viewing guide

Job **43837236** completed exit 0 in 23 seconds on sdfampere040, A100-SXM4-40GB.
All acceptance checks passed on the first attempt. Native report (~1.2 MB):

`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/jf_compute_overlap_20261108_r1/jf-64-repeat12-b16-s4-pool2.nsys-rep`

In the BD process, expand main **python thread 329811 → CUDA API** and GPU
streams **13/14**. Zoom to **11.16–11.27 s**, inside `PIPELINE_64_EVENTS`.
Stream 14's cube sum finishes at **11.224653271 s**. Stream 13's gather CPU call
is **11.202731358–11.202744607 s**: it returns **21.909 ms before** that finish.
Stream 13's first calibration CPU call also returns early, at **11.202965574 s**,
and all twelve calibration calls finish enqueueing by **11.203119262 s**.
The cube-sum CPU call returns at **11.203302820 s**. Thus this is actual task
submission while the other stream still has unfinished work, not merely a
parser launch. GPU gather starts at 11.210830012 s and the first calibration
starts at 11.215872562 s; submission and execution are visibly distinct.

Seven of eleven adjacent handoffs among the twelve callbacks after warmup have
the next gather launch **return** before the previous sum ends. Two more examples:

| Previous → next stream | Next gather launch returns (s) | Previous sum ends (s) | Lead (ms) |
|---|---:|---:|---:|
| 13 → 14 | 10.948952326 | 10.951415680 | 2.463 |
| 13 → 14 | 11.050099512 | 11.054966973 | 4.867 |
| 14 → 13 | 11.202744607 | 11.224653271 | 21.909 |

Actual pipeline device overlap is also nonzero but small: **292,481 ns** across
selected gather/calibration/count/sum kernels, or **1,205,029 ns** including all
kernels (parser and helper kernels too), summed over the whole pipeline phase.
These are aggregate overlap durations, not single intervals. Most kernels still
execute serially. The separate resident control is 11.614718–11.624303 s on
streams 19/20 and is not the evidence cited above.

This establishes that the existing pipeline permits concurrent submission when
compute is extended. It does not establish that all input stalls disappear, or
that two streams deliver a throughput improvement. This run used a different
A100 node from the baseline; no controlled speed comparison is claimed.

The profiler reports no warning/error for the CUDA-producing process. Expected
child no-CUDA/no-NVTX and disabled CPU-scheduling diagnostics remain. As in the
baseline, UCX unmatched-tag warnings appeared after PASS; the job exited 0 and
exact event checks passed. Their cause is not investigated. Committed job logs
remove progress formatting, NULs and trailing whitespace; byte-exact originals
are hashed in artifacts.json. Native workload JSON is copied byte-for-byte.
