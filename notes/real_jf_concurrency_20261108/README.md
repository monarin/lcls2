# Real Jungfrau: bounded concurrency comparison

This test reads the first **1,000 original events of mfx101555026 run 10** from
`/sdf/data/lcls/ds/prj/public01/xtc/`. It preserves all ten source streams;
Jungfrau is carried by five physical XTC files. No synthetic data was created.
The source JF image is `(32,512,1024)` uint16, 32 MiB per event. Physical JF
files contain 5, 6, 7, 7 and 7 panels (roughly 5–7 MiB per datagram).

The GpuTask declares `jungfrau.raw`, so the framework gathers a dense raw array
before the existing calibration kernel. The task converts panel presence to an
error flag and runs the existing accept/count and cube-sum kernels. It uses
frozen pedestal/gain/offset arrays, common mode off, and no masks. This differs
from sim10, which expands sampled pixels directly from field descriptors without
a dense raw gather. Both tests use the same event timestamps and original
calibration shapes, 100 cube bins (99 populated and one empty control), nominal
batch 7, 4 GiB framework budget, and 100 warmup events. Each CUDA stream owns a
6.25 GiB cube bank outside the framework budget.

At 512 KiB every real JF datagram is large, as it already is at the framework's
1 MiB default. The 8 MiB control deliberately makes every JF datagram small.
It is not the framework default. This tests the small-group credit restriction
on real data as well as checking whether avoiding it enables kernel overlap.

## Validation and setup failures

The final matrix includes a CPU reference pass and ten GPU cases, each capped at
1,000 events: one separate pixel acceptance, six clean runs (three per stream
count), one 8 MiB control, and two Nsight captures. Exact timestamps, full panel
presence, stream-bank counts and bin routing are checked throughout. Separate
pixel acceptance compares all pixels of raw and calibrated images against CPU
cversion-3 SHA256 at events 0, 500 and 999. Cube numerical sums are not newly
checked here; the accumulation kernels are reused unchanged, with exact counts
checked after draining.

Two preliminary attempts failed before any accepted throughput measurement.
Job 43825880 passed a changing NumPy `nda_raw` keyword into CPU calibration,
triggering its cached keyword comparison. The reference now calls calibration
normally with the event. Job 43826137 then caught that the GPU calibration helper
returns `(image, validity)`; the wrapper now unpacks the image. Failure logs and
the second attempt's report are retained here. These were benchmark wrapper
errors; no psana core/runtime or CUDA kernel was modified.

## Result: pressure drains disappear, kernel concurrency remains absent

Job **43826632**, sdfampere001, A100-SXM4-40GB, completed exit 0 in 2m53s.
All ten GPU cases passed. Nine preparation host tests and two SQLite-analysis
host tests passed; the latter exercise serial/overlapping streams and reject
missing kernels, mixed contexts and ambiguous launch correlation.

| Target | CUDA streams | Clean repeats | Callbacks | Warmup / steady pressure drains | Clean events/s |
|---|---|---|---|---|---|
| 512 KiB | 1 | 3 | 150 | 0 / 0 | 148.2–169.0; median 168.0 |
| 512 KiB | 2 | 3 | 150 | 0 / 0 | 153.6–186.2; median 185.8 |
| 8 MiB | 2 | 1 | 1,000 | 100 / 899 | 84.2 |

All two-stream cases split 500/500 events between banks. The 8 MiB control uses
one-event callbacks because two real JF datagrams cannot fit its target. Thus
the threshold changes batching as well as admission behavior; the rate difference
cannot be attributed solely to removal of pressure drains. The 100th drain is
counted as warmup because admission precedes the next task's warmup-boundary join.
Clean rates use 900 measured events plus completed GPU tail after a 100-event
warmup. This existing rate window starts inside the first measured task callback,
after the boundary join. That batch's input I/O/parsing and dense gather have
already been submitted; the join also completes its gather, excluding that
preparation from the denominator. It is therefore a callback-boundary rate,
not an exact 900-event interval starting before all input I/O. The full-trace
overlap result is independent of this timing boundary. Variability is visible
even within this allocation.

Separate Nsight Systems 2025.3.1 captures show **zero cross-stream kernel overlap**
at both targets, for both the four selected gather/calibrate/accept/sum kernels
and all recorded kernels. Each trace exercised execution streams 13 and 14.
Expected selected-kernel counts matched callbacks exactly: 150 × 4 at 512 KiB,
and 1,000 × 4 at 8 MiB. The CUDA-producing process has no warning/error diagnostics;
no-CUDA/no-NVTX warnings belong to child processes.

At 512 KiB, all **149** next dense-gather launch API starts occur **after** the
previous cube-sum device completion, by **22.245–59.684 ms**. At 8 MiB, all 999
handoffs also have positive gaps (5.697–16.558 ms). Work is not submitted early
enough for task-kernel overlap in these captures. This does not isolate which
read, completion wait, parsing or host-preparation operation causes the delay.
No core fix is included. The next investigation should inspect that interval.

This reproduces the missing kernel concurrency on full real JF data, as in the
[sim10 comparison](../mfx_expand10_512k_20261108/README.md). It also shows
admission drains under the 8 MiB small-group setting. This count-only experiment
does not distinguish small-group credit pressure from raw-slot exhaustion; the
[earlier sim10 diagnostic](../mfx_expand10_pressure_20261106/README.md) attributed
its failures to the credit path, but that cause was not separately instrumented
in this real-data run. The real run at the framework's default 1 MiB would already
classify these datagrams as large; we did not run a separate default-target case.

The earlier sim10 test used another node/time and a different I/O volume/path.
These rates are not a controlled real-versus-sim10 speed comparison. KvikIO is in
CPU-fallback mode; source cache is warmed by CPU preparation but uncontrolled.
Traces include warmup and perturb timing. No copy/compute-overlap, full numerical
cube acceptance, sustained performance or 10k qualification claim is made.

## Records

Full reports, native traces and SQLite exports:
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/real_jf_concurrency_20261108_r3/`

[validation-summary.json](validation-summary.json) is a coordinator-generated
compaction of all cases, deriving bin totals and callback counts. The full
per-case `report.json` files bound in artifacts.json are authoritative;
[artifacts.json](artifacts.json) binds scratch evidence by size and SHA256;
[reference.json](reference.json) holds CPU identities and pixel references.
Execution source hashes were rechecked across every accepted report, CPU module
reference and launch manifest. The job logs here normalize terminal progress
and trailing whitespace; the originals remain in the sibling `_logs` directory.
See [reproduction.md](reproduction.md) for exact commands, CSV queries and analysis
boundaries, and [status.json](status.json) for the independent review state.

Independent restricted Claude review **passed** code commit `e720d46a6`, with no
blocking findings. [Review responses](review-01/response.md) address four low
notes; the kernel-name-format robustness suggestion is deferred because the
captured exports use bare names. Closure changes only documentation/evidence.
