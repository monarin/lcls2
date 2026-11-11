# 100k-event matched baseline on Ada

Job **44189224 completed successfully** on sdfada016, November 10,
16:04:39–16:20:29 PST (15m50s total). Automatic audit passed; independent
results review pending. Measured harness: `7a727fd242754338be1720c457e579e6278edeb6`.
PR runtime: `67c327a148a08b2aaf972e67e1e02e367c2b298f` (PR #298).

| Streams | Repeat 1 GiB/s | Repeat 2 GiB/s | Median GiB/s | Median events/s | Median loop seconds |
|---|---:|---:|---:|---:|---:|
| 1 | 3.0187 | 3.0199 | **3.0193** | 965.4 | 103.59 |
| 2 | 2.8619 | 2.8783 | **2.8701** | 917.7 | 108.97 |

Full-pass order was 1,2,2,1 after 1000-event pilots (2.872/2.919 GiB/s).
Each full pass reads **312.7567 GiB**: 100k events, five JF streams,
500k physical requests, 335820000000 bytes, 1000 callbacks of 100 events.
Two streams delivered **4.94% less throughput** than one in this baseline.

The completed 10k baseline on the same node, physical GPU UUID, 24 CPU cores,
pinned runtime and settings measured 3.4094/3.2694 GiB/s (one/two streams).
The 100k medians are 11.44%/12.21% lower. Increasing the input tenfold did
**not** improve one-BD bandwidth here. This is two repeats per setting in
separate jobs, not proof that event count caused the difference, a storage
ceiling, or an explanation of the two-stream penalty. Weka conditions/backend
cache state remain uncontrolled. No kernel-overlap claim is made.

Controls: one L40S (`GPU-075afbaf-4e77-c882-8d56-bd1dc489b71a`), one BD,
24 physical CPU cores 0–23, NUMA interleave, batch100, SMD1000, 512 KiB bulk
target, 6 GiB input budget. KvikIO26.10 fallback ON, eight workers, 1 MiB tasks,
16 MiB bounce buffer, automatic direct I/O OFF. Descriptor staging only:
no expansion, calibration, cube, or payload D2H; XTC/descriptor processing is
included. Exclusive Slurm allocated ten GPUs, but only the selected UUID was
exposed to the BD; helper ranks had no visible GPU.

All four full passes have zero resident JF pages after initialization and
conditioning. Physical NIC `enp69s0f0np0` received **330.476–330.593 GiB** per
pass (1.05665–1.05703 times requested JF bytes); idle RX was 0.996–1.079 MiB/s,
and the other three monitored interfaces had zero receive bytes. NIC brackets
are slightly wider than loop timers. This supports network-backed delivery;
it does not measure an exact cache-miss fraction. SMD is not cold-gated and
backend caches are uncontrolled. No NVMe staging was used.

Validation: 22 local harness checks; 31 GPU tests passed with zero skips;
all six cases and automatic audit passed; exact timestamps/counts/runtime
identities/callback sizes verified. Every full pass had 500 requests and
335820000 bytes peak outstanding, zero admission-pressure drains, zero psana
CPU detector payload reads, and fully closed/drained resources. KvikIO fallback
still reads via CPU memory. Post-loop CuPy pool totals were ~329 MiB/658 MiB
for one/two streams, snapshots rather than peak GPU memory.

Accepted fixture (unchanged):
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sample10_100k`.
Its 10k source payloads are repeated ten times with unique outer timestamps.
The ~313 GiB owner FFB copy was removed; cleanup.json and a direct existence
check both confirm removal. Raw results, logs, receipts and source snapshots:
`/sdf/scratch/users/m/monarin/gpu-validation/sim10-ada-100k-baseline-20261110/attempt01`.

See results.json for compact per-case evidence, audit.json, runtime-evidence.json,
artifact-index.json for original paths/SHA256, slurm-job.txt and scope.md.
Full timestamp arrays remain in scratch, with exact-identity booleans and
raw-result hashes recorded here; an independent file-only reviewer cannot
rehash the external fixture or rerun GPU tests.

Next candidate is the corrected 1–4 BD / one GPU screen using this 100k input,
keeping the same CPU allowance and reporting total KvikIO workers. That sweep
has not been submitted as part of this task.

Submit the clean reviewed commit with modern Python:
`python submit.py --reviewed-sha FULL_SHA --attempt UNIQUE_SHARED_SCRATCH_PATH --submit`.
This pins sdfada016 for comparison and excludes sdfada011 (CUDA initialization
failed independently of psana). It creates new detached worktrees and refuses
an existing attempt path; inspect submission.json before considering any retry.

The native build differs only by the exact allow-listed gpu/notes packaging
exclusion in psana/meson.build. Any other configuration mismatch is rejected.
The audit repeats that check against the frozen receipt.

Round 1 passed on `7a727fd242754338be1720c457e579e6278edeb6`; Measured results are recorded above. ADA-08/09 are closed. Optional reverse-tree inventory
comparison (detecting deleted frozen sources) is deferred for this pinned runtime.
ADA-11 (`--chdir` in submit.py) is deferred; this submission is explicitly invoked
from the shared artifact root, which Slurm inherits and records as WorkDir.

Allow roughly 5 minutes for tests/startup, up to 15 minutes for copying ~313 GiB
and SHA reading ~313 GiB, about 10 minutes for pilots/four measured passes, and
5 minutes for conditioning/cleanup: ~35 minutes within the 60-minute request.
These are planning allowances, not measured timings or a guarantee.

If killed during threaded copy/hash, the executor can outlast the 120-second
termination grace period (BDS-02). After verifying the job and its processes
have stopped, inspect only its owner-controlled directory
`/sdf/data/lcls/drpsrcf/ffb/users/monarin/sim10-ada-100k-<jobid>` and remove that
specific orphan. Do not delete the accepted project fixture or another job's
stage. Normal completion removes the copy and records cleanup.json.
