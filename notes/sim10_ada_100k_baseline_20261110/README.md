# 100k-event matched baseline on Ada

Job **44189224** submitted on reviewed implementation `7a727fd24`; GPU results pending.
See scope.md for controls and acceptance checks.
The accepted source fixture is
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sample10_100k`.
Results and immutable source snapshots will be saved under
`/sdf/scratch/users/m/monarin/gpu-validation/sim10-ada-100k-baseline-20261110/`.

Submit the clean reviewed commit with modern Python:
`python submit.py --reviewed-sha FULL_SHA --attempt UNIQUE_SHARED_SCRATCH_PATH --submit`.
This pins sdfada016 for comparison and excludes sdfada011 (CUDA initialization
failed independently of psana). It creates new detached worktrees and refuses
an existing attempt path; inspect submission.json before considering any retry.

The native build differs only by the exact allow-listed gpu/notes packaging
exclusion in psana/meson.build. Any other configuration mismatch is rejected.
The audit repeats that check against the frozen receipt.

Round 1 passed on `7a727fd242754338be1720c457e579e6278edeb6`; GPU results
are still pending. ADA-08/09 are closed. Optional reverse-tree inventory
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
