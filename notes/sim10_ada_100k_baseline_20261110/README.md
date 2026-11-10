# 100k-event matched baseline on Ada

Implementation in preparation. See scope.md for controls and acceptance checks.
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
