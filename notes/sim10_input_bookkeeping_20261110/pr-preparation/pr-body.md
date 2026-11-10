## Motivation

Submitting 100 events across five XTC files produces 500 input groups. Previously
`InputGroupPool.issue()` polled retained groups and scanned slot occupancy for
every read. With two CUDA streams, this repeatedly revisited the previous batch's
live windows and delayed submission of the next batch.

Submit each admitted group plan together, track occupied slots in a set, and poll
only closed windows whose planned/live references have been released. Existing
CUDA completion and raw ownership checks still control reuse. Partial-plan failures remain pool-owned and are drained by `pool.close()`.
Physical read sizes, read counts and callback sizes are preserved. Tests cover partial submission,
delayed completion, consumer-thread notifications, stale notifications and retries.

On the original research implementation, a matched Perlmutter 10k-event comparison
improved two-stream throughput from **1.27 to 2.18 GiB/s**; one stream changed from
2.23 to 2.29 GiB/s. Pool polls fell from 50,101 to 101. This is descriptor staging,
local-page-cache-cold Lustre with KvikIO fallback; backend cache state is unknown.
See [measurement details and immutable evidence](https://github.com/monarin/lcls2/blob/6f61a8d760c269b61f0f3a93c4f424d10f5932a2/psana/psana/gpu/docs/performance/input_group_bookkeeping.md).

This focused branch starts from `features/psana2-gpu` at `e032e0d56`. It extracts
five code/test files from the reviewed research implementation and retains the
target's later IPC, memory-accounting and field-access fixes.

## How to test the behavior?

Build/install this checkout and activate its matching psana runtime, then run:

```bash
PS_PARALLEL=mpi python -m pytest -q psana/psana/tests/gpu/unit
```

- **Current target plus this patch:** 1,023 CPU unit tests passed in 10.19 s,
  loading checkout Python over the existing SDF Python 3.9 native extensions.
- **Original research implementation:** 970 CPU tests, 31 native GPU tests
  (zero skips), and all 12 matched clean/timing cases passed. Each measured case
  retained 50,000 reads / 33,582,000,000 bytes / 100 callbacks of 100 events.
- The exact PR-base runtime has not been rerun on GPU or rebenchmarked. Historical
  native results are linked as supporting evidence, not claimed as new-base runs.
- Independent Claude review of this extracted port: **pass**, no blocking
  findings, at `6f61a8d760c269b61f0f3a93c4f424d10f5932a2`.

## Checklist

- [x] Did you update `CHANGELOG.md` with your changes?
- [x] Does the PR clearly describe the problem and the solution?
- [ ] Do all GitHub Actions checks pass on this PR? (They are not yet required for merging, but please make sure they pass.)
- [ ] Does the PR use "Fix #XXX" notation to tell GitHub to close the relevant issue numbered XXX when the PR is merged?

The target branch has no root changelog, so this adds a minimal Unreleased/Fixed
entry following master's format.
No GitHub Actions runs have been reported for this head. The target branch's
psana/DAQ test workflows currently use scheduled/manual triggers, not PR triggers. No issue closure is asserted for this focused
bookkeeping change.
