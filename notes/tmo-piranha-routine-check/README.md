# Piranha routine check

Development branch: `features/piranha-routine-check` in `monarin/lcls2`.
Synchronized from `origin/master` at `25e6f4d6535b7b5b24e9cf0470ada5e9273b2ed0`
on November 10, 2026. Dedicated worktree:
`/sdf/home/m/monarin/lcls2_worktree/piranha-routine-check`.

## Current status

Draft PR: [monarin/lcls2#2](https://github.com/monarin/lcls2/pull/2).

The standalone allocation pilot and offline tests are implemented. It supports
single-camera and paired cold/warm trials. All 31 recorded camera connection
attempts on November 6 and 9 passed; the intermittent allocation GCP timeout
(issue 205) has not been reproduced. This is limited smoke-test evidence, not
proof of full-DAQ operation or absence of the fault.

The hold-on-error implementation passes 59 offline tests and independent
PR-readiness re-review on `fd2ab73d3332597d9ce51b1e1f97efff776ba34e` (two rounds,
expiry-race finding repaired). See the
[PR-readiness record](../tmo-piranha-pr-review-20261111/status.json) and the earlier
[implementation/review record](../tmo-piranha-hold-implementation-20261111/status.json).
The earlier optional labeling follow-up HOLD-IMPL-007 remains deferred.
The new hold path has not been tested on hardware. A new hardware test requires
a current exclusive-board window; historical confirmations are not ongoing
reservations. No hardware is accessed during local implementation/review.

## Code, design and review

- [Pilot and execution instructions](../../psdaq/psdaq/debugtools/piranha4_pilot/README.md)
- [Pilot source](../../psdaq/psdaq/debugtools/piranha4_pilot/pilot.py)
- [Offline lifecycle tests](../../psdaq/psdaq/debugtools/piranha4_pilot/test_pilot.py)
- [Original task transfer](../tmo-piranha-standalone-test-task-transfer-2026-11-06.md)
- [Standalone-path findings and initial test plan](../tmo-piranha-standalone-test-findings-2026-11-06.md)
- [Original source evidence](../tmo-piranha-review-20261106/evidence/manifest.json)
- [Offline path verification](../tmo-piranha-review-20261106/verification/result.md)
- [Initial plan review](../tmo-piranha-review-20261106/status.json)
- [Pilot implementation review](../tmo-piranha-review-20261106/pilot/status.json)
- [Staged hold-on-error implementation plan](../tmo-piranha-hold-plan-20261110.md)
- [Hold-plan review](../tmo-piranha-hold-review-20261110/round-01/review.json)

Historical preparation documents retain the status at the time they were written
(e.g. “offline only”). This overview and the later hardware records supersede
those time-specific status statements. Historical SHAs and absolute run paths
are retained as provenance; use [commit-map.json](commit-map.json) to locate the
transplanted commits and the table below to locate archived run data in this branch.
The historical hardware-tested pilot had SHA256
`d7c7e249753f29dccf5d713365b79758f8fb60da42fa4e336b5728d2ea10b804`.
New hold-enabled runs record hashes of all local runtime modules; do not
attribute the historical hardware results to the new implementation.

## Recorded hardware tests

FZP is lane 0 / serial 12097685; ATM IP1 is lane 1 / serial 12102856.
Both use `/dev/datadev_0` on `drp-srcf-cmp013`. November 6 tests were
operator-executed; November 9 tests were executed by Codex after Mona confirmed
the exclusive window. Camera/mode labels below come from the manifests rather
than directory names (some ATM directories were named `fzp`).

| Date | Cameras | Mode | Trials | Camera calls | Evidence |
|---|---|---|---:|---:|---|
| Nov 6 | fzp | cold | 1 | 1 | [summary](../tmo-piranha-review-20261106/hardware/piranha-pilot-fzp-20261106-165212/summary.json) |
| Nov 6 | fzp | cold | 3 | 3 | [summary](../tmo-piranha-review-20261106/hardware/piranha-pilot-fzp-20261106-170024/summary.json) |
| Nov 6 | fzp | warm | 3 | 3 | [summary](../tmo-piranha-review-20261106/hardware/piranha-pilot-fzp-20261106-170600/summary.json) |
| Nov 6 | atm | warm | 3 | 3 | [summary](../tmo-piranha-review-20261106/hardware/piranha-pilot-fzp-20261106-171152/summary.json) |
| Nov 6 | atm | cold | 3 | 3 | [summary](../tmo-piranha-review-20261106/hardware/piranha-pilot-fzp-20261106-171317/summary.json) |
| Nov 6 | fzp,atm | cold | 3 | 6 | [summary](../tmo-piranha-review-20261106/hardware/piranha-pilot-pair-cold-20261106-172032/summary.json) |
| Nov 9 | fzp,atm | cold | 3 | 6 | [summary](../tmo-piranha-review-20261109/paired-cold-3/summary.json) |
| Nov 9 | fzp,atm | warm | 3 | 6 | [summary](../tmo-piranha-review-20261109/paired-warm-3/summary.json) |

- [November 6 single-camera analysis](../tmo-piranha-review-20261106/pilot/operator-campaigns-20261106.json)
- [November 6 paired-cold analysis](../tmo-piranha-review-20261106/pilot/operator-paired-cold-20261106.json)
- [November 9 report](../tmo-piranha-review-20261109/results.md)
- [November 9 detailed verification](../tmo-piranha-review-20261109/verified-results.json)
- [November 9 live cleanup check](../tmo-piranha-review-20261109/post-run-check.json)

Each run includes manifests, provenance, raw UART captures, results and cleanup
records. Original `process.log` files are stored as `process.log.gz` using lossless
gzip so repository whitespace hooks cannot alter the captured output. Decode with
`gzip -dc path/to/process.log.gz`. JSON and the remaining evidence stay directly
browsable. [archive-manifest.json](archive-manifest.json) maps original paths to
repository paths and records original/stored SHA256 hashes. Nothing was removed
from the original result directories.

## Next work

Local implementation and independent review are complete. Obtain a new
exclusive-board window for hardware rehearsal and a bounded
cold/warm campaign. The existing-server GUI is write-capable, not an enforced
spy interface. Do not open another Rogue hardware root for inspection. See the
[implementation scope](../tmo-piranha-hold-implementation-20261111/scope.md) for
acceptance checks and remaining hardware boundaries.

## Consolidation checks

See [validation](validation-20261110.md) and the [17-test output](offline-tests-20261110.txt).
