# UED gain-map task closeout

The investigation and accounting branch is `features/ued-epix10ka-rdsrv421-daq`.
Its [post-mortem](../POST_MORTEM.md) and [PR-description draft](PR_DESCRIPTION.md)
cover the software ordering defect, evidence, correction, and remaining limits.
The standalone port and diagnostic/test scripts remain on that branch.

## Minimal PR branch

- Worktree: `/sdf/home/m/monarin/lcls2_worktree/ued-epixquad-isen-pr`.
- Branch: `fix/epixquad1kfps-isen-ordering`.
- Base: upstream master `a4be0f987ba5194119fed9ae5271d2ca2a245459`.
- Fix commit: `a4907da6f`; one file, four added lines, [patch](PR_FIX.patch).
- Diff scope: reassert ASIC IsEn=True after expert configuration, before map
  programming. No diagnostic code, hardcoded map or targeted overwrite.
- [Offline actual-source replay](test_pr_source.py) passes; [record](pr-offline-tests.json).
- [Build recipe](build-pr.sh) builds from that worktree into its own
  `install-ued-isen-pr` prefix, with the validated UED submodule bundle.
- [Build record](pr-build-record.json) and [operator checks](verify-pr-operator.log)
  passed; all 379 installed psdaq Python source comparisons matched.
- [Run-8 validation](pr-validation.json): 1,084 complete, undamaged frames at
  360.282122 Hz; zero outside-known FP/FN including the first frame; 68 known-bad
  FNs per frame. All 44 prior FP addresses read code 12 twice after EndRun.
  [Per-frame analysis](../run8-gainbits/README.md).

The PR is local only. Nothing has been pushed or submitted for review.
Do not cherry-pick the diagnostic branch's combined code/evidence commits into
master; use the clean branch. Further edits to its fix need validation recorded
against the revised commit.

## Scratch evidence

Archive root:
`/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930`.

The archive is a copy, not symlinks into FFB or the operator's home:

- `xtc/`: both timing and detector streams from the test runs.
- `operator-evidence/`: native diagnostic DMA/pre-write captures and saved
  ConfigDB configurations/backups.
- `notes/ued-*/`: standalone native data, YAML/config/profile snapshots, masks,
  all-frame and assessed mismatch reports, known-bad baseline, ASIC readbacks,
  execution logs, source/software manifests, test and analysis scripts.
- `daq-logs/2026/09/`: relevant September 30 detector, control, TEB and timing logs.
- `source/`: complete tracked-source snapshots for the diagnostic and PR branches.
- `archive-receipt.json`: completed phase counts, manifest hashes, source commits
  and source-snapshot hashes; a copy is retained on this branch.

[archive_evidence.py](archive_evidence.py) inventories without `--execute`.
Executing it copies each source, hashes it, independently re-reads/hashes the
destination, checks source size/mtime stability, and verifies recorded XTC
analysis checksums. Existing unequal destinations are not overwritten. Each
phase produces an immutable JSON manifest and SHA256SUMS file in scratch.
The initial data phase verified 126 files / 20,859,507,976 bytes; see
[the data receipt](data-archive-verified.json).

Early run-1/2 analyses covered snapshots, not their eventual complete files.
The archive retains each full finalized file and separately verifies the
original analyzed prefix. This distinction is preserved in the manifest;
those snapshot results must not be relabeled full-run analyses.

For independent archive integrity checks, run from the archive root:

```bash
sha256sum -c data-SHA256SUMS
sha256sum -c context-SHA256SUMS
sha256sum -c pr-SHA256SUMS
sha256sum -c source-SHA256SUMS
```

The SHA256SUMS files list only their corresponding completed phase. Raw files
are excluded from Git and preserved in scratch; the branch carries reports,
scripts, manifests and provenance. Full source snapshots retain the scripts
outside `notes`, including the standalone writer and packet decoder.

## Selection and final state

For the clean PR build, source [setup-pr.sh](setup-pr.sh). This selects the
tested install and bundle; it does not itself start DAQ or change hardware.
The launch configuration is [epix_pr_test.py](epix_pr_test.py), based on the
reviewed UED routes, alias TEST, with no diagnostic capture hook.

The standalone-port setup remains
`../standalone-daq-writer/setup.sh`; its code and launch recipe are preserved.
Changing an already-running process requires normal Unconfigure/disconnect
and restart, not just sourcing another setup script.

The DAQ was left **Configured**, alias TEST, recording enabled, last run 8.
Control, TEB, timing DRP and detector DRP use the clean PR install. Their
[runtime log headers](pr-runtime-evidence.log) identify commit `a4907da6f` and
the correct bundle. XPM/groupca and GUI processes were retained. Post-run
readback restored all gates without errors. The readback client uses the
diagnostic environment; its initial invocation under the clean PR environment
failed before hardware access because test-only helpers are absent there.

Closing documentation does not restore
TEST, switch production releases, alter XPM timing, or power off the detector.
The original TEST backup and restoration instructions remain in the parent
operating README. Upstream submodule pin/recipe publication and production
promotion are still follow-up work, as requested earlier.
