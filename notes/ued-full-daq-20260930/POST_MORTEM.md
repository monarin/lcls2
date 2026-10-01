# UED ePixQuad gain-map programming: post-mortem and PR evidence

The production UED detector acquired 43–44 extra low-gain pixels outside its
known-bad list after full DAQ Configure. The requested map was correct in
ConfigDB and Configure metadata. Direct ASIC reads showed that the extra
pixels actually held low-gain code 8 instead of requested high-gain code 12.

The software ordering defect was in `config_expert`: it enabled ASIC IsEn,
then applied expert ConfigDB values containing IsEn=0, and entered the map
writer without restoring the programming state. Reasserting IsEn=True after
expert settings, immediately before map programming, removed all extra FPs
in run 7 while retaining the original DAQ writer, write order and camera gates.

The diagnostic branch retains all investigation scripts and the successful
standalone-port alternative. A separate branch, `fix/epixquad1kfps-isen-ordering`,
starts from upstream master `a4be0f987ba5194119fed9ae5271d2ca2a245459` and contains
only the four-line correction, commit `a4907da6f`. A separate full build passed
live validation in run 8: 1,084 complete, undamaged frames at 360.282122 Hz,
zero FP/FN outside known-bad, and all 44 prior FP addresses reading FH twice
afterward. [Clean-branch validation](closeout/pr-validation.json).
Nothing has been pushed or opened
as a PR. [PR description draft](closeout/PR_DESCRIPTION.md).

## Failure and impact

The affected production module is on `drp-ued-cmp003`, lane 0 of `/dev/datadev_0`.
The full DAQ test used experiment `uedc00107`, alias `ued/TEST`, and the real
experiment-derived expand2 mask. For this test the map requests FH (code 12)
for 518,503 pixels and FL (code 8) for 22,169 pixels, with all 16 trbits set to 1.
The extra FPs were all at ASIC row 0, across eight ASICs. These were persistent
extra FL assignments, not the separate first-frame-only standalone anomalies.

The 352-pixel known-bad baseline was established independently by fixed-FL
tests and matched earlier UED experiment studies exactly: ASIC 8, columns 97
and 98, rows 0–175. Only 68 of these pixels overlap expand2's FL selection,
explaining the 68 expected known-bad FNs in mixed-map tests. The 43/44 extra
FPs were never added to the known-bad list. Fixed FH/FM and FL comparisons
are in [the historical comparison](../ued-commissioning-20260928/HISTORICAL_COMPARISON.md).

The map originated at
`/sdf/home/m/monarin/tmp/epix-per-pixel-gainmode/roi_ft_ued1015999_r185_t300_v2/roiFromAboveThreshold_r185_c0_calib_expand2.npy`.
The tested copy is `../ued-commissioning-20260928/map-inputs/expand2.npy`,
SHA256 `1739094ffe6e80a7eae0496961730399c0f1dce077f7426e445137fac4434894`.
Its provenance and selection counts are recorded in
[gain-map-input-review.json](../ued-commissioning-20260928/gain-map-input-review.json).

## Evidence that separated the hypotheses

| Check | Observation | Interpretation |
|---|---|---|
| ConfigDB and recorded Configure | Exact expand2 map; all 44 FP locations requested code 12; trbit 1 | Stored metadata was correct |
| Runs 1/2 near 540 Hz and run 3 at 360 Hz | 43/44 extra FPs persisted | Higher rate alone did not explain the failure |
| Corrected submodule selection | Extra FPs persisted with the intended test bundle | The initial environment drift was not a sufficient explanation |
| Standalone using either writer | Clean outside known-bad after startup frame | The inner DAQ map helper could work in a different calling context |
| Standalone AsicSyncInjEn A/B/A | Both states clean outside known-bad | This setting alone did not reproduce the FPs |
| External timing standalone, attempt 17 | 490 full frames at 359.959 Hz, clean outside known-bad | DAQ timing alone did not reproduce the full-Configure failure |
| Run 4 DMA/pre-write/disk diagnostic | All 51 matched input/pre-write pixel arrays equal; pre-write datagrams identical to disk | FPs were already present before conversion/file writing |
| Direct ASIC read after run 4 | All 44 FP locations read 8 twice, despite requested 12 | Wrong gain codes existed in actual pixel memory |
| Targeted overwrite and run 5 | All 44 accepted 12; 1,084 frames clean outside known-bad | The addresses were writable; correcting memory corrected data |
| Standalone port inside Configure, run 6 | 1,084 frames clean outside known-bad without repair | Alternative established a working reference, but changed three factors |
| Original writer + IsEn-only correction, run 7 | 1,086 frames clean outside known-bad; all 44 read 12 twice afterward | IsEn ordering correction alone was sufficient for this test |
| Clean master-based PR build, run 8 | 1,084 frames clean outside known-bad; all 44 read 12 twice afterward | Minimal four-line change also passes without diagnostic-branch runtime code |

Run 3 had 4,628 complete frames with 43 extra FPs; run 4 had 2,322 with 44.
The intermittent 44th address was raw segment 1, row 175, column 170.
Run-1/2 analyses were snapshots; the archive preserves the full finalized
files as well as the analyzed-prefix lengths and checksums. Runs 3–8 were
reviewed through EndRun. Full-stream byte checks do not establish an upstream
event-loss audit.

Source evidence: [run 3](run3-gainbits/README.md), [run 4](run4-gainbits/README.md),
[ASIC readback](asic-readback-run4/README.md),
[overwrite experiment](post-configure-overwrite/summary.json),
[standalone port](standalone-daq-writer/README.md),
[IsEn-only test](isen-ordering-test/README.md),
[external timing](../ued-commissioning-20260928/external-timing-20260930/README.md).

## Software cause and why earlier testing missed it

Before the fix:

```text
IsEn=True → apply expert settings (IsEn=0) → write map → IsEn=False
```

After the minimal correction:

```text
IsEn=True → apply expert settings (IsEn=0) → IsEn=True → write map → IsEn=False
```

All sixteen expert ASIC entries contain the acquisition value IsEn=0. The
`apply_dict` traversal writes those entries. The original map helper does not
manage IsEn and relies on its caller. The caller's earlier IsEn=True therefore
does not survive until programming. Log messages and an offline replay of the
actual full Configure functions confirmed this ordering.

The standalone wrapper explicitly sets IsEn=True just before invoking either
writer, so calling the DAQ *helper* from standalone did not exercise the faulty
full Configure sequence. Existing register-model tests checked map geometry
and values but not the state established by expert configuration. Post-run
YAML snapshots also hid the difference because every path ends with IsEn=False.

The standalone port incidentally corrected IsEn as well as changing camera
gating and map ordering. The initial comparison noticed only the latter two;
the subsequent full-call-sequence audit exposed the IsEn difference. Run 7
isolated that correction, without a pre-run ASIC read or targeted repair.

This establishes the software defect and demonstrates the remedy. The ASIC's
electrical mechanism for generating extra row-0 FL assignments under IsEn=0
is still unknown. We do not claim IsEn is a blanket SACI write-enable or that
all commands fail when it is zero. Successful SACI transactions and correct
Configure metadata did not prove correct pixel-memory contents.

## Minimal PR scope and validation requirements

The PR changes only `psdaq/psdaq/configdb/epixquad1kfps_config.py`: inside
`if writePixelMap`, after expert application, set IsEn=True on the configured
ASICs using the existing retry helper. Existing acquisition cleanup remains.
No standalone selector, hardcoded mask, targeted repair, diagnostic capture,
ConfigDB migration, timing change, or bad-pixel exception belongs in that PR.

The diagnostic branch has an extra condition to preserve its opt-in standalone
path. The clean PR does not need that condition because master has no such
adapter. It uses the original map algorithms and ConfigDB map input.

Offline regression executes the actual full Configure functions with the saved
TEST configuration. It checks IsEn=1 for all 16 matrix fills and all 22,169 FL
writes, correct final map, unchanged map-command order, final IsEn=False, and
no extra programming-state transition when writePixelMap=False. The clean PR
branch passes this test. The original port's sequence is separately checked
for preservation on the diagnostic branch.

Run 7 tested the correction in the diagnostic build: 1,086 undamaged full
frames, 360.248310 Hz, zero outside-known FP/FN including frame 1, 68 known-bad
FNs per frame, and double FH readback of all 44 prior FP addresses.
Run 8 repeated validation using a separately compiled master-based PR build:
1,084 complete undamaged frames at 360.282122 Hz, including a clean first
frame, zero outside-known FP/FN, the same 68 known-bad FNs per frame, and all
44 prior FP addresses reading FH twice after EndRun. Control, TEB, timing DRP
and detector DRP all used the clean install and their logs identify `a4907da6f`.
XPM/groupca and GUI processes were retained. No standalone adapter or diagnostic
capture hook was present in that build. The readback helper ran separately
from the diagnostic environment after acquisition; an initial import attempt
from the clean environment failed before hardware access because that branch
intentionally lacks the test-only helper. Source comparisons, dependencies,
operator access and launch parsing passed before acquisition.
Broader masks, repeated power cycles, and other detector families
are not covered by this short production-module test.

## Reproduction and retained artifacts

The diagnostic worktree is
`/sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq`, branch
`features/ued-epix10ka-rdsrv421-daq`. The clean PR worktree is
`/sdf/home/m/monarin/lcls2_worktree/ued-epixquad-isen-pr`.

Scratch archive:
`/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930`.
It retains both streams of test runs, native standalone `.dat` captures,
per-attempt YAML/profiles/maps, mismatch CSVs and count arrays, diagnostic
input/pre-write captures, ASIC readbacks, saved DB configurations/backups,
software manifests, logs, and analysis/test scripts. The archive receipt and
checksum manifests are described in [closeout/README.md](closeout/README.md).
Raw files remain outside Git; concise results, checksums, scripts and this
post-mortem stay on the diagnostic branch.

The software environment is `daq_20250402_r9`, Python 3.9.21, Rogue 6.1.3.
The test bundle is `lcls2_submodules_09282026_ued_epixquad_test1`, superproject
`4a28be03a7aa77373f0795dae7a1e0d003d4bc3c`, ePix
`44a7c1cb9ed1869e75b16e24bb610e7e3ee53698`, PCIe apps
`ccbe3de52fa7f6c09dac0216372121bcfbbc49e5`. Recursive pins and applied recipe
patches are recorded in `../ued-submodules-20260928/bundle-manifest.json`.
Do not substitute an unpatched recursive checkout or update dependencies with
`--remote` when reproducing this configuration.

## Remaining handoff items

- Review the minimal PR and publish it separately when requested. Keep the
  diagnostic branch and standalone port for reference and rollback.
- Upstream the validated ePix gitlink and necessary existing patch recipe in
  `lcls2_submodules`, so other users can reproduce the tested bundle. The local
  superproject candidate is not a published production release. Record the
  merged superproject SHA and use the established dated production-bundle
  release workflow; do not pull unrelated dependency changes during promotion.
- TEST remains the gain-map test configuration. BEAM was not changed. The
  original TEST backup and explicit restoration instructions remain in
  [the operating instructions](README.md); task closeout does not silently
  restore ConfigDB or change hardware power/timing.
- An isolated test has not established the ASIC-internal IsEn mechanism or
  resolved the independent 352 known-bad pixels/first-frame standalone behavior.
