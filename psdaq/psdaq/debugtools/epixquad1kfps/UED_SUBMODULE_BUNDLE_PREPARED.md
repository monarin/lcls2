# Prepared UED test submodules

Prepared 2026-09-28. Software checks passed; detector hardware tests remain.

## Candidate identity

- Directory: `/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1`.
- Superproject branch: `ued-epixquad-20260928`.
- Local commit: `4a28be03a7aa77373f0795dae7a1e0d003d4bc3c`.
- Parent: `f42780c93414182c652c620eaeb861bcf1b99333`.
- The commit changes only the `epix-quad-1kfps` gitlink, from
  `239cbd54083503efe7f41f452032534d9c989977` to
  `44a7c1cb9ed1869e75b16e24bb610e7e3ee53698` (PRs 19, 20, 21).
- PCIe apps remains at `ccbe3de52fa7f6c09dac0216372121bcfbbc49e5`.
- All 14 top-level submodules were initialized, totaling 106 repositories
  including the superproject and nested dependencies. All checkout SHAs match
  their recorded gitlinks.

The clone used Git partial-clone filtering. Firmware LFS blobs were not fetched;
LFS filters were disabled for the initialization command only. This is a Python
software test bundle, not a populated firmware-build checkout. No shared release
directory was changed and the superproject branch has not been pushed.

## Compatibility patches and environment

The pristine recursive checkout had no source modifications. The existing
superproject `patch.sh` was then applied unchanged: seven repositories have
source patches, affecting ten Python files. All ten compile with DAQ Python 3.9.
The applied diffs were inspected and saved in the manifest. These expected child
repository modifications are reproduced by the tracked patch recipe; they are
not extra child commits.

`patch.sh` SHA-256:
`457b352fa23530aa4b32f9009370ceab134410e9ae5e1bd87fb728f9c13f652f`.

Environment: `daq_20250402_r9`, Python 3.9.21, Rogue/PyRogue 6.1.3.
The PCIe apps Rogue 6.4 guard is disabled by the pre-existing site recipe;
successful imports do not establish hardware compatibility.

## Select the candidate

From this lcls2 worktree, in Bash:

```bash
source scripts/setup_ued_epixquad_test.sh
```

The helper sources the normal checkout environment, then sets `SUBMODULEDIR`
to the candidate and prepends this checkout's psdaq source to `PYTHONPATH`.
It only configures the shell. Its selection of the Python executable,
`SUBMODULEDIR`, and psdaq source path was checked.

The standalone hardware launchers still need their production-card/timing
adaptation. Sourcing this environment does not perform that adaptation or start
a test. Headless import/offline checks use `QT_QPA_PLATFORM=offscreen`; use a
working display when actually launching a GUI.

## Completed software validation

- Eight ADC-training mock scenarios passed: successful/failed training,
  per-ADC retries, PROM-write guard, diagnostic file, and cleanup.
- ADC tester pass/fail/timeout and both callers passed with mocked registers
  and a simulated monotonic clock.
- The actual SystemRegs model passed using Rogue 6.1.3 and emulated memory:
  bulk writes skip AsicMask; keyed WO writes and readback behave as expected.
  The previously documented caveat remains: a direct `AsicMask.set()` can still
  issue a write despite the RO declaration. RO is not a hardware write barrier.
- FH, FM, FL, MapFHL, and MapFML dry runs passed, including saved expected maps.
- The actual Rogue FileReader and decoder processed two synthetic full-image
  records with 40- and 80-byte image offsets, excluded a short record, and
  reported zero FP/FN pixels. This checks software, not real module payloads.
- Imports passed locally and on `drp-ued-cmp003` as Mona via the psbuild hop.
  All seven detector/PCIe/timing packages resolved into the private candidate.
  With the standalone helpers' import order, SURF and axi-pcie-core resolve
  from the ePix nested dependencies; the manifest records their exact pins.
- No detector/PCIe roots were instantiated on the production node, no device
  was opened, and no timing, ADC, PROM, or DAQ settings were changed.

The first headless import attempt hit the inherited unavailable X display; the
offscreen setting resolved it. The probe's timing-package name was corrected to
the repository's actual `LclsTimingCore`. Neither required a bundle source change.

## Evidence

Under [notes/ued-submodules-20260928](../../../../notes/ued-submodules-20260928/):

- `bundle-manifest.json`: recursive commits, source patches, recipe checksum.
- `before-patches.json`: pristine checkout evidence.
- `software-environment.json`: interpreter, Rogue version, actual import paths.
- `drp-ued-cmp003-imports.log`: production-host import-only check.
- `epix-fix-checks.log`, `standalone-checks.log`, `synthetic-decode.log`, and
  `dry-run-*.log`: offline test results.
- `check_epix_fixes.py`, `check_standalone.py`, `record_bundle.py`: rerunnable
  validation and manifest scripts. Use the DAQ Python interpreter.

## Remember after successful hardware tests

Mona requested that the tested versions be reproducible by others. After tests
pass, prepare an upstream `slac-lcls/lcls2_submodules` PR with the validated ePix
gitlink, necessary patch-recipe changes if any, and hardware-test results. Retain
the tested dependency pins rather than advancing them during promotion.

After upstream integration, publish the same versions through the release
maintainer workflow as a new `lcls2_submodules_MMDDYYYY` directory. Record the
merged superproject SHA, switch UED's `SUBMODULEDIR` when authorized, verify
operation, and keep the previous bundle for rollback. Other detectors require
their own adoption checks.

This follow-up was also saved in the shared ePix troubleshooting skill and synced
to Mona's skills repository, commit `1635b60`.
