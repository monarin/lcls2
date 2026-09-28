# Profiled standalone ePix Quad tests

Use one implementation with version-controlled host profiles, rather than
maintaining separate copies of scripts. These tools use Rogue directly and do
not require the LCLS DAQ or devGui. `capture_standalone.py` programs the gain
mode, records a Rogue `.dat` file, and calculates gainbit FP/FN statistics in
one process. It does not produce XTC2.

| Profile | Current state |
| --- | --- |
| `profiles/rdsrv421.json` | Existing C1100/PGP4/XpmMini bench settings, PROM-check bypass, 10 Hz camera auto-trigger. The new capture lifecycle needs hardware validation. |
| `profiles/drp-ued-cmp003.json` | Prepared for the first commissioning capture: KCU1500/PGP4 lane 0, local XpmMini, 10 Hz camera auto-trigger, ConfigDB-derived baseline. PROM startup and actual image sizes still need hardware validation. |

The production host uses a different card/environment from the bench. Its
settings and internal-trigger choice are recorded in
[the UED preparation report](../../../../notes/ued-settings-20260928/README.md).
`ready_for_test` permits the first commissioning capture; it does not establish
hardware acceptance. Payload sizes are strict acceptance criteria from the
existing decoder, still to be checked against UED data. PROM validation remains
enabled: failure stops startup and must not be bypassed automatically.
There is no CLI override for profile status or the host guard.

Timing can be `xpmmini` or `external`. The current shared `external` startup uses
LCLS-II/186 MHz initialization; it does not implement UED's full 119 MHz dual
trigger setup. The UED test intentionally uses XpmMini. A full DAQ reinitialize
and Configure is needed to reapply the production path afterward; Enable alone
is insufficient. Preserve/read back the saved settings as described in the
preparation report. Neither standalone timing choice starts the full DAQ.

## Prepare and preview

From the lcls2 checkout on SDF:

```bash
source scripts/setup_ued_epixquad_test.sh
epix_tools=psdaq/psdaq/debugtools/epixquad1kfps
epix_profile="$epix_tools/profiles/drp-ued-cmp003.json"
python "$epix_tools/capture_standalone.py" \
  --profile "$epix_profile" --mode MapFML \
  --no-default-pixels --raw-pixel 0,76,115 \
  --frames 50 --label sparse-map --repetition 1 \
  --outdir /tmp/ued-profile-preview-001 --dry-run
```

Choose a new output directory for every attempt. Dry runs save the proposed
settings and expected map without importing hardware modules or accessing a
device. They also permit incomplete profiles; successful dry runs do
not establish hardware readiness. The environment helper currently selects a
private SDF bundle. On a test host, source its OS-appropriate DAQ environment,
then set `SUBMODULEDIR` to the accessible copy of the pinned bundle and put
this checkout's `psdaq` directory first in `PYTHONPATH`. Do not assume SDF paths
are mounted there. See [bundle preparation](UED_SUBMODULE_BUNDLE_PREPARED.md).

For an approved hardware session with a validated profile, use the same command
without `--dry-run` and select a durable results location. Use `FH`, `FM`, and
`FL` for fixed-gain baselines; map modes are `MapFHL` and `MapFML`, with explicit
`--raw-pixel` entries or `--selected-map` input. Repeat with a new directory and
incremented `--repetition`. Record test conditions in `--label` and `--notes`.
The implementation retains the existing matrix-fill plus selected-pixel write
path; it does not implement a full detector individual-pixel write comparison.

## Execution and accounting

Hardware execution requires exclusive use of the PCIe card and detector; stop
conflicting DAQ/GUI users through the agreed operational procedure first.
DevRoot startup configures timing and can affect every application lane, even
though the profile selects one camera lane. Before DevRoot starts, the scripts
check the allowed host and read PCIe identity to verify board, image, and minimum
firmware. These identity checks do not prove lane or timing compatibility.
Camera startup retains the profile's PROM-validation choice. The recorder
requires ADC done/pass and never invokes ADC training or PROM programming.

The recorder loads the copied camera baseline YAML, disables triggering, writes
the gain map, opens image VC0 plus register VC1, and starts capture. It counts
only accepted complete, error-free image payloads toward `--frames`. Short and
error records remain in the file for diagnosis. `--timeout` and `--max-bytes`
bound acquisition; startup/register accesses retain the underlying libraries'
timeouts. Capture can include extra frames while it stops and drains. Cleanup
attempts to disable camera triggers/readout, stop the PCIe run, and close the
writer on success, failure, Ctrl-C, or SIGTERM. SIGKILL, host loss, and device
failure cannot guarantee cleanup; inspect state before another attempt.

Each attempt contains:

- `test.json`: label/repetition/notes, command, timestamps, OS/kernel, mode,
  selected pixels, acquisition status/counts, available hardware identities,
  runtime versions/import paths, data checksum, analysis, and errors.
- `profile.json`, `source-profile.json`, `resolved-profile.json`: original JSON,
  resolved source settings, and effective settings referring to copied inputs.
- Copies and hashes of baseline YAML and selected-map inputs, plus
  `expected-map.npy`.
- `software.json`: lcls2 commit and tracked diff, script hashes, superproject
  commit, recursive actual/recorded submodule commits, tracked diffs, untracked
  file lists, and `patch.sh` checksum. Untracked dependency files are listed,
  not archived; use committed dependencies for reproducible tests.
- `console.log`; for captures, `capture.dat`, `first_raw.npy`, and
  `mismatches.npz` with per-pixel FP/FN occurrence counts when decoding succeeds.

Existing output directories are rejected. Once an attempt directory is created,
failures retain a failed/interrupted record and any partial data. Invalid CLI
arguments or an unreadable profile fail before creating an attempt. Records are
not made read-only; preserve them alongside the raw data for review. A
`completed` status means acquisition/decoding completed, **not** that gainbit
mismatches passed a scientific acceptance threshold. Active/control rows are
reported separately using the existing decoder masks. Partial or malformed
trailing records are rejected by analysis.

## Other entry points

The gain-only writer and optional GUI now require the same profile for hardware:

```bash
python "$epix_tools/write_gain_mode_standalone.py" \
  --profile "$epix_profile" --mode FH --dry-run
python "$epix_tools/launch_xpmmini_writer_gui.py" \
  --profile "$epix_profile" --dry-run
```

Gain-only legacy dry runs without a profile still work. Explicit CLI overrides
for device, lane, data VC, PROM bypass and YAML are recorded by the headless
recorder; defaults come from the selected profile. YAML paths starting with `./` or
`../` resolve relative to the profile JSON; other relative paths resolve under
`SUBMODULEDIR`. Absolute paths are also accepted. The recorder always loads
its camera baseline and currently requires all 16 ASICs, without charge
injection or bitmap readback. Use it for automatic test accounting: the GUI and
gain-only writer do not create attempt directories. The older
`diagnose_standalone_access.py` is still a bench diagnostic and has not been
adapted for production UED.

## Offline verification

In the DAQ Python environment, from this checkout:

```bash
PYTHONPATH="$PWD/psdaq${PYTHONPATH:+:$PYTHONPATH}" \
  QT_QPA_PLATFORM=offscreen python -m unittest discover \
  -s psdaq/psdaq/tests -p test_epixquad_standalone.py -v
```

Tests exercise profile isolation, host/card gates, partial-startup cleanup,
all five gain maps, image counting, timeout cleanup, preservation of failed
attempts, overwrite prevention, truncated files, and the real Rogue stream
writer/reader with synthetic images. Native Rogue checks skip if Rogue is not
installed. No detector or PCIe roots are opened.

After successful hardware tests, publish the validated superproject pins and
required patch recipe through the agreed lcls2_submodules review, then promote
the exact tested bundle under the production date-based name. Script commits
and software-only tests do not constitute that promotion.
