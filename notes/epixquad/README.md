# ePixQuad repeat tests and debugging

This is the maintained entry point for branch
`features/ued-epix10ka-rdsrv421-daq`. Use the shared implementations under
[`psdaq/psdaq/debugtools/epixquad1kfps`](../../psdaq/psdaq/debugtools/epixquad1kfps/README.md).
The dated notes preserve the September 2026 investigation; their process IDs
and running-state statements are historical, not current operating instructions.

The user stopped DAQ and powered off the detector after run 8. This cleanup
performs no hardware access, ConfigDB restoration, timing changes, deployment,
or build replacement. TEST restoration remains pending unless separately recorded.

## Choose the module and timing

| Profile | Hardware / timing | Validation scope |
|---|---|---|
| `rdsrv421.json` | Bench C1100, PGP4 lane 0, XpmMini/internal camera trigger | Earlier bench tests established mapping; the shared headless lifecycle still needs validation on this host. Its PROM-check bypass is bench-specific. |
| `drp-ued-cmp003.json` | Production KCU1500, PGP4 lane 0, XpmMini/internal camera trigger | FH/FM/FL, both map helpers, and readout-clock/rate comparisons tested in September 2026. PROM validation enabled. |
| `drp-ued-cmp003-external.json` | Production KCU1500, UED 119 MHz dual-trigger path, XPM0/group 0 | Attempt 17: 490 full frames near 360 Hz, no FP/FN outside the production known-bad list after the first frame. Operator controls groupca RUN. |

The JSON profiles are in the shared tools' `profiles/` directory. Do not copy
production known-bad masks, carrier identity, clock settings, or restoration
snapshots to another module. Historical profile values must be checked against
the current card, firmware, detector and approved configuration before use.

## Repeat a standalone test

From this checkout on SDF, prepare an offline preview:

```bash
source scripts/setup_ued_epixquad_test.sh
epix_tools=psdaq/psdaq/debugtools/epixquad1kfps
python "$epix_tools/capture_standalone.py" \
  --profile "$epix_tools/profiles/drp-ued-cmp003.json" \
  --mode FH --frames 50 --save-config-yaml --label fh-preview \
  --outdir /tmp/epixquad-fh-preview-001 --dry-run
```

Use a new output directory each time. Select `rdsrv421.json` for bench preview.
On the bench, first source its OS-appropriate environment and select an
accessible copy of the pinned bundle; SDF paths need not be mounted there.
See [profile/environment instructions](../../psdaq/psdaq/debugtools/epixquad1kfps/STANDALONE_PROFILES.md).

For an authorized hardware session, establish exclusive card/VC ownership and
confirm power, identity and ADC status, then omit `--dry-run` and use a durable
output directory outside the source tree. The direct capture helper quiesces
the camera on exit; it does not restore every prior register. Save the initial
state and arrange the appropriate restoration or fresh DAQ initialization.

1. Capture FH, FM and FL baselines before mixed gain. Keep first-frame statistics
   separate from persistent mismatches; gain bits cannot distinguish FH from FM.
2. Capture MapFHL/MapFML with explicit `--selected-map /path/to/mask.npy
   --no-default-pixels`. The retained experiment-derived expand2 input is
   [`../ued-commissioning-20260928/map-inputs/expand2.npy`](../ued-commissioning-20260928/map-inputs/expand2.npy).
3. Compare `--gain-writer standalone` and `--gain-writer daq` using identical
   inputs. The latter exercises the inner map helper, **not full DAQ Configure**.
4. For the production module, the tested full DAQ clock override is
   `--asic-ro-clk-ticks 3` (full period 3, half-period readback 1). Confirm a full
   frame at 10 Hz before 360 Hz. Do not interpret this as setting
   `AsicRoClkHalfT=3`. Keep the same clock for comparisons.
5. To isolate sync injection, use `--asic-sync-inj-en 0` / `1`. Save YAML with
   `--save-config-yaml`; these snapshots do not read ASIC pixel SRAM.
6. For UED groupca timing, select the external profile, omit internal rate
   overrides, and wait for `READY FOR GROUPCA RUN` before RUN. Stop groupca
   afterward. See [the external procedure](../ued-full-daq-20260930/external-timing-preparation/README.md).

The historical `run_once.py` wrapper and `restore_camera_idle.py` are retained
for reproducing that exact UED commissioning session. They are restricted to
`drp-ued-cmp003`, use its saved preflight identity/state, and write beside the
dated notes. They are not the generic entry points for a new module/session.
Use `capture_standalone.py` with fresh output paths for new work.

## Tools retained for diagnosis

| Need | Entry point |
|---|---|
| Connectivity / current registers | Shared `probe_connection.py`; UED-specific `../ued-settings-20260928/read_settings.py` |
| Rogue recording decode | Shared `read_xpmmini_rogue_file.py` |
| Per-pixel mismatch export for either module | Shared `export_mismatches.py`; retain detector identity before deriving a known-bad mask |
| Production-module standalone comparison | `../ued-commissioning-20260928/review_gain_map.py` and `review_payloads.py` |
| Production expand2 XTC gain bits | `../ued-full-daq-20260930/review_xtc_gainbits.py` |
| ASIC memory at the 44 historical FP addresses | `../ued-full-daq-20260930/read-fp-pixels.py` (plan by default; `--execute` accesses hardware; `--overwrite-fh` additionally writes pixels) |
| DMA → pre-write → disk capture | `../ued-full-daq-20260930/diagnostic-hook/` and `psdaq/drp/EpixQuadDiagnostic*` |
| Original writer / standalone-port comparison in DAQ | `../ued-full-daq-20260930/isen-ordering-test/` and `standalone-daq-writer/` |
| ConfigDB schema / TEST backup and restore | [Rollback notes](../ued-config-rollback-20260930/README.md), [DAQ instructions](../ued-full-daq-20260930/README.md) |
| Bundle pins / applied patches | [Bundle preparation](../../psdaq/psdaq/debugtools/epixquad1kfps/UED_SUBMODULE_BUNDLE_PREPARED.md) |

Production-specific reviewers deliberately validate the recorded carrier identity,
expand2 map, 352-pixel known-bad baseline and trbits. Do not bypass those guards
to analyze a different detector. The generic decoder/mismatch exporter is the
starting point for new bench-module baselines.

ASIC readback changes programming/address state and gates even without pixel
writes. Only run it during an authorized idle session, verify identity, and
restore/check controls afterward. For a clean reproduction, acquire before
any readback or targeted repair. Never read an unpowered detector.

The diagnostic hook is opt-in and bounded. Its analysis compares matched DMA
payloads, pre-write datagrams and finalized disk files. It does not compete for
the DAQ-owned data VC. Keep it disabled for ordinary production use.

## Data and findings

The [post-mortem](../ued-full-daq-20260930/POST_MORTEM.md) explains why full
Configure failed while both standalone map helpers passed: expert settings
reset IsEn to 0 before map programming. Reasserting True immediately before
the write fixed the tested map; the electrical mechanism is not established.
The minimal fix is commit `a4907da6f`, published as
[PR #152](https://github.com/slac-lcls/lcls2/pull/152). Check current PR/release
state before assuming it is deployed.

| Run | Mode | Result outside known-bad |
|---|---|---|
| 1–2 | Unfixed DAQ, about 540 Hz | 43/44 extra FPs in the analyzed snapshots; finalized files are retained |
| 3 | Unfixed DAQ, 360 Hz | 43 extra FPs; preferred failing comparison |
| 4 | Unfixed DAQ + diagnostic capture | 44 extra FPs; already present before disk writing |
| 5 | Targeted repair of 44 addresses | Zero FP/FN |
| 6 | Standalone writer port in full Configure | Zero FP/FN |
| 7 | Original writer + IsEn correction | Zero FP/FN |
| 8 | Clean master-based PR build | Zero FP/FN in all 1,084 complete frames, including the first |

The production known-bad list has 352 pixels (ASIC 8, columns 97/98). Its
overlap with expand2 produces 68 known-bad FNs per mixed-map frame. The extra
43/44 FPs were never folded into that baseline.

All eight runs and both streams, including the matching smalldata, are shared:

```python
from psana import DataSource
ds = DataSource(exp="uedc00107", run=8,
    dir="/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/xtc")
```

Use run 3 for the unfixed comparison. Directory traversal and data reads are
enabled for other SDF users; the configuration backups and other evidence
remain private. On October 1, all 16 smalldata files were checksum-verified
and an event from every run was read using `DataSource(dir=...)`.

## Evidence retention and software checks

Generated logs, repeated YAML/register snapshots, per-frame tables and one-off
historical comparison scripts were removed from the working tree only after
matching their SHA256 against the scratch archive. [archived-files.csv](archived-files.csv)
maps every removed file to its preserved copy and checksum; [cleanup.json](cleanup.json)
records the pre-cleanup commit and full source snapshot. Original Git history
is also retained. No raw data or installed builds were deleted.

Historical documents link to archived outputs. They remain useful evidence,
but old absolute launch paths, PIDs, configuration keys and prerequisites must
be checked again before a new experiment. The saved TEST and BEAM configurations
are backups, not permission to overwrite current ConfigDB entries.

Run the [offline checks](OFFLINE_CHECKS.md) after changing reusable code. New
capture output belongs in a separate dated scratch directory. The validated
submodule gitlinks/patch recipe still need the normal upstream publication and
production promotion workflow; this cleanup does not promote them.
