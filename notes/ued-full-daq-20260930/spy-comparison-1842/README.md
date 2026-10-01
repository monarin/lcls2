# DAQ spy YAML versus a fresh standalone reference

September 30, 2026. DAQ snapshots came from `drp-ued-cmp003`:
`/tmp/config_20260930_184051.yml` (KCU1500) and
`/tmp/config_epixquad_20260930_184227.yml` (camera). The operator saved these
while DAQ was running without XTC recording, then stopped DAQ and authorized
a bounded standalone capture. Preflight found no DRP, devGui, or capture
process; the monitoring IOC and CA repeater remained running.

**The fresh standalone capture again has zero FP/FN outside the known-bad
list after frame 1. The camera's `AsicSyncInjEn` differs from the DAQ snapshot:
False standalone, True DAQ. This is a candidate explanation, not a proven cause.**

Follow-up: the [standalone False/True/False test](../sync-aba-20260930/README.md)
completed later that evening and remained clean in all three conditions.
Enabling this field alone did not reproduce the DAQ FP pattern.

## Capture and assessment

Attempt 13 ran 18:55:46–18:57:03 PDT with the intended September test bundle,
the actual DAQ raw-map writer, the same `expand2.npy` MapFHL map, full readout
clock period 3, and internal trigger period 434028 ticks (359.99981568 Hz).
The wrapper requested 51 frames; 87 complete images were recorded including
pipeline/drain frames. All 86 assessed images after the first have zero FP
and exactly 68 FN, all within the existing 352-pixel known-bad list.
The first image is retained: 124 FP and 22044 FN, excluded by the agreed policy.

Image sequences 3669389–3669475 are consecutive. Native records contain 72
images of 1095248 bytes, 15 images of 1095288 bytes, and 39 timing-only records
of 48 bytes; all Rogue error bytes are zero. There are no capture cleanup
errors or line-buffer errors, and ADC status is done/pass. Host arrival rate
over this short capture is 230.8 Hz, so this is evidence of gain correctness
at the programmed internal period, not a sustained 360-Hz throughput test.

See [gain review](attempt13-review.json), [payload sizes](attempt13-payloads.json),
[sequences](attempt13-sequences.json), [capture metadata](attempt13-test.json.gz),
[software provenance](attempt13-software.json), and
[persistent mismatches](attempt13-after-first-mismatches.csv).
The uncommitted native file remains under
`../../ued-commissioning-20260928/attempt13-mapfhl/capture.dat` (95290056 bytes),
SHA256 `dc347ebb6af02c10d6cb11ef38677c66b18f6bc231f5d9961c57d4e8b36009ad`.

## Full YAML comparison

The new `--save-config-yaml` capture option saves SaveConfig-equivalent RW/WO
fields twice: `prepared` performs a full root read while camera readout and
triggering are disabled; `armed` saves cached values after PCIe `StartRun`,
before camera enable. This avoids slow full reads during bounded acquisition.
Disabled devices, including ASICs after map writing, can still contain cached
values. Neither these files nor GUI SaveConfig verify pixel SRAM contents.

All **1681 camera fields** occur in both snapshots: **1672 match**, nine differ.
All **401 DAQ PCIe fields** occur in standalone: **382 match**, nineteen differ.
Standalone exposes 32 additional SysMon/QSFP fields, not comparison failures.
The machine-readable [comparison](attempt13-comparison.json) and complete CSVs
([camera](attempt13-camera.csv), [PCIe](attempt13-pcie.csv)) preserve every field.

| Setting | Standalone | DAQ spy | Interpretation |
|---|---|---|---|
| Camera `AcqCore.AsicSyncInjEn` | False | True | Main additional camera-setting candidate |
| `AsicSyncInjDly` | 1000 | 1000 | Matches |
| `AsicRoClkT` / `AsicRoClkHalfT` | 3 / 1 | 3 / 1 | Matches; YAML write-key values must not be mistaken for readback |
| `AdcPipelineDelay` | 68 | 68 | Matches |
| Camera `TrigSrcSel` | 3 | 0 | Internal versus external trigger |
| Auto-trigger period | 434028 | 15625000 | DAQ auto-trigger is disabled; its stored 10 Hz is inactive |
| KCU `UseMiniTpg` | True | False | Local versus external timing |
| KCU `ClkSel` / `ModeSelEn` | LCLS-II Clock / UseClkSel | LCLS-I Clock / UseModeSel | Different timing path; ModeSel is Lcls2Protocol in both |
| Event buffer 0 `TriggerSource` | EVR | XPM | Expected timing-path difference |
| EVR channel 4 `RateType` / `RateSel` | FixedRates / 3 | ControlWord / 6 | Different trigger selection; do not interpret code as camera rate |
| Buffer 0 / 4 `TriggerDelay`; EVR trigger 4 `Delay` | 8577 / 42; 11186 | 8577 / 42; 11186 | Numeric delays match, clocks/sources differ |
| Lane 0 EventBuilder timeout / bypass / blowoff | 144675 / 0 / False | 144675 / 0 / False | Matches |

Other camera differences are the three auto-period/frequency representations,
`RdoutEn` and `TrigEn` (snapshot before enable versus active DAQ), and two
StreamRepeater counters. `AutoTrigEn` is False in both saved snapshots; the
standalone script subsequently enables it for acquisition. All saved ASIC
scalar fields match, including all sixteen `trbit=1`; this does not establish
pixel-memory equality. Camera acquisition pulse lengths and force/value
settings also match.

Remaining KCU differences are master enables for buffers 1,2,3,5,6,7; unused
lanes 1–3 timeout/blowoff; and inactive XpmMini rate. Standalone StartRun enables
all buffers and clears all lane blowoffs; DAQ uses buffers 0/4 and lane 0.

## Meaning of the SYNC difference and next experiment

The checked-out ePix firmware RTL
`epix-quad-1kfps/firmware/common/EpixQuadCommon/rtl/AcqCore.vhd:617`
selects normal `asicSync` when `asicSyncInjEn=0`, and the alternate
`asicSyncInj` waveform when enabled. Lines 554–568 form the alternate pulse
using `AsicSyncInjDly` and `AsicAcqWidth`. Thus this field selects the ASIC SYNC
waveform; it is worth isolating even though the other pulser settings match.
This is source interpretation, not an oscilloscope measurement of this module.

The previous full DAQ run 3 has 43 recurring extra FP at approximately 360 Hz,
while this standalone reference is clean. The saved DAQ YAML is from a later
unrecorded acquisition, so it cannot independently establish every run-3
register value. Both use the intended bundle and map. A controlled next test
should keep the full DAQ sequence, rate, and map fixed and change only
`expert.EpixQuad.AcqCore.AsicSyncInjEn` in backed-up TEST to False, then
reconfigure and compare the same FP/FN coordinates. Repeating both settings
would help separate this field from Configure-to-Configure variability.
No such change was made in this investigation. If this does not explain the
FPs, the remaining timing path and full Configure command ordering still need
isolation; the YAML alone cannot resolve them.

## Cleanup and reproducibility

Initial capture cleanup succeeded. The separate idle-restoration subprocess
then failed before hardware access because Qt expected a display. Its log is
preserved in [initial failure](attempt13-initial-restore-failure.log). A retry
with `QT_QPA_PLATFORM=offscreen` succeeded; [readbacks](attempt13-restoration.json)
confirm readout False, trigger source 3, auto period 15625000, auto trigger and
trigger enables True (the established 10-Hz idle state), and ADC done/pass.
PCIe remains in standalone configuration; a normal DAQ start/Configure is
required before DAQ use. Hardware access is finished. No ConfigDB, external
timing sequence, power, IOC, or Slurm state was changed.

The restore helper now sets headless Qt itself. The wrapper now preserves its
execution audit even if restoration times out or produces no JSON, and removes
the preceding restore result before launch to avoid stale attribution. The
attempt-13 wrapper crashed while archiving its failed restore, so its usual
`execution-attempt13.json` was not produced; capture metadata, original failure,
and successful manual retry are preserved separately rather than fabricating it.
The two offline capture/writer test files pass (27 tests), including a regression
for a restore crash, missing output, timeout, and stale-result exclusion; see
[test log](script-tests.log). A local run initially lacked `SUBMODULEDIR`; the
reported passing run used the configured DAQ environment on the detector node,
with hardware access mocked. The source-only
capture instrumentation does not change the installed DAQ gain writer; the
isolated install has not been rebuilt for these diagnostic script changes.

From a compatible Python environment, `compare_attempt13.py` regenerates the
fresh archival comparison without hardware access. `compare_saved.py` preserves
the earlier attempt-12 reconstruction; it verifies historical script contents
against commit `7377c1524` rather than the now-instrumented capture source.
