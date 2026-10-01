# Standalone capture with UED groupca timing

Prepared offline after the detector was powered off September 30, 2026.
Attempt 17 has now run; see the [external timing result](../../ued-commissioning-20260928/external-timing-20260930/README.md).
The 490 recovered images were clean outside known-bad; a payload-format fix
was validated offline afterward. This extension compares real UED external
timing against the clean internal-trigger attempt 15, while retaining the
standalone baseline and DAQ raw gain-map writer. It does not execute full DAQ
Configure or write ConfigDB.

## Changes and timing recipe

`run_once.py --timing-mode ued-external` selects the dedicated
`drp-ued-cmp003-external.json` profile. The original profile and default internal
mode remain available. The external path uses:

- `/dev/datadev_0`, KCU1500, PGP4 lane 0, image VC0 and register VC1.
- Fiber timing (`UseMiniTpg=False`, local XpmMini hardware disabled),
  `ClkSel=0` (119M), `ModeSelEn=1`, `ModeSel=1` (LCLS-II protocol).
- Run buffer 4: EVR event code 6 (nominal 1080 Hz), delay 11186 in
  `EvrV2TriggerReg[4]`, buffer delay 42.
- DAQ buffer 0: XPM group 0, delay 8577. With `start_ns=94000`, the DAQ
  calculation is `int(start_ns/(1000/119) - PartitionDelay[0]*238 + 9)`.
- Camera `TrigSrcSel=0`, `AutoTrigEn=False`; readout and trigger enabled only
  once the recorder is attached. No auto-trigger frequency is written.

The code verifies register readbacks, an up timing link, a valid upstream
timing ID, and `PartitionDelay[0]=11` before arming. The latter is inferred
from the saved DAQ delay values and must be confirmed live; a mismatch aborts
for review. The timing ID uses the same `timTxId('epixquad')` function as DAQ.
The configuration and connection routines follow the current
`epixquad1kfps_config.py`; the saved DAQ YAML is the numerical reference.

All trigger buffers remain disabled during baseline/map programming. Only
buffers 0/4 and application lane 0 are armed. This intentionally retains
quiescent map programming from our standalone controls; it does not reproduce
every trigger transition during full DAQ Configure.

## Operator sequence for the first hardware test

The sequence was exercised in attempt 17. For a future repeat, use the
corrected decoder and a new attempt number after authorization to acquire. Keep the full DAQ and
non-spy device GUIs stopped. Leave the monitor IOC alone.

On `ued-daq` as `uedopr`, with a working GUI display:

```bash
source /sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq/notes/ued-full-daq-20260930/setup-test.sh
export EPICS_PVA_ADDR_LIST=172.21.36.255
groupca DAQ:UED 0 0
```

Use the existing UED XPM PV service. `xpmpva DAQ:UED:XPM:0` is an optional
monitor/control GUI, not the PV server. Do not launch another PV server or a
full DAQ just to run this capture. Neither GUI was opened during preparation.

In groupca, verify group 0 Master and its readout selection match the UED
360-Hz test sequence, and leave **Run unchecked** until the recorder reports
ready. The script does not set or verify the upstream sequence rate; the
profile's 360 Hz is an expected operator setting. The 1080-Hz EVR run trigger
is separate from the 360-Hz XPM readout trigger. No DAQ transition messages or
experiment/run number are needed for the Rogue file recording.

On `drp-ued-cmp003` as `monarin`, use the source checkout (no DAQ rebuild):

```bash
source /sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq/notes/ued-full-daq-20260930/setup-test.sh
export PYTHONPATH="$UED_GAINMAP_SRC/psdaq:$PYTHONPATH"
export QT_QPA_PLATFORM=offscreen
cd "$UED_GAINMAP_SRC"
python notes/ued-commissioning-20260928/run_once.py \
  --attempt 18 --mode MapFHL --selected-map "$UED_GAINMAP_MASK" \
  --frames 51 --gain-writer daq --asic-ro-clk-ticks 3 \
  --asic-sync-inj-en 1 --timing-mode ued-external \
  --wait-for-trigger-seconds 120 --save-config-yaml
```

Attempt 18 must still be unused; choose a new number if it exists. Do **not**
pass `--rate-hz` or `--rate-sequence-hz`; external mode rejects these internal
camera overrides. The watchdog defaults to 300 seconds: 120 for the operator
wait plus 180 for setup/acquisition overhead. The operator wait starts after
arming; the separate 30-second acquisition timeout starts on the first full
image. Timing-only records cannot satisfy the first-image condition.

Wait for this message in the capture terminal:

```text
READY FOR GROUPCA RUN: XPM 0, group 0; waiting up to 120 s for first full image.
```

Then check **Run** in groupca. Readiness is also recorded in
`attempt18-mapfhl/ready.json` and `console.log`. The JSON state changes from
`ready` to `stopped` on normal completion or an acquisition exception; do not
use a stale ready message to start a completed capture.

When the capture completes, **uncheck Run in groupca**. The script never
writes XPM PVs. It bounds native file size, stops local trigger buffers, drains
and closes the writer, and the wrapper restores the verified camera idle
settings: readout off, internal source 3, auto trigger period 15625000 (10 Hz),
auto trigger and trigger enabled. The PCIe remains in external-test
configuration; normal DAQ must configure again before use.

## Analysis and evidence

The attempt folder contains native `capture.dat`, expected map, mismatch
counts/exports, camera identity, timing configuration/readbacks, before/after
counters, full prepared/armed YAMLs, and source/bundle provenance. The wrapper
writes execution and restoration records separately. Frame and byte counts
can exceed the requested 51 frames due to in-flight/drain frames, within the
existing 512-MiB file limit.

Use the existing offline reviewer after capture:

```bash
python notes/ued-commissioning-20260928/review_gain_map.py \
  notes/ued-commissioning-20260928/attempt18-mapfhl \
  notes/ued-commissioning-20260928/attempt18-mapfhl/review-after-first
```

Exclude the first full image from assessment but retain its counts. Compare
FP/FN outside the 352-pixel known-bad list against the clean internal control,
and compare exact FP coordinates against
`../run3-gainbits/after-first-mismatches.csv` (43 DAQ FPs). The unchanged
68 known-bad FN pixels are expected. Confirm full-frame integrity and rate
evidence before interpreting any gain-bit difference.

Offline preparation passed the external timing, capture-writer, and standalone
test suites, including the real installed Rogue register descriptions without
starting a root. Tests cover route/delay readbacks, link/partition failures,
only-two-buffer arming, separate operator/acquisition timeouts, byte limits,
readiness delivery to the terminal, interrupt/timeout cleanup, internal-mode
regressions, and dry-run isolation. See [tests.log](tests.log).

The real wrapper dry run used attempt 1700 and passed without device access,
capture output, readiness, or restoration. See [dry-run.log](dry-run.log),
[execution](dry-run-execution.json), [metadata](dry-run-test.json.gz), and
[software](dry-run-software.json). Attempt 1700 is an offline preparation
record, not a detector measurement. No ConfigDB, EPICS PV, detector, PCIe,
timing sequence, IOC, power, or Slurm state was changed during preparation.
