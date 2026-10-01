# Full UED DAQ gain-map test instructions

Prepared September 30, 2026; isolated DAQ build subsequently completed successfully
on the same day. Installed-software checks passed on `drp-ued-cmp003` and
`ued-daq`. The build did not change ConfigDB or launch DAQ. TEST backup and
BEAM-to-TEST copying were subsequently completed as uedopr on September 30
at 17:20 PDT; see the execution record below. No DAQ or timing/hardware changes
were made. The ConfigDB upload preview and DAQ configuration parsing passed earlier.
The commands below are for the operator.

Use the existing **ued/TEST** ConfigDB alias for this experiment. Back up TEST,
refresh only the three configurations needed here from BEAM, and restore TEST
afterward. Keep BEAM unchanged. See the
[schema/rollback review](../ued-config-rollback-20260930/README.md) before
switching software generations. The backup and three-entry copy are complete;
MapFHL was subsequently uploaded by the operator and verified at TEST key 1406.
Post-test restoration remains pending.

## Software selection

- lcls2 source: `/sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq`,
  branch `features/ued-epix10ka-rdsrv421-daq`, tested baseline commit
  `84564f7178e528f27a38277e0f23fcade9dca7b1` (documentation-only descendants are
  equivalent for the build). This includes lcls2 PRs 126, 127, 128 and the
  tested raw gain-map path. Use a **fresh DAQ build** of this source.
- SUBMODULEDIR: `/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1`.
  Superproject `4a28be03a7aa77373f0795dae7a1e0d003d4bc3c`; ePix
  `44a7c1cb9ed1869e75b16e24bb610e7e3ee53698`; PCIe apps
  `ccbe3de52fa7f6c09dac0216372121bcfbbc49e5`. Use this already patched bundle
  unchanged, including its nested pins and recorded site patch recipe.
- Environment: `daq_20250402_r9`, Python 3.9, Rogue 6.1.3.

The worktree's old `install/` dates from June and is not the tested source.
The current UED setup link selects `lcls2_091026` and the old March submodule
bundle. Neither is the requested new combination. The private bundle has not
been promoted to a site release or pushed upstream.

## 1. Build the isolated install as monarin

The site recipe `/sdf/group/lcls/ds/ana/sw/conda2/rel/ued/install_release_daq.sh`
uses `source setup_env_daq.sh` followed by `build_all.sh`. We use that same DAQ
build path in an isolated prefix. The current Meson project always includes
psana components even with DAQ enabled; this is not a separate analysis build.
AMI and release publication are outside this test build.

The original `psbuild-rocky9-01` recommendation was checked and corrected:
that host has no SDF group software mount or this SDF home directory. The
isolated build uses Rocky 9.6 on `drp-ued-cmp003`, with two compiler jobs at
nice level 15. Preflight found only the monitor IOC and its CA repeater under
uedopr. Building does not start DAQ or open the detector.

```bash
cd /sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq
nice -n 15 bash notes/ued-full-daq-20260930/build-isolated.sh
```

The helper uses `builddir-ued-gainmap-20260930` and
`install-ued-gainmap-20260930`, leaving the old install and shared UED release
in place. It runs Meson through `build_all.sh -d -c`, then installs the psdaq
entry-point wheel with dependencies and network package lookup disabled.
The root psana wheel is deliberately skipped: its packaging hard-codes
`install/` and would pull libraries from the stale June install. Meson still
installs the newly compiled psana libraries and Python modules; psana command
line entry points are not part of this DAQ-specific packaging step.

Build completed with exit 0 from source `60fdd7f3944edf839351d43f7e4b8d44d6b4c4b0`.
Both hosts loaded the intended install and bundle, matched 392 installed psdaq
Python files to source, resolved all `drp`/`teb` libraries, and passed Unix
read/execute permission checks. Binary hashes matched across hosts.
See [build record](build-record.json), [build log](build.log), and verification
logs for [the detector host](verify-drp-ued-cmp003.log) and
[the control host](verify-ued-daq.log).

`groupca` import produces a nonfatal p4p `ClientProvider.close` teardown warning,
including with the [deployed release](deployed-groupca-check.log); both exit 0.
The warning remains documented, not fixed by this build. Imports are checked
in separate processes to model DAQ service separation. Direct SSH to
`drp-ued-cmp002` was blocked by host-key verification, so that host was not
runtime-checked. These checks do not substitute for the full DAQ test below.

## 2. Select the new build in the UED operator shell

Use a fresh Bash shell as `uedopr` on `ued-daq`, with the normal working GUI
display. Set the following again after sourcing any setup script:

```bash
source /sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq/notes/ued-full-daq-20260930/setup-test.sh
command -v drp daqmgr
python - <<'PY'
import os
from psdaq.configdb import epixquad1kfps_config as config
import ePixQuad
import lcls2_pgp_pcie_apps
print('TESTRELDIR:', os.environ['TESTRELDIR'])
print('SUBMODULEDIR:', os.environ['SUBMODULEDIR'])
print('DAQ configuration:', config.__file__)
print('ePix:', ePixQuad.__file__)
print('PCIe:', lcls2_pgp_pcie_apps.__file__)
PY
test -r "$UED_GAINMAP_MASK"
```

An install owned by monarin can be run by uedopr; it needs read/execute access,
not ownership or write permission. The build uses umask 022. Logs and data
continue to use the operator's existing output paths. The operator subsequently
ran the verification as uedopr on ued-daq (UID 14797): all 392 source comparisons
passed, artifact hashes matched, and there were no import warnings. To repeat
the read-only installed-software check:

```bash
QT_QPA_PLATFORM=offscreen python "$UED_GAINMAP_SRC/notes/ued-full-daq-20260930/verify-install.py"
```

The executables and psdaq import must resolve to the new install; detector
and PCIe imports must resolve to the test bundle. This is an import check,
not hardware initialization. Slurm constructs the DRP Python path from
TESTRELDIR; the standalone source-PYTHONPATH helper is not enough for full DAQ.

The prepared cnf inherits the inspected `~/daq/scripts/epix.py`, validates its
checksum, selects ePix/timing and required DAQ services, and corrects the ePix
fabric interface from `ens2` to verified `ens2np0`. It retains lane mask 0x1,
device 0, type `epixquad1kfps`, timebase 119M, platform 0 and existing output
directory. It omits Andor, Wave8, Piranha, EPICS archiver and AMI from this
focused test. It does not restart the monitor IOC or the separate XPM services.
A dedicated disabled debug-control file prevents an old `/tmp` pattern request
from overriding ConfigDB. Do not add other EPIXQUAD_DEBUG overrides.

## 3. Back up TEST, refresh its test devices from BEAM, then store the map

**Backup and copy completed September 30, 2026, 17:20 PDT.**
TEST advanced from key **1125** to **1405**. BEAM remains at **1402**. All four
TEST entries were read back: the three requested entries match BEAM exactly,
and `epixquad_0` is unchanged. The original backup is
`/sdf/home/u/uedopr/daq/gainmap-test-20260930/alias-backups.json`.
The [execution record](test-prepared.json) includes hashes and verified values.
DAQ had no Slurm jobs and its control endpoint did not respond before copying;
only the monitor and timing services were running.

Do not rerun the completed backup/copy blocks below or overwrite the backup.
In the operator shell, retain the existing record directory for all subsequent
snapshots and restoration:

```bash
export UED_GAINMAP_RECORD="$HOME/daq/gainmap-test-20260930"
```

The initial copy had `gain_mode=1`, `start_ns=94000`, and `AdcPipelineDelay=68`.
**MapFHL upload was verified September 30 at 17:27 PDT, TEST key 1406.** All
540672 pixels match the mask: 22169 FL pixels (code 8) and 518503 FH pixels
(code 12). Map mode is 5; all 16 trbits are 1. `start_ns=94000`, pipeline 68,
and `AsicRoClkHalfT=3` remain correct; obsolete AsicMask is absent. Other TEST
entries and BEAM (key 1402) are unchanged. See [readback details](mapfhl-readback.json).
The full verified configuration is saved in the operator record directory as
`test-mapfhl.json`. The uploader also regenerated the `rateEnum` metadata from
the current template; no additional numeric timing/register changes were found.

The backup, copy, and upload commands below are retained for reproducibility.
These steps are complete for MapFHL; continue at section 4 for the DAQ test.

The read-only September 30 comparison found TEST at key **1125** and BEAM at
key **1402**. TEST already has `epixquad1kfps_0`, `epixquad_0`, `timing_0`, and
`trigger_0`. Its ePix configuration uses the current raw-map schema, but
`start_ns=44000` and `AdcPipelineDelay=61`, versus BEAM's 94000 and 68. Timing
and trigger configurations also differ. Refresh the three required entries
before applying the map; preserve TEST's unrelated `epixquad_0` entry.

Leave DAQ unconfigured and ensure TEST is available for this experiment.
Use the same operator shell and record directory throughout preparation,
recording, and restoration. First save all current TEST entries, the three
BEAM source entries, and their current keys. This block is read-only for
ConfigDB and refuses to overwrite an existing backup:

```bash
mkdir -p "$HOME/daq/gainmap-test-20260930"
export UED_GAINMAP_RECORD="$HOME/daq/gainmap-test-20260930"
python - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
from psdaq.configdb.configdb import configdb
url = 'https://psdmint.sdf.slac.stanford.edu/ws-auth/configdb/ws'
db = configdb(url, 'ued', False, 'configDB', user='uedopr')
names = ['epixquad1kfps_0', 'timing_0', 'trigger_0']
assert {'BEAM', 'TEST'} <= set(db.get_aliases())
keys = {alias: db.get_key(alias) for alias in ('BEAM', 'TEST')}
assert all(key is not None for key in keys.values())
test_names = db.get_devices('TEST')
assert set(names) <= set(test_names), test_names
assert set(names) <= set(db.get_devices('BEAM'))
snapshot = {'url': url, 'hutch': 'ued', 'utc': datetime.now(timezone.utc).isoformat(),
            'keys': keys, 'replace_devices': names, 'configs': {}}
for alias, devices in [('TEST', test_names), ('BEAM', names)]:
    configs = {name: db.get_configuration(alias, name) for name in devices}
    assert all(isinstance(c, dict) and ':types:' in c and c['detName:RO'] == name
               for name, c in configs.items())
    snapshot['configs'][alias] = configs
assert keys == {alias: db.get_key(alias) for alias in keys}, 'Alias changed during backup'
with (Path(os.environ['UED_GAINMAP_RECORD'])/'alias-backups.json').open('x') as f:
    json.dump(snapshot, f, indent=2)
print('Saved TEST and BEAM source configurations at keys', keys)
print('Detector ID:', snapshot['configs']['BEAM']['epixquad1kfps_0']['detId:RO'])
PY
```

The next block **writes only TEST**, copying the saved BEAM configurations for
`epixquad1kfps_0`, `timing_0`, and `trigger_0`. It checks that both aliases still
match the backup keys before the first write. If a key changed, inspect the
change and make a fresh backup in a new record directory before proceeding.
It does not create an alias or copy unrelated BEAM devices:

```bash
python - <<'PY'
import json, os
from pathlib import Path
from psdaq.configdb.configdb import configdb
snapshot = json.loads((Path(os.environ['UED_GAINMAP_RECORD'])/'alias-backups.json').read_text())
assert snapshot['hutch'] == 'ued'
db = configdb(snapshot['url'], 'ued', False, 'configDB', user='uedopr')
names = ['epixquad1kfps_0', 'timing_0', 'trigger_0']
assert snapshot['replace_devices'] == names
assert snapshot['keys'] == {alias: db.get_key(alias) for alias in ('BEAM', 'TEST')}, 'Alias changed since backup'
for name in names:
    cfg = snapshot['configs']['BEAM'][name]
    db.modify_device('TEST', cfg)
    assert db.get_configuration('TEST', name) == cfg, name
for name, cfg in snapshot['configs']['TEST'].items():
    if name not in names:
        assert db.get_configuration('TEST', name) == cfg, name
assert db.get_key('BEAM') == snapshot['keys']['BEAM'], 'BEAM changed during preparation; investigate'
print('Copied and verified', names, 'to TEST; key', db.get_key('TEST'))
PY
```

If copying fails partway, keep DAQ unconfigured and inspect TEST. The original
configurations are in `alias-backups.json`; section 6 restores the three
entries even after a partial copy. Do not overwrite this backup or switch
the test upload to BEAM.

MapFHL matches the latest standalone control. Mask label 1 is FL, label 0 H:

```bash
python -m psdaq.configdb.epixquad_store_gainmap \
  --inst ued --alias TEST --name epixquad1kfps --segm 0 \
  --id serial1234 --user uedopr --file "$UED_GAINMAP_MASK" \
  --map 0:H --map 1:L --test
```

The preview should show shape `(4,352,384)`, 22169 selected FL pixels, and
trbit 1 for all ASICs. Run the same command **without `--test`** to write the test alias.
This is an actual ConfigDB update; preview alone does not install the map.
The current detector ConfigDB ID is `serial1234`; if the fresh backup reports
a different ID, preserve that ID instead. Both the utility's pswww endpoint
and the DAQ's psdmint endpoint returned the same configuration in this review.

Use this current schema-aware utility. It sets `user.gain_mode=5` and
`user.pixel_map_raw`; it also omits the obsolete `expert.SystemRegs.AsicMask`
write (under `expert.EpixQuad` in the complete tree). Do not reintroduce it by
pasting the old expert block back into the new configuration. The ADC pipeline
delay should remain 68 and start_ns 94000. Full Configure applies clock period
3 itself; no standalone register write is needed.

Verify the saved configuration, not only the upload's exit code:

```bash
python - <<'PY'
import json, os
from pathlib import Path
import numpy as np
from psdaq.configdb.configdb import configdb
db = configdb('https://psdmint.sdf.slac.stanford.edu/ws-auth/configdb/ws',
              'ued', False, 'configDB', user='uedopr')
c = db.get_configuration('TEST', 'epixquad1kfps_0')
mask = np.load(os.environ['UED_GAINMAP_MASK']).astype(bool)
actual = np.asarray(c['user']['pixel_map_raw']).reshape(mask.shape)
np.testing.assert_array_equal(actual, np.where(mask, 8, 12))
assert c['user']['gain_mode'] == 5
assert all(c['expert']['EpixQuad']['Epix10kaSaci%d'%i]['trbit'] == 1 for i in range(16))
assert c['expert']['EpixQuad']['RdoutCore']['AdcPipelineDelay'] == 68
assert c['user']['start_ns'] == 94000
assert 'AsicMask' not in c['expert']['EpixQuad']['SystemRegs']
with (Path(os.environ['UED_GAINMAP_RECORD'])/'test-mapfhl.json').open('x') as f:
    json.dump(c, f, indent=2)
print('Verified MapFHL; selected FL pixels:', int(mask.sum()))
PY
```

## 4. Start full DAQ and select external 360 Hz timing

Stop any existing platform-0 DAQ jobs through the config that launched them;
do not run two platform-0 DAQs simultaneously. Leave the monitor IOC alone.
Close standalone capture/devGui so no second register client owns VC1.

```bash
daqmgr start "$UED_GAINMAP_CNF"
daqmgr status "$UED_GAINMAP_CNF"
```

The prepared launcher sets `control -C TEST`; the normal UED launcher defaults
to BEAM. Creating or editing configurations does not switch an already-running
DAQ's selection. Confirm **TEST** in the control GUI before Configure.

Starting fresh DRP processes is intentional: standalone left local-XpmMini
timing selected, and a mere Enable does not restore the production path.
In the control GUI, select `epixquad1kfps_0` and `timing_0`, TEST configuration,
and the intended current experiment. Connect and Configure, checking the
matched DRP/control/TEB log group under `~/daq/logs/2026/09/`.

Before accepting data, the DRP header must show the new TESTRELDIR and test
SUBMODULEDIR. Require successful ADC checking and the message
`Using user.pixel_map_raw write path`, with codes 8 and 12. There must be no
`loaded debug override` message, AsicMask verification failure, ADC timeout,
Configure error, or missing/1600-byte image subframe stream.

With acquisition stopped, program the Quad 360 Hz sequence from `uedopr@ued-daq`.
The installed operator wrapper selects 360 Hz by default; it has no `--a360`
option. `--cam 0` avoids Andor camera PV updates. Use a subshell because the
operator setup script changes the environment, and the outer shell must retain
our isolated DAQ test build.

```bash
# Preview only: no timing writes.
(
  source /sdf/home/u/uedopr/daq/scripts/setup_env.sh &&
  python /cds/group/pcds/dist/pds/ued/scripts/ued_seq_setup_new.py \
    --cam 0 --pv DAQ:UED:XPM:0 --eng 0 --test
)

# Apply after checking the preview.
(
  source /sdf/home/u/uedopr/daq/scripts/setup_env.sh &&
  python /cds/group/pcds/dist/pds/ued/scripts/ued_seq_setup_new.py \
    --cam 0 --pv DAQ:UED:XPM:0 --eng 0
)
```

Verified on September 30, 2026 as `uedopr`: the preview printed
`seqprogram --pv DAQ:UED:XPM:0 --seq 0:/cds/group/pcds/dist/pds/ued/scripts/ued_360Hz.py --start --reset`.
Only the preview was executed during this verification. The apply command
starts/resets engine 0 and sets `DAQ:UED:XPM:0:PART:0:SeqMask` to 0, removing
DAQ control over sequence enable/disable. The sequence requests control bit 0
and labels it `360 Hz VTS1`. The interactive `ued_seq` alias invokes this same
wrapper after sourcing the normal operator environment.

Verify successful programming, the selected XPM sequence/readout group, and
the measured DAQ event rate after enabling acquisition. The run trigger
(EVR event code 6, normally 1080 Hz) and 360 Hz
DAQ/readout trigger are separate. Do not set the camera AutoTrigFreqHz for this
full DAQ test. The expected camera mode is external source 0, AutoTrigEn=0,
TrigEn=1, RdoutEn=1. Expected PCIe source is UseMiniTpg=false, ClkSel=0.

For strict interpretation, retain the saved full BEAM configuration: full
DAQ also reapplies other expert settings, including database AsicSyncInjEn=1
versus standalone's deliberate 0. This experiment tests full DAQ behavior,
not a claim that rate is the sole difference from standalone.

## 5. Record and repeat

1. Enable recording and take a short MapFHL run containing at least 51 complete
   detector frames (about one second at 360 Hz gives a useful short sample).
   Stop on missing images, persistent damage or Configure failure; keep the logs.
2. End the run, **Unconfigure → Configure**, and repeat twice. Three independent
   Configure cycles exercise repeated gain-map writes; BeginRun alone may not.
3. Unconfigure. Repeat the upload preview/write with `--map 0:M --map 1:L` for
   MapFML. Verify the same raw pixel codes/counts and trbit 0 for all ASICs;
   save as `test-mapfml.json`. Take three short runs with Configure between them.
4. Record experiment, run number, mode, configure cycle, measured rate, both
   software paths, configuration backup and matched DRP/control/TEB logs.

Analyze raw XTC gain bits against this same mask, verify the map/trbits embedded
in each run's Configure, and retain the first frame separately. After excluding
frame 1, the standalone reference is 0 FP and 68 FN per frame, all within the
352-pixel list in `../ued-commissioning-20260928/fl-persistent-mismatch-candidates.csv`.
Export every new FP/FN coordinate and occurrence count; do not expand the known
list during comparison. Preserve damage/missing-image counts separately.
The standalone `.dat` review scripts are not XTC readers. Supply the recorded
experiment/run numbers for the run-aware analysis; do not infer success from
an image display alone.

### First full-DAQ result: run 1

[Run 1 review](run1-gainbits/README.md) confirms 43 persistent FP pixels outside
the known-bad list and 68 FN pixels within it across 4,898 assessed frames.
The coordinates exactly match the historical production DAQ pattern. The
measured rate was about 540 Hz. The DRP header shows the new test install but
the **older `lcls2_submodules_03122026` bundle**, so this run does not validate
the intended new install/submodule pair. The reviewed FFB file was still in
progress; see the report for the exact complete-prefix scope.

### Run 2 with the intended test bundle

[Run 2 review](run2-gainbits/README.md) confirms the new test bundle in the DRP
header, but the FP discrepancy persists: 44 FP pixels outside the known-bad
list and 68 FN pixels within it in all 3,428 assessed frames at about 540 Hz.
All 43 historical/run-1 FPs remain, plus raw `(segment=1, row=175, col=170)`.
The saved hardware configuration is identical to run 1 in all 1,920 fields.
This excludes the old bundle as the sole explanation for the recurring FP
pattern; it does not isolate rate or programming/configuration sequence.
The reviewed FFB copy was still in progress, as detailed in the report.

### Run 3 at 360 Hz

[Run 3 review](run3-gainbits/README.md) confirms that the 43 historical extra
FPs persist with the intended test bundle at about 360 Hz. All 4,627 assessed
frames have 43 FP pixels outside the known-bad list and 68 FN pixels within it.
The additional run-2 FP is absent, but run 3 also involved a fresh Configure.
The finalized ePix stream contains 4,628 full undamaged images through EndRun;
the timing stream has exactly the same event timestamps. Saved hardware
configuration matches run 2 in all 1,920 fields. The old bundle and higher
rate are therefore insufficient explanations for the persistent 43-FP pattern.

## 6. Restore TEST after the experiment

End/Unconfigure and stop this test cnf before restoration. Confirm no other
operator is using or editing TEST. Preserve the final test configurations
alongside the run records, then restore only the three entries changed by
this experiment from `alias-backups.json`. This also recovers a partial copy
from section 3. If TEST has acquired unrelated changes, coordinate those
before restoring. This block writes TEST; it does not write BEAM:

```bash
python - <<'PY'
import json, os
from pathlib import Path
from psdaq.configdb.configdb import configdb
folder = Path(os.environ['UED_GAINMAP_RECORD'])
snapshot = json.loads((folder/'alias-backups.json').read_text())
assert snapshot['hutch'] == 'ued'
db = configdb(snapshot['url'], 'ued', False, 'configDB', user='uedopr')
names = ['epixquad1kfps_0', 'timing_0', 'trigger_0']
assert snapshot['replace_devices'] == names
original = snapshot['configs']['TEST']
key = db.get_key('TEST')
assert key is not None
assert set(db.get_devices('TEST')) == set(original), 'TEST device list changed; inspect before restoring'
current = {name: db.get_configuration('TEST', name) for name in original}
assert all(isinstance(c, dict) and ':types:' in c for c in current.values())
assert key == db.get_key('TEST'), 'TEST changed during readback'
for name in set(original) - set(names):
    assert current[name] == original[name], 'Unrelated TEST entry changed: ' + name
with (folder/'test-before-restore.json').open('x') as f:
    json.dump({'key': key, 'configs': current}, f, indent=2)
for name in names:
    db.modify_device('TEST', original[name])
    assert db.get_configuration('TEST', name) == original[name], name
assert all(db.get_configuration('TEST', name) == cfg for name, cfg in original.items())
print('Restored original TEST contents; new key:', db.get_key('TEST'))
PY
```

Restoration creates a new ConfigDB key containing the saved contents; it does
not rewind the key number to 1125. If restoration fails partway, inspect the
saved files and current TEST entries before retrying; the exclusive output
file prevents silently replacing the pre-restoration record. Keep all backups,
map snapshots, run numbers, and logs. No alias deletion is needed.

Return to the previous complete release/bundle and its compatible alias
(normally BEAM), with fresh detector initialization and Configure. BEAM's
existing raw schema is not compatible with pre-migration code. The BEAM entry
in `alias-backups.json` preserves today's schema, not the removed legacy schema.
This test does not promote the bundle; upstream pin/patch publication follows
successful validation.
