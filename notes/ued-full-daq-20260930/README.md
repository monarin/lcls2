# Full UED DAQ gain-map test instructions

Prepared September 30, 2026. This preparation did not build an install, write
ConfigDB, start DAQ, or change timing/hardware. The ConfigDB upload preview and
DAQ configuration parsing passed. The commands below are for the operator.

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

## 1. Build on the Rocky 9 build host as monarin

In a fresh Bash shell on `psbuild-rocky9-01`:

```bash
cd /sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq
git status --short
git merge-base --is-ancestor 84564f717 HEAD
source setup_env_daq.sh
export SUBMODULEDIR=/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1
export BUILDDIR="$PWD/builddir-ued-gainmap-20260930"
export TESTRELDIR="$PWD/install-ued-gainmap-20260930"
./build_all.sh -d -j 8
test -x "$TESTRELDIR/bin/drp"
cmp psdaq/psdaq/configdb/epixquad1kfps_config.py \
    "$TESTRELDIR/lib/python3.9/site-packages/psdaq/configdb/epixquad1kfps_config.py"
```

Proceed only after a successful build/install and source comparison. Do not
use `-f` or overwrite the existing shared UED release. The build above has not
been executed as part of this instruction-preparation task.

## 2. Select the new build in the UED operator shell

Use a fresh Bash shell as `uedopr` on `ued-daq`, with the normal working GUI
display. Set the following again after sourcing any setup script:

```bash
export UED_GAINMAP_SRC=/sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq
source "$UED_GAINMAP_SRC/setup_env_daq.sh"
export TESTRELDIR="$UED_GAINMAP_SRC/install-ued-gainmap-20260930"
export DAQ_TESTRELDIR="$TESTRELDIR"
export DAQ_CONDA_PREFIX="$CONDA_PREFIX"
export DAQ_CONDA_DEFAULT_ENV="$CONDA_DEFAULT_ENV"
export SUBMODULEDIR=/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1
export PATH="$TESTRELDIR/bin:$PATH"
export PYTHONPATH="$TESTRELDIR/lib/python3.9/site-packages"
export PYTHONDONTWRITEBYTECODE=1
export UED_GAINMAP_CNF="$UED_GAINMAP_SRC/notes/ued-full-daq-20260930/epix_gainmap_test.py"
export UED_GAINMAP_MASK="$UED_GAINMAP_SRC/notes/ued-commissioning-20260928/map-inputs/expand2.npy"
hash -r
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

## 3. Back up BEAM, then store the exact gain map

Before editing the configuration, leave DAQ unconfigured. Save a fresh backup
in the operator's directory (not over the recorded September 28 snapshot):

```bash
mkdir -p "$HOME/daq/gainmap-test-20260930"
export UED_GAINMAP_RECORD="$HOME/daq/gainmap-test-20260930"
python - <<'PY'
import json, os
from pathlib import Path
from psdaq.configdb.configdb import configdb
db = configdb('https://psdmint.sdf.slac.stanford.edu/ws-auth/configdb/ws',
              'ued', False, 'configDB', user='uedopr')
cfg = db.get_configuration('BEAM', 'epixquad1kfps_0')
with (Path(os.environ['UED_GAINMAP_RECORD'])/'beam-before.json').open('x') as f:
    json.dump(cfg, f, indent=2)
print('Saved original', cfg['detName:RO'], cfg['detId:RO'])
PY
```

MapFHL matches the latest standalone control. Mask label 1 is FL, label 0 H:

```bash
python -m psdaq.configdb.epixquad_store_gainmap \
  --inst ued --alias BEAM --name epixquad1kfps --segm 0 \
  --id serial1234 --user uedopr --file "$UED_GAINMAP_MASK" \
  --map 0:H --map 1:L --test
```

The preview should show shape `(4,352,384)`, 22169 selected FL pixels, and
trbit 1 for all ASICs. Run the same command **without `--test`** to write BEAM.
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
c = db.get_configuration('BEAM', 'epixquad1kfps_0')
mask = np.load(os.environ['UED_GAINMAP_MASK']).astype(bool)
actual = np.asarray(c['user']['pixel_map_raw']).reshape(mask.shape)
np.testing.assert_array_equal(actual, np.where(mask, 8, 12))
assert c['user']['gain_mode'] == 5
assert all(c['expert']['EpixQuad']['Epix10kaSaci%d'%i]['trbit'] == 1 for i in range(16))
assert c['expert']['EpixQuad']['RdoutCore']['AdcPipelineDelay'] == 68
assert 'AsicMask' not in c['expert']['EpixQuad']['SystemRegs']
with (Path(os.environ['UED_GAINMAP_RECORD'])/'beam-mapfhl.json').open('x') as f:
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

Starting fresh DRP processes is intentional: standalone left local-XpmMini
timing selected, and a mere Enable does not restore the production path.
In the control GUI, select `epixquad1kfps_0` and `timing_0`, BEAM configuration,
and the intended current experiment. Connect and Configure, checking the
matched DRP/control/TEB log group under `~/daq/logs/2026/09/`.

Before accepting data, the DRP header must show the new TESTRELDIR and test
SUBMODULEDIR. Require successful ADC checking and the message
`Using user.pixel_map_raw write path`, with codes 8 and 12. There must be no
`loaded debug override` message, AsicMask verification failure, ADC timeout,
Configure error, or missing/1600-byte image subframe stream.

Use the hutch's Quad 360 Hz sequence, not the Andor exposure sequence. History
records this operator-shell command:

```bash
type -a ued_seq
ued_seq --help
# If this is the hutch wrapper exposing --a360:
ued_seq --a360
```

The exact wrapper is not present in this checkout; confirm its availability
in the normal `uedopr` shell. Do not substitute `ued_seq_setup --a360`: the
checked-in `ued_seq_setup` does not support that flag. If the wrapper is absent,
the hutch timing setup must be identified before running; do not guess an XPM
sequence engine. Verify the selected XPM sequence/readout group and measured
DAQ event rate. The run trigger (EVR event code 6, normally 1080 Hz) and 360 Hz
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
   save as `beam-mapfml.json`. Take three short runs with Configure between them.
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

End/Unconfigure and stop this test cnf when finished. Keep the previous release
and `beam-before.json` for deliberate rollback. This test does not promote the
bundle; upstream pin/patch publication follows successful validation.
