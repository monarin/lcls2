# Original DAQ writer with IsEn ordering corrected: run 7

> Historical September 2026 record. For current repeat-test entry points and archived evidence, see [the repeat-test guide](../../epixquad/README.md). Process IDs, running-state statements and launch paths below describe that session.

Run 7 passed: 1,086 complete, undamaged images at 360.248310 Hz, with zero FP
or FN pixels outside the unchanged 352-pixel known-bad list. The first frame
is also clean. All frames retain the expected 68 known-bad FNs. After EndRun,
all 44 formerly affected ASIC locations read code 12 (FH) twice, with trbit 1.

The sole gain-programming correction was to reassert `IsEn=True` after the
expert configuration and before the original DAQ map writer. The original
DAQ algorithm, per-ASIC matrix-fill/exception order, camera gating, timing,
ConfigDB map and final IsEn=False cleanup were retained. No standalone helper,
44-pixel correction or pre-run ASIC readback was used. The detector log records
the new IsEn marker followed by the original raw-map path: 22,169 pixel updates
in 12.483 seconds. There is no STANDALONE_DAQ_WRITER invocation.

This is hardware evidence that the IsEn correction alone is sufficient for
this map and run. It supports the ordering defect as the cause of the observed
FPs. The electrical mechanism producing additional row-0 FL pixels remains
unresolved; this test does not establish that IsEn is a blanket SACI write-enable.
It is one short test after a fresh Configure, not a long-term reliability study.

## Preserved standalone port and active selection

The standalone port from commit `5c1800656` remains available without changes
to its adapter, helper or launch files. Their installed SHA256 values are
unchanged and recorded in [deployment.json](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/isen-ordering-test/deployment.json). The new IsEn
code runs only when the standalone adapter is not selected. An offline full
Configure replay confirms the standalone register sequence is exactly unchanged.
[pre-isen-config.py.gz](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/isen-ordering-test/pre-isen-config.py.gz) also preserves the installed
configuration module before this change.

The detector process currently runs the **original DAQ writer with the IsEn
correction**, PID 3530560 on drp-ued-cmp003. DAQ is left Configured, alias TEST,
recording enabled, last run 7. The post-run read restored all original trigger
and readout gates without errors. Other DAQ processes were not restarted.

For a future launch of the corrected original writer:

```bash
source /sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq/notes/ued-full-daq-20260930/isen-ordering-test/setup.sh
```

The prior standalone port remains selectable using
`notes/ued-full-daq-20260930/standalone-daq-writer/setup.sh`.
Sourcing a setup file only changes the shell; switching an existing detector
process requires a normal Unconfigure/disconnect and restart. Do not launch
a second DAQ instance. No ConfigDB values were changed for this test.

## Validation and evidence

- [Three offline regression tests](test_ordering.py) pass: full Configure writes
  with IsEn=1 despite DB IsEn=0, exact original map-command trace is preserved;
  standalone full Configure is unchanged; no-map updates do not acquire an
  extra IsEn transition. [Test output](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/isen-ordering-test/offline-tests.log).
- [Operator verification](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/isen-ordering-test/verify-operator.log) passed imports, installed-source
  comparisons, libraries and access modes. The installed Runner parsed the
  launch, which explicitly clears EPIXQUAD_STANDALONE_GAINMAP.
- [Configure record](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/isen-ordering-test/configure.json), [run record](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/isen-ordering-test/run.json),
  [DRP evidence](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/isen-ordering-test/drp-evidence.log), [result summary](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/isen-ordering-test/summary.json).
- [Run-7 analysis](../run7-gainbits/README.md),
  [ASIC readback CSV](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/isen-ordering-test/after-run/pixels.csv), [readback record](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/isen-ordering-test/after-run/readback.json).

Detector log:
`/sdf/home/u/uedopr/daq/logs/2026/09/30_23:26:24_drp-ued-cmp003:epixquad1kfps_0.log`.

As in run 6, reset/rediscovery finalized the optional one-shot diagnostic
capture incomplete with zero pairs. No new DMA/pre-write comparison is claimed.
The result uses the complete finalized XTC2 detector stream and post-run ASIC
readback. Prior run-4 paired captures already located the FPs before disk writing.

Run the offline regression from the repository root with:

```bash
env PYTHONPATH=psdaq \
  /sdf/group/lcls/ds/ana/sw/conda2/inst/envs/ps_20241122/bin/python \
  notes/ued-full-daq-20260930/isen-ordering-test/test_ordering.py -v
```
