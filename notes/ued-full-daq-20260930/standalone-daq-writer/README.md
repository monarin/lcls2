# Standalone gain writer inside DAQ Configure

Run 6 is clean outside the existing known-bad list, without the targeted
44-pixel repair: 1,084 complete, undamaged frames at 359.971857 Hz, zero FPs,
zero FNs outside known-bad, and the usual 68 known-bad FNs per frame. The first
frame has the same clean pattern. After EndRun, all 44 original FP addresses
read code 12 (FH) twice, with trbit 1. See [run-6 results](../run6-gainbits/README.md).

The opt-in adapter calls the unchanged standalone `_write_gain_mode` helper
and `_selected_pixels_from_map` loader. It uses the exact expand2.npy file,
validates its SHA256 and every requested pixel against ConfigDB-derived
Configure metadata, requires all 16 ASICs/MapFHL/trbit 1, and refuses concurrent
debug map overrides. The helper SHA256 is
`b8191dd350dcd7dedaa76f77c1babe517d2b40fa2fd7118fa94f8a6ac19f7008`.

The standalone helper fills all 16 ASIC backgrounds with FH first, then writes
22,169 selected FL pixels in input order, and disables ASIC configuration
access. It completed in 12.032 seconds in this DAQ Configure. The original DAQ
helper instead performs a background fill followed by exceptions for each ASIC
in turn. The adapter also disables camera TrigEn, AutoTrigEn and RdoutEn during
programming, using the standalone gate helpers, and restores them on success.
The existing DAQ Run-trigger disable/enable and remaining Configure sequence
still surround the write. This test changes both write sequence and camera
gating; it does not isolate which difference removes the FPs.

There is no post-Configure correction in this test. The earlier 44-pixel
overwrite was a manual diagnostic, not a permanent hook. Its scripts/evidence
are retained for accounting but were not invoked. Full background/map writing
during the new Configure replaces the previous targeted hardware edits. No
ASIC readback occurred between Configure and BeginRun in run 6; the only
targeted readback was after EndRun, with pixel writing disabled.

## Deployment and selection

The source changes are in `psdaq.configdb.epixquad1kfps_config` and the new
`psdaq.debugtools.epixquad1kfps.standalone_daq_writer` module. They were copied
to the isolated `install-ued-diag-20260930` Python package while the detector
process was stopped. C++ binaries, conda and submodule pins were unchanged;
[deployment hashes](deployment.json) and the compressed original config module
preserve the before/after record. [Operator verification](verify-operator.log)
passed installed Runner parsing, source comparisons, imports and library checks.

`EPIXQUAD_STANDALONE_GAINMAP` selects the adapter and points to the reviewed
expand2 mask. It is exported only for epixquad1kfps_0 by
[epix_standalone_writer.py](epix_standalone_writer.py). The normal DAQ writer
remains the default when that environment variable is absent.

For future operator launches, use:

```bash
source /sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq/notes/ued-full-daq-20260930/standalone-daq-writer/setup.sh
```

The detector process currently uses this variant and DAQ is left Configured.
Do not start a second DAQ instance. Returning to the original launch requires
normal Unconfigure/disconnect and restarting the detector with the diagnostic
launch lacking the selector; merely sourcing a different shell setup does not
change an already running detector process.

The detector log is
`/sdf/home/u/uedopr/daq/logs/2026/09/30_22:59:51_drp-ued-cmp003:epixquad1kfps_0.log`.
It records STANDALONE_DAQ_WRITER begin/complete markers, the helper hash,
mask hash, write count and elapsed time. The other DAQ processes retained their
21:59 launch. Rediscovery selected the new detector PID before Configure.

## Validation and evidence

Four offline tests pass: exact ASIC register-write trace equivalence with the
standalone helper, rejection of mismatched Configure pixel metadata, rejection
of wrong trbits, and gate/ASIC cleanup on an injected write failure. See
[tests.log](tests.log). No 44-pixel exception list is used by the adapter.

- [Summary and artifact hashes](summary.json).
- [Configure record](configure.json), [run record](run.json),
  [DRP evidence](drp-evidence.log).
- [Post-run ASIC CSV](after-run/pixels.csv) and [complete record](after-run/readback.json).

The optional one-shot DMA diagnostic capture was finalized empty during DAQ
reset/rediscovery (`disconnect-before-EndRun`, incomplete, zero pairs). No
paired input/pre-write comparison is claimed for run 6. The gain result comes
from the complete finalized XTC2 detector stream and post-run ASIC readback.
There were no ConfigDB writes or known-bad-list changes.
