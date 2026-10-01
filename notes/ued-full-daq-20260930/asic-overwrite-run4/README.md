# Targeted FH overwrite after UED run 4

> Historical September 2026 record. For current repeat-test entry points and archived evidence, see [the repeat-test guide](../../epixquad/README.md). Process IDs, running-state statements and launch paths below describe that session.

All 44 FP locations accepted FH and retained it through the tested cleanup
sequence and a later read after trigger gates had been restored.

| Stage | Code at all 44 addresses | trbit |
| --- | --- | --- |
| Before overwrite, two reads | 8 (FL) | 1 |
| Immediately after targeted writes, two reads | 12 (FH) | 1 |
| After all affected ASICs returned to readout-ready state, two reads | 12 (FH) | 1 |
| Separate follow-up after trigger gates restored, two reads | 12 (FH) | 1 |

The overwrite test took 2.212 seconds. The separate follow-up read took
1.796 seconds. Exactly one code-12 write was issued at each of the 44 targets.
Only ASIC row 0 on ASICs 2, 3, 4, 5, 6, 8, 10 and 15 was addressed, using the
run-4 mismatch CSV and shared raw-to-ASIC coordinate conversion.

This demonstrates that these addresses can accept and retain FH under this
quiescent programming sequence. It supports investigating the original full
DAQ programming sequence or a subsequent overwrite, rather than treating the
44 locations as permanently unable to store FH. It does not establish which
operation originally left them at code 8.

## Procedure and state

Used existing DAQ Rogue servers on drp-ued-cmp003, without a new hardware root
or DMA reader. Preflight checked identity and unallocated DAQ state. Trigger
and readout gates were temporarily disabled. All 44 baseline values and trbits
were checked before the first pixel write. Each ASIC was rechecked immediately
before writing, then read twice after writes. Each was returned to readout-ready
state with CmdPrepForRead, original IsEn, and original local enable.

After all affected ASICs completed cleanup, configuration access was re-entered
to read the same addresses again; no pixel writes occurred in that stage.
Trigger/readout gates were then restored. A separate invocation subsequently
paused the restored gates, read all 44 addresses twice, and restored them again.
Both invocations verified gate restoration, unchanged trbits, and empty cleanup
error lists. DAQ remained unallocated. The follow-up establishes retention after
gate restoration; this test did not measure trigger counts or acquire images.

The hardware pixel settings are intentionally left at FH, matching the existing
DB map. No ConfigDB values, full gain map, DAQ Configure or XPM sequence were
changed. The run-4 XTC2 data and original FL readback evidence remain unchanged.
A normal Configure would rewrite the map, so any frame test of this targeted
repair must preserve the current map or explicitly repeat the repair after
Configure. No frame test was started.

## Evidence

- [Summary and artifact checksums](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/asic-overwrite-run4/summary.json).
- [Overwrite CSV](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/asic-overwrite-run4/measurement/pixels.csv) and [full record](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/asic-overwrite-run4/measurement/readback.json).
- [Follow-up CSV](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/asic-overwrite-run4/after-trigger-restore/pixels.csv) and
  [full follow-up record](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/asic-overwrite-run4/after-trigger-restore/readback.json).
- [Execution log](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/asic-overwrite-run4/execution.log), [follow-up log](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/asic-overwrite-run4/after-trigger-restore.log),
  and [preflight](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/asic-overwrite-run4/preflight/readback.json).
- [Five passing offline checks](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/asic-overwrite-run4/offline-tests.log), covering target mapping,
  exact writes with address-specific storage, retention across cleanup,
  read-failure cleanup, and refusal with an unexpected trbit.

The shared [reader](../read-fp-pixels.py) defaults to preflight; `--execute`
performs reads only, and pixel writes additionally require `--overwrite-fh`.
The overwrite requires the saved 44 addresses to read code 8 with trbit 1.
It will refuse to repeat the overwrite against their current code-12 state.
