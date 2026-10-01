# UED run 6: standalone writer in Configure, no targeted repair

> Historical September 2026 record. For current repeat-test entry points and archived evidence, see [the repeat-test guide](../../epixquad/README.md). Process IDs, running-state statements and launch paths below describe that session.

All 1,084 complete, undamaged images have zero FPs and zero FNs outside the
352-pixel known-bad baseline. The 1,083 frames assessed after the first are
clean, and the first frame has the same pattern. The usual 68 known-bad FNs
remain. The measured rate is 359.971857 Hz over 3.008569 seconds.

Configure used the actual standalone map-writing helper with expand2.npy:
16 full-matrix FH fills, then 22,169 selected FL writes, in standalone input
order. No 44-pixel overwrite or ASIC bitmap readback occurred between Configure
and acquisition. After EndRun, both direct reads of all 44 previously affected
addresses returned code 12 (FH), with trbit 1. The full Configure write replaced
the targeted repair applied for run 5.

The recorded configuration still exactly matches expand2 MapFHL, with all
16 trbits 1 and matching detector carrier IDs. All 1,188,121,052 bytes of the
finalized stream were parsed through EndRun, with no partial tail, damaged
image or missing image segment. This is not an upstream event-loss audit.

Data:
`/sdf/data/lcls/drpsrcf/ffb/ued/uedc00107/xtc/uedc00107-r0006-s001-c000.xtc2`.

- [Full review and checksums](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run6-gainbits/review.json), [per-frame counts](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run6-gainbits/frames.csv),
  [remaining known-bad mismatches](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run6-gainbits/after-first-mismatches.csv), [counts](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run6-gainbits/counts.npz).
- [Implementation, deployment, interpretation and limitations](../standalone-daq-writer/README.md).
- [Post-run ASIC readback](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/standalone-daq-writer/after-run/pixels.csv).

The existing run-4/run-5 reviewer was used again, and its output hashes were
checked. The register-write sequence and camera gating together are validated
in this DAQ test; their individual effects have not yet been separated.

Follow-up: [run 7](../isen-ordering-test/README.md) isolated an additional
difference, IsEn ordering. Reasserting IsEn=True in the original DAQ path
alone also produced clean frames, while retaining its write order and gates.
