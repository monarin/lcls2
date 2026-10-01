# UED run 7: original DAQ writer, IsEn-only correction

All 1,086 complete, undamaged frames are clean outside the unchanged known-bad
list: zero FPs, zero FNs outside known-bad, and 68 known-bad FNs per frame.
The first frame has the same result as the subsequent 1,085 assessed frames.
Measured rate: 360.248310 Hz over 3.011811 seconds.

The recorded map exactly matches expand2 MapFHL; all 16 trbits are 1 and the
detector carrier IDs match the known-bad baseline. All 1,190,308,972 bytes of the
finalized detector stream were parsed through EndRun, with no partial tail,
damaged image or incomplete image. This is not an upstream event-loss audit.

No post-Configure correction or ASIC readback occurred before the run. After
EndRun, all 44 prior FP locations read FH (12) twice with trbit 1.

Data:
`/sdf/data/lcls/drpsrcf/ffb/ued/uedc00107/xtc/uedc00107-r0007-s001-c000.xtc2`.

- [Review and hashes](review.json), [per-frame counts](frames.csv),
  [remaining known-bad mismatches](after-first-mismatches.csv).
- [Implementation, comparison and deployment](../isen-ordering-test/README.md).
- [Post-run ASIC reads](../isen-ordering-test/after-run/pixels.csv).
