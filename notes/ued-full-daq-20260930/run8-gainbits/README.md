# UED run 8: clean master-based IsEn PR branch

All 1,084 complete, undamaged frames have zero FP/FN outside the unchanged
352-pixel known-bad baseline. The first frame is clean as well as the 1,083
assessed frames after it. Each frame has the expected 68 known-bad FNs.
Measured rate: 360.282122 Hz.

The full DAQ acquisition processes (control, TEB, timing DRP and detector DRP)
ran from a separate build of `fix/epixquad1kfps-isen-ordering`, commit `a4907da6f`,
based on upstream master `a4be0f987`. The branch changes one file by four lines:
reassert IsEn=True after expert configuration, before map writing. There is no
standalone adapter, diagnostic capture hook, targeted correction or pre-run
ASIC readback. XPM/groupca timing and GUI processes were retained.

The recorded map exactly matches expand2, with all 16 trbits 1 and the expected
carrier IDs. All 1,188,121,052 bytes of the finalized detector stream were
parsed through EndRun. No damaged or incomplete image was found. This is not
an upstream event-loss audit. Post-run direct reads of all 44 prior FP locations
returned FH code 12 twice, with trbit 1; gates were restored without errors.

Data:
`/sdf/data/lcls/drpsrcf/ffb/ued/uedc00107/xtc/uedc00107-r0008-s001-c000.xtc2`.
Both streams are also copied and checksum-verified under
`/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/xtc/`.

- [Full review](review.json), [per-frame counts](frames.csv),
  [remaining known-bad mismatches](after-first-mismatches.csv).
- [Validation record](../closeout/pr-validation.json),
  [post-run ASIC reads](../closeout/pr-after-run/pixels.csv).
- [Post-mortem and PR evidence](../POST_MORTEM.md).
