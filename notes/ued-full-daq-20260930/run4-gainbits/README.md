# UED run 4: incoming versus pre-write versus disk

> Historical September 2026 record. For current repeat-test entry points and archived evidence, see [the repeat-test guide](../../epixquad/README.md). Process IDs, running-state statements and launch paths below describe that session.

Run 4 (`uedc00107`, September 30, 2026) places the extra FP pixels upstream
of DRP image conversion and disk writing in the captured sample. All 51 paired
incoming/pre-write images have identical uint16 pixel words; all 51 pre-write
datagrams are byte-for-byte identical to the corresponding recorded datagrams.
The incoming data already contain 44 FP pixels outside the 352-pixel known-bad
baseline. This is evidence against conversion or disk-byte corruption causing
these FPs; it does not rule out acquisition/configuration differences or effects
of recording load before the incoming capture point.

| Stage | Matched frames | Assessed after first | FP outside known per frame | FN outside known per frame |
| --- | ---: | ---: | ---: | ---: |
| Incoming lane-0 DMA | 51 | 50 | 44 | 0 |
| Pre-write datagram | 51 | 50 | 44 | 0 |
| Matching XTC2 events | 51 | 50 | 44 | 0 |
| Entire recorded detector stream | 2322 | 2321 | 44 | 0 |

All stages have 68 FNs, all inside the known-bad list. The first frame already
has the same pattern. The full run contains 2,322 complete, undamaged images
over 6.450576 seconds (359.812825 frames/s); all 2,321 assessed images have the
same mismatch coordinates. No pixels were added to the known-bad baseline.

All 43 run-3 FP coordinates persist. The additional zero-based raw coordinate
is `(segment=1, row=175, col=170)`, previously seen in DAQ run 2. Run 4's entire
FP and FN coordinate sets exactly match DAQ run 2. The extra pixel is not a new
location introduced by this diagnostic build.

The recorded configuration matches expand2 MapFHL, with 22,169 FL pixels,
pixel codes 8/12 and all sixteen trbits equal to 1. Carrier IDs match the
known-bad detector. The complete 2,542,443,604-byte finalized stream was parsed
through EndRun with no partial tail. This is not an audit of upstream event loss.

## Evidence and reproduction

- [Full-run review](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run4-gainbits/review.json), [per-frame counts](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run4-gainbits/frames.csv),
  [mismatch coordinates](after-first-mismatches.csv), [count arrays](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run4-gainbits/counts.npz).
- [Three-stage comparison](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/diagnostic-hook/run4-comparison/comparison.json),
  including SHA256 checksums, timestamps, pulse IDs and per-event equality.
- [Prior-run coordinate comparison](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/diagnostic-hook/run4-comparison/prior-run-comparison.json).
- [Capture manifest](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/diagnostic-hook/run4-comparison/capture-manifest.json)
  and [DRP log evidence](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/diagnostic-hook/run4-comparison/drp-evidence.log).

The capture reports `status=complete`, `paired=51`, zero capture errors and
finish reason EndRun. All sampled events were selected for disk writing. The
incoming timing identity was checked against the hook, and the captured
Configure equals the disk Configure byte-for-byte.

Diagnostic input directory:
`/sdf/home/u/uedopr/daq/gainmap-diagnostic-20260930/epix-diag-3485185-1790830746683655190`.

Finalized detector stream:
`/sdf/data/lcls/drpsrcf/ffb/ued/uedc00107/xtc/uedc00107-r0004-s001-c000.xtc2`.
Detector-local destination was
`/u2/pcds/pds/ued/uedc00107/xtc/uedc00107-r0004-s001-c000.xtc2`.

From the worktree, reproduce with the existing `diagnostic-hook/compare.py`
and `review_xtc_gainbits.py`, supplying these inputs and fresh output directories.
Analysis used `ps_20241122/bin/python`, `PS_PARALLEL=none`, and
`PYTHONPATH=psdaq:install/lib/python3.9/site-packages`. Comparator and full-run
review agree on the file SHA256; full-run output checksums were verified.

The next useful comparison is the full DAQ configuration/write/start sequence
against the clean standalone external-timing capture. A fresh diagnostic run
with XTC2 recording disabled can additionally distinguish recording-load effects;
it requires restarting the one-shot diagnostic DRP and analyzing input/pre-write
samples without the current comparator's disk-selection requirement. No further
acquisition, register changes or ConfigDB writes were performed during this review.
