# UED run 3 at 360 Hz — September 30, 2026

> Historical September 2026 record. For current repeat-test entry points and archived evidence, see [the repeat-test guide](../../epixquad/README.md). Process IDs, running-state statements and launch paths below describe that session.

**The 43 historical extra FPs persist at 360 Hz with the intended test bundle.**
Experiment `uedc00107`, run 3 contains 4,628 complete, undamaged ePix images.
After excluding the first image, all 4,627 assessed images have 43 FP pixels
outside the 352-pixel known-bad list and 68 FN pixels inside it. Both coordinate
sets exactly match run 1 and the historical expand2 DAQ result. The first image
already has this same pattern.

| Acquisition | Assessed frames | Measured rate | FP outside known | FN inside known |
| --- | ---: | ---: | ---: | ---: |
| Run 1, older submodule release, snapshot | 4898 | 539.9707 Hz | 43 | 68 |
| Run 2, intended test bundle, snapshot | 3428 | 539.9911 Hz | 44 | 68 |
| Run 3, intended test bundle, finalized stream | 4627 | 359.8640 Hz | 43 | 68 |

The extra run-2 FP at zero-based raw `(segment=1, row=175, col=170)` is absent
in run 3. The logs show a new Configure/map write before run 3, so its absence
cannot be attributed solely to the rate change. All other mismatch coordinates
are unchanged. No pixels were added to the known-bad baseline.

The recorded map is the exact expand2 FH/FL input, with 22,169 selected FL
pixels, pixel codes 8/12, all sixteen trbits equal to 1, and detector carrier IDs
matching the known-bad baseline. All 1,920 non-private hardware Configure
fields match run 2, including elementwise array comparisons.

This reproduces the full DAQ discrepancy at the nominal rate used for the
successful standalone DAQ-writer capture. Neither the older submodule bundle
nor the higher 540 Hz rate alone explains the persistent 43-FP pattern. It
does not yet isolate the cause: full DAQ configuration order, trigger state,
and other runtime settings still differ from standalone. The next investigation
should compare the complete Configure/write/start sequence against standalone;
no new hardware test was started during this review.

## Provenance and completeness

Same launch as run 2:
`/sdf/home/u/uedopr/daq/logs/2026/09/30_18:22:15_drp-ued-cmp003:epixquad1kfps_0.log`.
Its header verifies the isolated test install and
`/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1`.
The log records another raw-map write at line 4065, Configure at line 4104,
and the run-3 output file at line 4108. Matching control/TEB logs contain the
new Configure, Enable/Disable, EndRun and subsequent Unconfigure.

Local detector output:
`/u2/pcds/pds/ued/uedc00107/xtc/uedc00107-r0003-s001-c000.xtc2`.
Analysis used the finalized FFB copy:
`/sdf/data/lcls/drpsrcf/ffb/ued/uedc00107/xtc/uedc00107-r0003-s001-c000.xtc2`.

All 5,065,115,508 bytes were parsed through EndRun, with no trailing partial
datagram, damaged image event or missing image segment. Image timestamps span
12.857636 seconds, giving 359.863962 frames/s. The finalized timing stream
`uedc00107-r0003-s000-c000.xtc2` independently contains the same 4,628 event
timestamps in exactly the same order, no top-level event damage, and EndRun.
Stream agreement is not an audit of possible upstream trigger losses.

## Reproduction and evidence

```bash
cd /sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq
PYTHONPATH="$PWD/install/lib/python3.9/site-packages" PS_PARALLEL=none \
  /sdf/group/lcls/ds/ana/sw/conda2/inst/envs/ps_20241122/bin/python \
  notes/ued-full-daq-20260930/review_xtc_gainbits.py \
  /sdf/data/lcls/drpsrcf/ffb/ued/uedc00107/xtc/uedc00107-r0003-s001-c000.xtc2 \
  /tmp/ued-run3-gainbits-new-review
```

Use a new output directory. The reviewer now labels finalized-through-EndRun
files separately from in-progress snapshots; gain-bit classification is unchanged.

- [review.json](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run3-gainbits/review.json): hashes, map/identity checks, counts and rate.
- [after-first-mismatches.csv](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run3-gainbits/after-first-mismatches.csv): all 111 coordinates.
- [frames.csv](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run3-gainbits/frames.csv): per-frame timestamps and gain-bit counts.
- [counts.npz](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run3-gainbits/counts.npz): full-coordinate occurrence arrays and masks.
- [prior-run-comparison.json](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run3-gainbits/prior-run-comparison.json): exact coordinate differences.
- [config-comparison.json](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run3-gainbits/config-comparison.json): run-2/run-3 Configure comparison.
- [stream-comparison.json](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run3-gainbits/stream-comparison.json): timing/ePix timestamp agreement.

All reviewer output checksums were verified after analysis. DAQ, timing,
hardware registers and ConfigDB were not modified by this review.
