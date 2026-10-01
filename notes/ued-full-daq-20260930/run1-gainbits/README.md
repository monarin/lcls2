# UED run 1 gain-bit review — September 30, 2026

> Historical September 2026 record. For current repeat-test entry points and archived evidence, see [the repeat-test guide](../../epixquad/README.md). Process IDs, running-state statements and launch paths below describe that session.

**The historical extra-FP pattern recurs in full DAQ.** In experiment
`uedc00107`, run 1, every one of 4,898 assessed frames has 43 FP pixels outside
the 352-pixel known-bad list and 68 FN pixels inside it. Both coordinate sets
exactly equal the historical expand2 experiment `ued1015980`, run 2, lists.
The FPs lie on raw rows 175/176. The first recorded frame has the same pattern;
excluding it does not change this finding. Do not add these 43 pixels to the
known-bad baseline during the comparison.

The saved Configure contains the exact expand2 map (22,169 FL pixels), pixel
codes 8/12, and all sixteen trbits equal to 1: MapFHL. Carrier IDs match the
production detector used to establish the known-bad list. Thus the Configure
map agrees with the intended input, while the event gain bits disagree at the
43 FP coordinates. Configure metadata alone is not hardware pixel readback.

## Software discrepancy

The supplied detector log, lines 8–9, reports:

- `TESTRELDIR=/sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq/install-ued-gainmap-20260930`
- `SUBMODULEDIR=/sdf/group/lcls/ds/ana/sw/conda2-v4/rel/lcls2_submodules_03122026`

The second path is the older release, not the intended
`/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1`.
`psdaq.utils.enable_epix_quad1kfps` derives its library paths from this variable.
This result therefore does **not** validate the intended new software pair or
establish whether the FP pattern persists with the new submodule bundle.
Before a comparison run, relaunch from the prepared test environment and
check both paths in the new DRP header. No relaunch or live setting change was
performed during this analysis.

## Data and scope

Source log:
`/sdf/home/u/uedopr/daq/logs/2026/09/30_17:57:30_drp-ued-cmp003:epixquad1kfps_0.log`.
The matching control and TEB logs show Configure and the recorded acquisition
beginning at approximately 18:04 PDT. The detector log names the local file:
`/u2/pcds/pds/ued/uedc00107/xtc/uedc00107-r0001-s001-c000.xtc2`.

Analysis used the readable FFB transfer:
`/sdf/data/lcls/drpsrcf/ffb/ued/uedc00107/xtc/uedc00107-r0001-s001-c000.xtc2.inprogress`.
It contained 5,362,577,408 bytes at inspection. We parsed the complete
5,361,578,548-byte prefix, containing 4,899 full, undamaged image events.
The remaining 998,860 bytes form an incomplete final datagram and were not
interpreted. No EndRun was present. This is a bounded snapshot, not a finalized
whole-run integrity or event-loss check.

The complete image timestamps span 9.070862 seconds, giving 539.9707 frames/s.
This is a **540 Hz** acquisition, whereas the recent standalone comparison
was at 360 Hz. The differing bundle and trigger conditions prevent attribution
of the discrepancy to the gain writer or rate alone.

## Reproduction and artifacts

On SDF, using the existing generic psana datagram decoder (no calibration or
detector hardware access):

```bash
cd /sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq
PYTHONPATH="$PWD/install/lib/python3.9/site-packages" PS_PARALLEL=none \
  /sdf/group/lcls/ds/ana/sw/conda2/inst/envs/ps_20241122/bin/python \
  notes/ued-full-daq-20260930/review_xtc_gainbits.py \
  /sdf/data/lcls/drpsrcf/ffb/ued/uedc00107/xtc/uedc00107-r0001-s001-c000.xtc2.inprogress \
  /tmp/ued-run1-gainbits-new-review
```

Use a new output directory. If transfer finalizes, use its finalized filename;
additional complete frames may then be available. The reviewer hashes the
complete prefix, validates detector identity and Configure against the input,
and retains first-frame counts separately.

- [review.json](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run1-gainbits/review.json): checksums, Configure checks, counts, rate and scope.
- [frames.csv](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run1-gainbits/frames.csv): every analyzed frame's timestamp and mismatch counts.
- [after-first-mismatches.csv](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run1-gainbits/after-first-mismatches.csv): all 111 coordinates,
  occurrence counts and known/historical membership.
- [counts.npz](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run1-gainbits/counts.npz): full-coordinate occurrence arrays and input masks.

Two independent executions of the reviewer produced identical summaries and
coordinate CSV hashes. A separate set comparison confirmed exact equality to
both historical coordinate sets. The full DAQ hardware was not accessed by
the review.
