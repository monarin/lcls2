# UED run 5: post-Configure targeted FH repair removes extra FPs

> Historical September 2026 record. For current repeat-test entry points and archived evidence, see [the repeat-test guide](../../epixquad/README.md). Process IDs, running-state statements and launch paths below describe that session.

After normal TEST Configure, overwriting only the 44 saved FP addresses to FH
produced 1,084 complete, undamaged frames with zero FPs and zero FNs outside
the 352-pixel known-bad baseline. All 1,083 frames assessed after the first have
this pattern, and the first frame is also clean outside known-bad. The usual
68 known-bad FNs remain. Image timestamps span 3.007479 seconds at 360.102242 Hz.

The test sequence was:

1. Normal TEST Configure, starting from unallocated. Direct ASIC reads then
   showed 43 of the 44 saved targets at code 8 (FL), with trbit 1. Those 43
   coordinates exactly match run 3's FP set. The intermittent additional pixel
   was already at code 12 (FH).
2. While Configured, disable trigger/readout gates, write code 12 once to each
   of the 44 targets, and verify two immediate reads plus two reads after the
   ASIC readout-ready cleanup. All returned 12, with unchanged trbits. Restore
   and verify all gates; remain Configured.
3. BeginRun/BeginStep/Enable without another Configure. Record run 5 for three
   seconds, then Disable/EndStep/EndRun, ending in Configured state.
4. Read the same 44 addresses twice after the run. All still returned 12 with
   trbit 1; restore all gates and ASIC access settings.

This links the event FPs to the incorrect retained pixel settings after the
full Configure path: the targeted repair corrected both hardware readback and
recorded event gain bits. It does not yet identify the particular write/order
operation within Configure that creates the wrong settings. This repair is a
diagnostic intervention, not a permanent fix; a future Configure rewrites the
map and may recreate the problem.

DAQ is left **Configured**, with recording enabled and the 44 hardware pixels
at FH. No ConfigDB changes were made. The unchanged intended Configure map
matches expand2 exactly (22,169 FL pixels, codes 8/12, all trbits 1), so the
repair brings hardware into agreement with the recorded configuration.

## Evidence

- [Full-run review](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run5-gainbits/review.json), [per-frame counts](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run5-gainbits/frames.csv),
  [remaining known-bad coordinates](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run5-gainbits/after-first-mismatches.csv), and [counts](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/run5-gainbits/counts.npz).
- [Test summary](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/post-configure-overwrite/summary.json).
- [Post-Configure repair CSV](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/post-configure-overwrite/repair/pixels.csv),
  [full repair record](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/post-configure-overwrite/repair/readback.json).
- [Post-run ASIC CSV](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/post-configure-overwrite/after-run/pixels.csv),
  [full post-run record](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/post-configure-overwrite/after-run/readback.json).
- [DAQ run record](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/post-configure-overwrite/run.json) and
  [DRP transitions](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/post-configure-overwrite/drp-transitions.log).

The finalized detector stream is
`/sdf/data/lcls/drpsrcf/ffb/ued/uedc00107/xtc/uedc00107-r0005-s001-c000.xtc2`.
All 1,188,121,052 bytes were parsed through EndRun, with no partial tail or
damaged/incomplete image events. This is not an upstream event-loss audit.
The existing read-only reviewer and analysis environment used for run 4 were
used again, and output hashes were checked. The diagnostic capture hook is
one-shot and was not rearmed: no fresh paired DMA/pre-write sample is claimed
for run 5.

The initial Configure status probe timed out after five seconds while the
controller was busy; Configure itself completed successfully. The saved
configured-status.json verifies this. The request was not repeated. The helper
now resets timed-out REQ sockets and retries status during transitions. The
five offline reader tests passed, including explicit acceptance of already-FH
targets; default overwrite mode still requires code 8 baseline values.
