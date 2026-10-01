Proposed title: Fix ePixQuad1kfps IsEn ordering before gain-map programming

> Historical September 2026 record. For current repeat-test entry points and archived evidence, see [the repeat-test guide](../../epixquad/README.md). Process IDs, running-state statements and launch paths below describe that session.

Full DAQ Configure enables ASIC IsEn, then applies expert ConfigDB settings
that restore the acquisition value IsEn=0 before programming the pixel map.
Reassert IsEn=True after expert settings and immediately before map writing;
retain the existing IsEn=False cleanup for acquisition.

On the production UED module, the original sequence produced 43–44 extra
low-gain pixels outside the known-bad list. ConfigDB and recorded Configure
maps were correct, but direct ASIC reads returned FL code 8 at requested FH
code-12 locations. DMA/pre-write/disk comparisons placed the mismatch before
disk writing. With only the IsEn ordering correction, run 7 recorded 1,086
complete, undamaged frames at approximately 360 Hz with zero FP/FN outside
known-bad, and all 44 prior FP addresses read FH twice after EndRun.

The change preserves the original map algorithm, write order, trigger settings
and ConfigDB contents. Offline full-Configure replay verifies programming-state
ordering, unchanged pixel commands, final cleanup and the no-map-update path.
The 68 expected known-bad FNs remain. The electrical reason IsEn=0 creates
additional row-0 FL assignments is not yet established.

Clean master-based branch: `fix/epixquad1kfps-isen-ordering`, commit `a4907da6f`,
base `a4be0f987`. A separate full build of this branch passed run 8 with control,
TEB, timing DRP and detector DRP all using that install: 1,084 complete,
undamaged frames at 360.282 Hz, zero FP/FN outside known-bad including the first
frame, and all 44 prior FP addresses reading FH twice after EndRun. No targeted
repair, pre-run ASIC readback, standalone adapter or diagnostic capture hook
was used. Diagnostic scripts, standalone-port experiments
and hardware-test artifacts are retained on the diagnostic branch and are not
part of the minimal code diff.

Full investigation: `notes/ued-full-daq-20260930/POST_MORTEM.md` on branch
`features/ued-epix10ka-rdsrv421-daq`.
