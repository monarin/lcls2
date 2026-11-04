# Stage 5 scope

Generate and publish prjdatpsana run 2 from the first 30,000 events of public
mfx101555026 run 10: 30 steps, 1,000 events each. User explicitly authorized
full generation and persistent implementation/review loops, including sending
code and calibration metadata to Claude. Branch: work/synthetic-mfx-scan.
No core psana changes, calibration writes, source/run-1 modifications, permission
changes, logbook registrations or messages to others.

Use existing frozen SMD/manifest and source stream files. Preserve original
correlated detector payloads and L1 timestamps, same-stream transition control
fields, source SMD vs bigdata Configure differences, all retained SlowUpdates.
Add standard scan Names/BeginStep on s000, rewrite BeginRun identity on all
streams, rebuild SMD offsets. Use a shared gap-derived schedule, no repetition.

Staging: /sdf/data/lcls/ds/prj/prjdatpsana/xtc/.synthetic-mfx-r0002-stage5-20261104
Final: /sdf/data/lcls/ds/prj/prjdatpsana/xtc, smalldata subdirectory.
Results: /sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/r0002/stage5-20261104

Acceptance: source/schedule/calibration/collision preflight; focused negative
checks and separate small writer/validator/DataSource fixture; independent
Claude review of exact committed code before full generation (max three rounds);
full byte-level source comparison, schemas, state machine, hashes, offsets and
EOF for every output record; all 30,000 events through normal calibrated serial
DataSource steps with 30 counts of 1,000, original times and scan state; raw and
calibrated first/last event in every step match 60 source samples. Publication
requires successful structural report and review, exact generator/validator
provenance, no-overwrite hard links, bigdata before SMD, fsynced journal. Final
published DataSource smoke and unchanged run-1 file stat check. Broader full-run
MPI/GPU performance work is subsequent scope; prior Stage 4 MPI acceptance passed.

Only source and local/destination files are read or written; calibration/logbook
calls are GET-only. Review must never execute writers or replay publication.
