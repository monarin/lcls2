# Stage 1 contract

Read-only preflight of xpp101605526 run 311 for prjdatpsana run 3 (2,400
events) and run 4 (400,000 events), each with one step. User explicitly requests
the implementation/Claude review loop. Branch: work/synthetic-mfx-scan; base
744470e3a. At most three review rounds. No core psana changes.

Freeze seven source SMD/configuration inventories and selected Jungfrau
calibration payloads privately, audit all source JF headers/shapes/damage,
inspect representative raw/calibrated distributions, specify selected streams,
event mapping/timestamp seams and exact sizes, and check destination collisions,
capacity/quota availability. Save compact reproducible evidence and review it.

Only HTTP GET and source filesystem reads. Local scripts/reports and private
snapshot files are permitted. No synthetic XTC generation, calibration writes,
public data publication, permission changes, file staging requests or external
human messages. Private raw/calibration payloads stay out of Git/public results.
Sharing authorization is a later deployment prerequisite, not a blocker for
this read-only stage. Filesystem free space is not project-quota proof.
