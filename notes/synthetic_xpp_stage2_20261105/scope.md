# Stage 2 scope

User authorized Stage 2 after the explanation that it includes experiment-only
calibration deployment (validity 3–4), private run-3 generation and validation.
This authorizes those specific writes; no further confirmation is required.
It does not authorize publication of private XPP event payloads or changing
permissions. Stage 3 publication and the 400k-event run-4 generation are excluded.

Source and exact mapping are frozen by Stage 1 (reviewed 786938d79, evidence
07cff96d5). Retain streams 2/3/5/6, 2400 original 1M events, one enabled step.
Create only a new owner-only results directory. Validate every copied datagram,
identity, timestamp, offset, file extent, and raw/calibrated DataSource samples.
Also exercise two repetition seams and truncation in an eight-event private
fixture using a shortened three-event source cycle; test second rollover.

Deploy exactly eight frozen types to cdb_prjdatpsana/jungfrau_000010, run 3 to 4.
No writes to source/shared DBs, detector-name mappings, existing run-1/2 records,
logbook, or final xtc/. Journal each request before sending it; refuse automatic
retries after uncertain/partial writes. Preserve frozen source IDs as provenance.
Check unchanged previous documents, run-1/2 selections and all 40 existing files.

Use Codex implementation and independent restricted Claude review, maximum three
rounds. Round 1 checks code, private generated fixtures, validation and a concrete
deployment dry run before any DB write. After approval, execute the authorized
plan once and review complete execution/acceptance evidence. Never rerun a
mutation merely for review. Branch work/synthetic-mfx-scan; no psana core changes.
