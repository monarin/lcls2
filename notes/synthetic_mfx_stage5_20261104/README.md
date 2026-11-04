# Stage 5: full synthetic MFX run 2

Published: 30 steps × 1,000 events. Exact XTC+SMD size:
1,019,012,911,780 bytes. Source retains its original event timestamps and bytes;
only run identity, standard scan payloads, inserted transitions and SMD offsets
change. Ten streams, one chunk each, 30,373 records per stream, 119 inserted
transitions. Run-2 calibration already has verified validity 1–2; no new DB writes.

The scripts are adapted copies of the reviewed Stage 3 implementation; original
Stage 3 files remain unchanged. The inherited make_pilot.py/verify_pilot.py names
refer here to the full run-2 fixture. Four threads process independent streams.
Generation hashes output while writing; validation hashes while comparing every
record with the source, avoiding separate terabyte-scale hash-only passes.
Publication checks the validated size/inode/mtime of each staged file, not another
full rehash. Staging is private (0700); this detects ordinary intervening changes
but does not protect against a hostile owner changing bytes and restoring mtime.
No such concurrent writers are expected. Complete-set publication is not atomic;
the intent/link journal makes partial outcomes explicit, with no blind retry.

preflight.json records exact size, source alignment, schedule, no disk/logbook
run-2 collision, live calibration metadata matching the verified 13 records,
and filesystem free bytes. Weka's project quota API requires admin authentication;
filesystem free space is not quota proof. Any write/quota failure leaves private
staging for inspection rather than exposing incomplete final SMD names.
The first path-only probes failed because the copied script's snapshot path was
incorrectly changed to r0002; fixed to the unchanged r0001 Stage 1 snapshot before
any fixture writes. Both failed logs are retained.

small_verification.json and small_test.log cover a separate 9-event [2,3,4]
fixture using the same writer/validator with test-only patched counts/schedule.
It is not the full fixture, cannot satisfy publication's required full counts,
and does not establish full-run acceptance. small_datasource.log checks normal
run.steps() and scan values. unit_checks.log covers negative control/header,
timestamp rollover, insufficient-gap and publication-provenance cases.
schedule_crosscheck.json compares the new schedule with the independent Stage 1
audit: first 29 boundaries match, final event-30000 boundary uses the same first
three timestamps with EndRun replacing BeginStep. Tightest adjacent output gap
is 2 ns; no event timestamps move.

acceptance.py source mode uses timestamp filtering and batch_size=1 to select
60 boundary samples. Destination mode walks the entire normal calibrated
DataSource with all detector streams, checks all event times, step counts,
BeginStep times and scan values, and matches raw/calibrated sample fingerprints
against source mode. Calibration payloads/selection IDs are checked against the
frozen manifest for all 23 detector/type selections; calibration options are
cversion=3, cmpars=None, with NaNs normalized only for fingerprinting.

The runtime pins the previously used psana build. No core files changed, so
core MPI test suites are not part of this fixture-generation task.

## Review follow-up

Round 1 passed the generator/validator at a644240e0. The full write completed
with those exact unchanged files. Source calibrated acceptance passed all 60
samples and 23 frozen selections (source_samples.json). Publication now requires
--acceptance and --reference, binds both reports to the reviewed acceptance code,
and records their paths/hashes in the journal. post_publish.py checks unchanged
run-1 stats, final run-2 hard links and a calibrated first-event read. These
publication/final-check changes received a focused second review while generation
ran. The exact small DataSource stdin command is saved as
small_datasource_command.py.txt. Acceptance remains deliberately 30 x 1000.

Round 2 also passed at a3852808ada2032a7e6893bbd74bda0a028f8662, no blockers.
Use that commit and round-02/review.json for publication. The generator,
validator, helper and acceptance files remain identical to round 1. Runtime
outcomes are recorded separately from code review below. Optional additional gate negative cases are
deferred; no executable changes follow round 2.

## Published fixture

All 20 files were published without overwrites at 2026-11-04T22:24:12Z:

```text
/sdf/data/lcls/ds/prj/prjdatpsana/xtc/prjdatpsana-r0002-s000-c000.xtc2
... through s009
/sdf/data/lcls/ds/prj/prjdatpsana/xtc/smalldata/prjdatpsana-r0002-s000-c000.smd.xtc2
... through s009
```

Use explicit disk discovery; the logbook has no registered files for this run:

```python
from psana import DataSource

ds = DataSource(exp="prjdatpsana", run=2,
                dir="/sdf/data/lcls/ds/prj/prjdatpsana/xtc", live=False)
for run in ds.runs():
    step_value = run.Detector("step_value")
    for step in run.steps():
        for evt in step.events():
            scan_index = step_value(evt)  # 0 through 29
```

Set `PS_FAKESTEP_FLAG=0`. Calibration validity is already 1–2; this stage made
no calibration writes. Original detector payloads and L1 timestamps are retained.

`verification.json` passed every record in every stream: source-byte comparison,
schemas, SMD offsets, transition state, timestamps, hashes and EOF.
`destination_acceptance.json` passed all 30,000 events through serial
`DataSource.run.steps()`, with exactly 1,000 per step, original timestamps and
scan state. All 60 first/last-per-step raw and calibrated samples match
`source_samples.json`; all 23 calibration selections match the frozen manifest.
The acceptance used `PS_SMD_N_EVENTS=1001`, deliberately crossing step boundaries.
These are full runtime checks by the coordinator; Claude reviewed the code and
available evidence before these operations completed.

`publication.jsonl` binds publication to both acceptance reports, the full
validation and reviewed commit a3852808ada2032a7e6893bbd74bda0a028f8662.
The private staging files remain hard links to the final files, so retaining
staging does not duplicate the approximately 1.019 TB of XTC/SMD data.
The 136 MiB full generation trace is stored outside Git at:

```text
/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/r0002/stage5-20261104/generation.json
```

The compact `generation_summary.json` records its hash and per-file hashes.
Its `generated_not_yet_validated` label is the historical generation checkpoint;
subsequent verification and status records establish completion.
Full-run MPI/GPU performance measurements and logbook registration remain
separate work; Stage 4 established pilot MPI acceptance and Seema's access.

`published_smoke.json` passed from the final directory: all 20 hard links,
all 20 unchanged run-1 file stats, and the first event's source-matching raw and
calibrated arrays. `completion_checks.json` verifies executable files against
the final reviewed commit. `published_acls.txt` records all final file ACLs,
including effective read access for ps-prj and prjdatpsana. A run-2 own-login
check by Seema has not been repeated.

The durable results directory above contains `final_records/`, the complete
generation trace, and both reviewed Git implementation archives. Its
`SHA256SUMS` covers those artifacts and the earlier `pre_generation_records/`
checkpoint; `results_SHA256SUMS` is the repository copy of that manifest.
Code review passed at a3852808ada2032a7e6893bbd74bda0a028f8662. The later
completion commit records evidence and documentation only.
