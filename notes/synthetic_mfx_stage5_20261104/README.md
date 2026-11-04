# Stage 5: full synthetic MFX run 2

In progress: 30 steps × 1,000 events. Exact planned XTC+SMD size:
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
core MPI test suites are not part of this fixture-generation task. Follow the
review/status records; no full generation/publication result is claimed yet.

## Review follow-up

Round 1 passed the generator/validator at a644240e0. The full write has started
with those exact unchanged files. Source calibrated acceptance passed all 60
samples and 23 frozen selections (source_samples.json). Publication now requires
--acceptance and --reference, binds both reports to the reviewed acceptance code,
and records their paths/hashes in the journal. post_publish.py checks unchanged
run-1 stats, final run-2 hard links and a calibrated first-event read. These
publication/final-check changes receive a focused second review while generation
runs. The exact small DataSource stdin command is saved as
small_datasource_command.py.txt. Acceptance remains deliberately 30 x 1000.
