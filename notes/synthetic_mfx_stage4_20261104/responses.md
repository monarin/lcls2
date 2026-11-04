# Stage 4 review responses

Round 1 passed implementation `24c927cab7b76a18df9e48bf799df6f6651fec8b` with
five low-severity, nonblocking findings. Local serial/MPI results stand; the
collaborator own-login gate remains open.

| Finding | Disposition and evidence |
| --- | --- |
| S4-R1-01 | Accepted. README documents all nine cases' post-PASS UCX shutdown warnings, zero exits, assertion results and the fact that BTL selection does not constrain the active UCX PML. Original logs retained; an additional bounded PML diagnostic is saved in mpi_transport_probe.log. No runtime/core fix is claimed. |
| S4-R1-02 | Accepted documentation remedy for existing runs. README explicitly binds all nine cases to pilot.json SHA256 c5abc67a6825f065229f7e9f043150bd87ee4474f1484634dab87a2a4c3f42e5; archive and current file match. Recording the hash directly in MPI reports is deferred to a future driver revision, as suggested. Historical reports remain unchanged. |
| S4-R1-03 | Accepted. status.json now marks the initial one-event probe superseded by the archived full pilot comparison. |
| S4-R1-04 | Accepted documentation remedy. Review cross-checked the three step timestamps, and schedule_crosscheck.json independently compares the serial output with the Stage 1 original BeginStep plus inserted schedule. An in-driver assertion is deferred to a future revision; no rerun is necessary. |
| S4-R1-05 | Fixed. Collaborator output includes Python version/executable and file paths/hashes for psana, Dgram, Jungfrau, UtilsJungfrau; it prints these before data access so failed runs retain diagnostic context. README treats a mismatch under another build as inconclusive and requests environment/error output. collaborator_environment_probe.json exercises the collector locally without claiming collaborator acceptance. |

Only collaborator_check.py changes executable behavior in this correction. The
serial/MPI computation drivers and their inputs are unchanged; all eleven report
code bindings are verified in provenance.json. Six negative acceptance tests pass
again in unit_checks_followup.txt. The collaborator script still requires its
own-login execution before Stage 4 can be called fully complete.

Round 2 passed `0c9e57ef20bbe1d0cc4701d0325e232a11290b4d` and explicitly
closed S4-R1-01 through S4-R1-05. New nonblocking S4-R2-01 requested only the
final archive hash check: accepted and addressed in handoff_provenance.json.
The final archive script equals the reviewed Git blob; its old version is
preserved. This evidence-only action was not a code revision or another review
round. No blocking findings remain; collaborator execution remains external.
