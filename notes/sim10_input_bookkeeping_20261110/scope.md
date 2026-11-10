# Scope
Reduce measured Sim10 input bookkeeping cost without changing physical reads,
512 KiB target, batch100, ownership, small-stream credits, or execution overlap.
Base: 1805e1d9522a4e45aa611a59d83a4cf051678104 (reviewed trace results).
Implement batch group submission, constant-time busy-slot checks, and a queue of
closed input windows whose last planned/live reference has been released.
Keep CUDA dependency checks, delayed consumers, failure quarantine and retryable
cleanup intact. Reader-internal pending-list scans are audited but are not part
of the first change unless required by evidence. No read coalescing changes.

Validate CPU ownership/failure/controller tests plus new adversarial candidate
and partial-batch tests. Run GPU ownership/transition tests before a matched
Perlmutter 10k old/new x 1/2-stream comparison, same frozen native runtime and
KvikIO controls. Isolate Python changes; do not overwrite frozen install-B.
Keep 50k reads / 33582000000 bytes / 100 callbacks invariant and owner-scoped
zero local-page residency gate. Network/backend cache state remains unverified.
Use timing runs to confirm scan removal; no timing thresholds in pytest.
Independent Claude review, up to three rounds; save exact commits and receipts.
