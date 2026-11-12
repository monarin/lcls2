# PR preparation validation

Starting commit: ea79549daa3b84c1357db693e7f37e86b737455d. Runtime files match
the prior reviewed snapshot f967cd725; current changes touch only the fixture,
README, changelog, and this review record.

Fresh initial suite: 58 tests, 20 failures and 1 error in 144.525 seconds.
Preserved in initial-tests.txt.gz. A direct fake warm run failed waiting for
ready.json before either child published any status, with both process logs
empty. Exact sources copied to /tmp reproduced the failure. Import profiling
(import-timing.txt) measured ~2.24 seconds before argument parsing, longer than
several fake-test startup deadlines. This is a fixture startup limit; the
production initialization budget is unchanged at 60 seconds by default.

Repair: fake controllers allow a separate 15-second parent readiness wait and
20-second test observation wait. All worker, allocation, hold and cleanup
budgets remain unchanged. The target-deadline assertion measures from the
actual failure detection rather than including interpreter imports.

Corrected full suite: 58 tests passed in 162.249 seconds, zero skips, on pinned
Python 3.9.21. Includes the actual software-only PyRogue/SimpleClient proof,
parent loss, native GIL-retaining cleanup, both camera orders, immutable capture,
release and expiry. See round-01/tests.txt. Hold and rehearsal dry-runs both
report hardware_opened=false; see dry-runs.json. No camera or DAQ was accessed.

Fetched upstream master at 03623f2be58f192f6b8ec75f2aa894a7aada6eff. Piranha
config, barrier, pilot and workflow paths are unchanged from the branch base
25e6f4d65. The requested draft targets monarin/lcls2 master, which is the
branch's existing base; no unrelated upstream changes are added to this PR.

The historical deployment-source snapshots retain original trailing whitespace;
scoped diff checking excludes only those two immutable snapshots. Current edits
pass git diff --check. Hardware hold rehearsal and GUI/SSH attachment remain
unverified. HOLD-IMPL-007 remains the accepted optional labeling follow-up.

The original failed log is stored as lossless gzip to preserve captured trailing
whitespace through formatting hooks. Decode it with gzip -dc.
