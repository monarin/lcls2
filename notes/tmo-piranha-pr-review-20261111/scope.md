# PR preparation and review

User requests the next step as a PR with the review loop. On inspection, branch
HEAD ea79549daa3b84c1357db693e7f37e86b737455d already contains stages 1–5 plus
rehearsal controls, previously reviewed in two rounds. Preserve that implementation
and prepare a draft PR in monarin/lcls2 against master; no hardware execution.

Fresh validation found 20 failures/1 error in 58 tests (initial-tests.txt.gz), mostly
fake workers timing out before ready.json. A preserved diagnostic failed before
worker code entry, with empty logs and no worker status files. Import profiling
of the exact local-copy test sources showed about 2.24 seconds to import, including
about 1.79 seconds in logging; a fresh unittest.mock import took 2.05 seconds
(1.94 seconds without site initialization). These exceed fixture timeouts of
0.8–2 seconds. Source copies on /tmp also reproduced the startup failure.

Scope includes repairing fixture startup allowance while retaining all tested
runtime protocol/hold/cleanup deadlines, adding the required changelog entry,
validating offline, independent review and any necessary scoped fixes, pushing
and opening the requested PR. Do not widen production deadlines, fabricate a
hardware pass, overwrite original review receipts or drop the initial failed run.
HOLD-IMPL-007 remains the previously accepted low-priority labeling limitation.

Use an in-session independent reviewer, as in the earlier Piranha plan loop,
with a detached commit snapshot, at most three rounds and a 15-minute reviewer
limit. Prior implementation review remains historical evidence; this loop
assesses PR readiness, the new test repair and any newly identified regressions.
