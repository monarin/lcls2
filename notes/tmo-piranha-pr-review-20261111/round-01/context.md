# Independent PR readiness review

Reviewed SHA: 820f992408c9232839273ccdd5e9121700785e96.
Detached snapshot: /tmp/piranha-pr-review-20261111-r1.
New changes base: ea79549daa3b84c1357db693e7f37e86b737455d.
Implementation base: 6abb2627ac378e93a92e73c4049c151b52d766b0.
Target: draft PR in monarin/lcls2, features/piranha-routine-check -> master.

Read notes/tmo-piranha-pr-review-20261111/{scope.md,validation.md,pr-body.md},
the staged design, and prior implementation review at
notes/tmo-piranha-hold-implementation-20261111/round-02/review.json.
Inspect current runtime files and tests in psdaq/psdaq/debugtools/piranha4_pilot
and the new fixture-only diff from ea79549da. Runtime files are unchanged from
the prior passing review at f967cd725; test_inspection.py, README and changelog
are the only new source/doc edits outside review records.

Fresh suite failed with 20 failures/1 error out of 58; preserved losslessly in
initial-tests.txt.gz. A preserved diagnostic in /tmp/piranha-pr-diagnostic-success-20261111
has empty worker logs, no ready/status files and a startup timeout. Byte-identical
local-file sources also failed before startup. Import profiling showed ~2.24s.
New fixture permits 15s parent ready wait and 20s observation; actual allocation,
hold, worker watchdog and cleanup budgets are unchanged. The target-deadline
assertion now uses actual failure detected_monotonic, excluding interpreter time.
Corrected complete suite: 58 pass, zero skips, 162.249s; round-01/tests.txt.
Both dry-runs: hardware_opened=false. Commit hooks passed; current diff check passed.
Do not conceal the initial failures or treat historical hardware results as
validation of the new hold implementation.

Assess fixture repair correctness (does it mask a runtime/hold regression?),
current lifecycle and evidence risks, previous accepted deferred HOLD-IMPL-007,
operator docs and PR claims. Identify blockers to opening this *draft* PR versus
future hardware acceptance; do not require a real hardware run for this scope.
Do not recursively launch a review loop or delegate. Read-only tools/local git
inspection only, no edits, network, services, tests, GUI or hardware access.
Return JSON per existing review schema (reviewed_commit, verdict, summary,
checks, limitations, findings); stable finding IDs PR-001 etc; severity/blocking,
file/line/evidence/requested_change. Fifteen-minute maximum. Coordinator records
responses and supplies a fresh detached snapshot if fixes require re-review.
