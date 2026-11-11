# Independent implementation review, round 1 of 3

Base: `6abb2627ac378e93a92e73c4049c151b52d766b0`.
Review: `36b48dbabaeb00514fcfa091f827650d28b76bad`.
Detached checkout: `/tmp/tmo-piranha-hold-implementation-20261111-r1`.

Mona requested local implementation of the staged Piranha hold-on-error design
and this review loop. Review all runtime code, tests and operator docs under
`psdaq/psdaq/debugtools/piranha4_pilot`, compared with the staged plan
`notes/tmo-piranha-hold-plan-20261110.md` and the new scope/validation/evidence
under `notes/tmo-piranha-hold-implementation-20261111`. Baseline original pilot
is available in earlier source/evidence references, but do not invoke Git/shell.
Use only Read, Glob and Grep. No edits, commands, hardware, network clients,
MCP tools or delegation; do not initiate another review loop.

Scope: opt-in CLI/status controls, coordinated parent/worker decision points
retaining an early-success peer, exact target classification, immutable original
and bounded late evidence, cached-only diagnostic client, finite deadlines and
run-scoped release, parent-loss and native-cleanup watchdogs. Successful-trial
rehearsal is implemented but no hardware trial is authorized. Preserve default
lifecycle, host/source/exclusivity gates and the deployed commands/waits.
No new DAQ/hardware state or production source changes are permitted.

Acceptance: identify races, failure masking, lost evidence, unbounded waits,
unsafe cleanup assumptions, stale control/release confusion or hardware reads
caused by passive inspection. Verify all selected roots remain before shutdown
until a parent decision, last trials and warm reuse, no retry/resume after a
failure, and original failure remains nonzero. Review bounded metadata/rings,
diagnostic startup injection, pinned SimpleClient cached value semantics, and
ordinary timers versus shared inspection deadlines. Evaluate test adequacy and
documented distinctions between fake fault injection, real software-only server,
and still-unperformed hardware/GUI/SSH rehearsal.

Supplied evidence (reviewer must not claim to run tests): 54 tests passed in
36.091 seconds with zero skips using pinned DAQ Python 3.9.21; full log is
`offline-tests.txt`. This includes 17 original regressions, 16 inspection tests
(several matrices with real fake subprocesses), and 21 capture/client tests.
The pinned PyRogue/SimpleClient software-only proof is `software-proof.json`:
RO node attached before startup, cached JSON read, unchanged original snapshot,
bounded late rotation, and all three sockets closed after teardown. That test
uses explicit free loopback ports and no heartbeat; it is not a gateway startup,
GUI or hardware validation. Initial test fixture failures and repairs are
disclosed in `validation.md`. Diff check and commit hooks passed. A dry-run of
both cameras with --hold-on-error reported hardware_opened:false.

Pinned libraries are readable (only source reads allowed):
`/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/daq_20250402_r9/lib/python3.9/site-packages/pyrogue/`.
Deployed gateway/config snapshots are under
`notes/tmo-piranha-review-20261106/evidence/`. Existing November 6/9 camera results
are historical, not evidence for new hold behavior. November 10 plan approval is
not implementation approval. List missing/denied evidence as a limitation.

Return structured review with stable IDs, precise file/line evidence, severity,
blocking flag and requested changes. Do not approve missing required local
validation. Hardware validation remains explicitly out of scope, not a required
local test. There are no prior implementation findings to resolve this round.
