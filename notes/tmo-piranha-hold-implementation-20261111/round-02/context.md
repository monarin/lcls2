# Independent re-review, round 2 of 3

Review exact snapshot `f967cd725480cd09728969fa434ffcd273f44aae` at
`/tmp/tmo-piranha-hold-implementation-20261111-r2`.
Latest runtime/code commit: `a5a4c5f0407431dfea5ec12ba9c07b9ff3f1842a`.
The reviewed snapshot adds only metadata to that unchanged tested code.
Previous review: `36b48dbabaeb00514fcfa091f827650d28b76bad`.
Original task base: `6abb2627ac378e93a92e73c4049c151b52d766b0`.

Mona requested local implementation of the November 10 staged Piranha hold plan
and an independent review loop. No hardware, GUI, SSH/network-route rehearsal,
DAQ operations, production edits or remote pushes are in scope. Use only file
reading tools; no commands, edits, MCP, external clients or delegation, and no
nested review loop. Do not claim to have executed tests.

Start with these files under `notes/tmo-piranha-hold-implementation-20261111`:

- `round-01/review.json`: the prior review and IDs HOLD-IMPL-001 through 006.
- `round-01/response.md`: accepted fixes; nothing is disputed/deferred.
- `round-02/changes.diff`: generated zero-context diff since the previous review.
- `round-01/after-fixes-tests.txt`: 58 tests passed in 44.192 seconds, zero skips.
- `validation.md`, `status.json`, and `software-proof.json`: evidence/limitations.

Review the fixes and potential regressions in the complete current runtime/test
files under `psdaq/psdaq/debugtools/piranha4_pilot`. The original acceptance
contract is `scope.md` and `notes/tmo-piranha-hold-plan-20261110.md`.

Changes: parent SIGINT/SIGTERM in an acknowledged hold now requests cooperative
parent_signal release, allowing the existing cleanup grace before forced kill;
worker signal cancellation does not create a camera failure. Cleanup failure
after release preserves release_reason/cleanup_failed consistently in parent
hold.json and summary. Channel resets become EOF. The real InspectionRecorder
and endpoint are exercised inside a worker subprocess using fake software nodes,
with identical camera calls throughout the hold and only shutdown/stop on
release. The README explains advertised loopback versus physical host.
Status records the exact tested runtime SHA; review_sha is the LAST completed
review until the new receipt exists. The metadata-only commit is deliberate:
an implementation commit cannot record its own SHA inside itself.

Verify the new assertions for parent signals, direct successful-worker signal
cancellation, operator/timeout release followed by cleanup hang, genuine AF_UNIX
reset with unread data, real recorder node/file updates and no added UART calls.
All 58 tests (17 baseline, 20 inspection, 21 evidence/client) passed against the
latest code. Real pinned software-only root/client integration is included in
that run; no skips. Hooks and diff check pass. Earlier three repeated 16-test
runs were at the previous code commit, not additional proof of these repairs.

Pinned source files outside the detached worktree were denied to the first
reviewer; do not represent them as read. Real cached-client behavior is supported
by the supplied software-only proof. It disables the fixture heartbeat and uses
an available test port triplet, not production gateway startup. Hardware/GUI/SSH
proof remains deliberately out of scope. Prior plan approval is not code approval.

Use the same finding IDs when confirming or reopening prior issues. Identify any
regressions or still-missing required local evidence. Return the required JSON
with reviewed_commit equal to f967cd725480cd09728969fa434ffcd273f44aae, clear
verdict, checks, limitations and any remaining actionable findings.
