# Local validation record

After the round-1 review repairs: **58 tests passed in 44.192 seconds, zero
skips**; see `round-01/after-fixes-tests.txt`. This includes four new regression
tests for the review findings and expanded assertions on signal/cleanup labels.

Initial implementation run: **54 tests passed in 36.091 seconds, zero skips**; see
`offline-tests.txt`. Default/hold dry-runs and `git diff --check` also pass.
Three subsequent complete lifecycle repeats at `36b48dbab` (16 tests each) also pass; see
`lifecycle-repeats.txt`. These ran against the same implementation snapshot while
the first independent reviewer read it; these predate the round-1 repairs.

Hardware execution: **none**. Runtime: pinned DAQ Python 3.9.21. System
`python3` is 3.6.8 and is not appropriate for this Python 3.9+ project.

The combined suite is run with:

```
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 /sdf/group/lcls/ds/ana/sw/conda2/inst/envs/daq_20250402_r9/bin/python -m unittest discover -s psdaq/psdaq/debugtools/piranha4_pilot -p 'test_*.py' -v
```

Coverage includes the 17 original regression cases; coordinated subprocess
success/failure ordering, early success, both timeouts, final cold and warm
trials, warm reuse, bounded hold/release, stale control/release messages,
original/late disk failures, missing acknowledgments, peer death, parent
SIGINT/SIGTERM/SIGKILL, and independent cleanup with native GIL retention.
Evidence tests cover bounded capture/loss metadata, original snapshot
immutability, four-file rotation, diagnostic publication and cached-only access.

The real pinned PyRogue/SimpleClient proof uses only a software root and loopback
ports. It verifies actual endpoint discovery, pre-start RO diagnostic attachment,
cached JSON retrieval without hardware operations, unchanged original evidence,
and all three listener ports closed after teardown. See `software-proof.json`.
The fixture disables its heartbeat and selects an available port triplet.
It is not a production gateway startup, GUI, SSH tunnel, or hardware rehearsal.

Initial software fixture attempts timed out due to occupied auto-bind ports and
startup-failure cleanup in the fixture. The fixture now selects free test ports
and cleans up failed starts; no production startup behavior was changed. A later
lifecycle run exposed a `/proc` disappearance race in the test helper; treating
ProcessLookupError as exited fixes that test observation. Final evidence is
recorded separately; earlier failed attempts are not counted as passing checks.

Implementation review is distinct from the already-passing November 10 plan
review. No new hardware, GUI attachment, network-route validation, or real fault
reproduction is claimed by these offline checks.
