# Piranha hold-on-error inspection: staged implementation plan

November 10, 2026. Plan only; no implementation or hardware activity in this task.

## Intended behavior and evidence

With inspection explicitly enabled, an allocation GCP timeout saves its original
UART evidence and retains the existing camera processes/Rogue roots for a bounded
inspection window. No further trials run. The operator connects to each existing
ZMQ server, inspects cached state and delayed UART responses, and then releases
the hold for normal cleanup. This preserves a live post-failure state, not a
frozen camera or an atomic hardware snapshot.

The reviewed pilot is commit `49a8aa1d5c6e8866108055da009ec8cc70e3e904`, script
SHA256 `d7c7e249753f29dccf5d713365b79758f8fb60da42fa4e336b5728d2ea10b804`.
It currently catches exceptions in `child_main()`, writes `uart-failure.json`
and `failure.json`, then unconditionally calls `connectionShutdown()` and
`root.stop()`. `wait_files()` detects failure and the parent terminates peers.
A successful peer also calls shutdown before publishing its result. Merely
adding a sleep in the exception handler cannot preserve the paired test.

The pinned `piranha4_config.py` initializes with `pollEn=False`, replaces the
UART receiver, and calls `ClinkDevRoot.start()`. The deployed gateway root
creates `pr.interfaces.ZmqServer(root=self, addr='*', port=0)`; each child's log
reports its assigned base port. The deployed CameraLink `software/scripts/devGui.py`
constructs a new root and has no `--spy` argument. Existing-server GUI/VirtualClient
attachment is the supported route to investigate here, not another device opener.

November 9's 3 paired cold plus 3 paired warm trials all passed (12 camera calls);
no real target exception has yet been captured. Those results validate the
existing test path only, not a future hold implementation or fault reproduction.
Source and result references are listed below.

## Stage 1 — Define the inspection contract and status

Add proposed options `--hold-on-error` (off by default) and `--hold-seconds 900`
(15 minutes; finite, positive, at most 3600 seconds). Reject a supplied hold
duration without a selected inspection mode. CLI dry-run displays the choices
without hardware imports/opens. Existing host, source-hash, exclusive-board,
output-directory, and advisory-lock gates remain in force throughout the hold.

Initially qualify only the existing exact target classifier: an exception
starting `*** await: prompt not seen:` in `allocation_gcp_wait`, after successful
root initialization. Init errors, entry/CPA/VT/VV failures, capture failures,
process exits, native hangs and parent watchdog expiry retain fail-and-cleanup
behavior. An unresponsive process cannot promise a usable GUI; do not describe
this feature as a general crash or deadlock debugger.

Publish atomic, run-specific status records containing campaign ID, trial,
camera/lane, PID, failure classification, state, host/base port, UTC times,
monotonic hold deadline and reason for release. The final result remains failed
and the CLI exits nonzero after a target timeout, even if inspection/recovery
later appears successful. No retry or resume of a failed campaign.

Acceptance: documented state transitions and error matrix; disabled mode preserves
existing lifecycle and failure semantics; no changing production source or UART
commands/timeouts to make a failure more likely.

## Stage 2 — Coordinate both camera processes before cleanup

In inspection-enabled mode, introduce a parent-controlled post-allocation decision
point for every selected camera. A child that succeeds saves its trial capture
and announces `allocation_done`, but waits before `connectionShutdown()` and
before the next allocation. A child with the target exception saves evidence,
announces `target_failure`, and waits before either shutdown call. Use explicit
per-trial messages with campaign/trial identity and an inherited parent-control
channel; do not infer a decision from a stale file or another trial's result.

Parent decisions:

1. All children succeed: authorize ordinary deallocation; wait for cleanup/result
   acknowledgments before next warm allocation or cold-process teardown.
2. Target failure: record failure immediately, cancel further trials and request
   hold from all initialized children. Enter `held` only after every selected
   child acknowledges that it is parked before cleanup with its root alive.
3. Missing acknowledgment by the existing allocation deadline, non-target failure,
   dead child, or failed evidence write: record inspection unavailable/incomplete
   and perform bounded cleanup of the whole group. Never wait indefinitely for
   a peer or convert a watchdog failure into successful inspection.

Do not interrupt or rewrite the deployed `connectionInfo()` internals. A peer
already inside that call may finish its current GCP/VT/VV operations before it
can acknowledge hold. Log when each camera actually parked; no claim of
simultaneous freezing. A peer that finished early must remain parked, even on
the last cold or warm trial, so its root cannot disappear ahead of a failure.
Keep the original barrier lifecycle until the parent chooses cleanup; the
control channel must be independent of the hardware/DAQ barrier.

During `held`, suspend the ordinary wait-for-next-command/allocation timers in
both parent and children and use the shared inspection deadline instead. Keep
Rogue receive/server threads running; do not use SIGSTOP or a debugger stop that
also prevents UART delivery and GUI responses. Parent retains the board lock.
No automatic UART command, clear, retry, reset or polling enable occurs on hold.

Acceptance: either camera can fail first or second; early-success, both-failure,
last-trial and warm-reuse cases retain both roots without running another trial.
Off-mode scheduling remains unchanged; enabled-mode coordination overhead is
recorded as an experiment difference.

## Stage 3 — Preserve evidence and expose useful diagnostics

Save the original failure capture and traceback before cleanup or client
attachment, once, under immutable filenames. Mark target event and hold entry
with monotonic timestamps. Retain the existing raw-frame format and phase
classification. Snapshot errors are explicit and prevent declaring a ready hold.

Keep late RX in a separate bounded capture (initial limits: 4 MiB/20,000 events
per camera, separate from the original failed segment). Use a fixed-size ring
and fixed-count rotating on-disk snapshots rather than unbounded files. Record
late-capture overflow and overwritten intervals visibly; original evidence must
never be replaced. Bound snapshot frequency and response size; do not block the
Rogue RX callback on disk writes or remote clients. Disk/capture I/O errors end
inspection and trigger cleanup. Finite hold duration also bounds total retention.

Expose software-only diagnostics for current `_cur`/`_resp`, raw RX count/last-RX
time, failure phase, hold state and capture-loss counters using read-only
LocalVariables or a read-only snapshot accessor. These fields are not currently
ordinary nodes in the Rogue tree. Bound buffer serialization and report snapshot
time/coherence limitations: buffers may change while RX continues. Do not add a
new parser lock to the original receiver or claim a fully atomic receiver/device
snapshot; any locking needed for our capture ring must remain short.

Obtain the actual ZMQ endpoint from the existing server object using the pinned
API; verify this offline instead of assigning ports such as 9099 by assumption.
Write per-camera attach instructions and all required ports in `hold.json`.
No replacement server/root/device open. No external notifications are required.

Acceptance: original snapshot is byte-for-byte unchanged after late RX, cached
inspection and release; late data and losses are distinguishable; inspector
access adds no UART TX or hardware reads; endpoint metadata matches the live
server. Instrumented TX captures help identify deliberate subsequent commands,
but are not a complete audit of all possible GUI register writes.

## Stage 4 — Attach, release, and recover predictably

Provide a small documented Python inspection client using the existing
PyRogue endpoint. Start with cached/software fields: `SimpleClient.value()`
retrieves a cached value; `get()` initiates a read. Document a GUI command such as
`python -m pyrogue gui --server=<host>:<actual-base-port>` and optional
`VirtualClient` access. Use the matching deployed environment. Validate client
compatibility on a software-only root before a hardware rehearsal.

The generic GUI/VirtualClient remains write-capable: calling it “spy mode” does
not enforce read-only behavior. The initial inspection procedure avoids ReadAll,
PollEn changes, register refreshes, GCP/VT/VV, UART clear/escape and reset commands.
Cached values may be stale because polling is disabled. A later deliberate
hardware read or command is an intervention: save a pre-intervention snapshot,
record what was done and when, and separate its evidence from passive observation.
A separate purpose-built read-only proxy/UI is outside the initial scope.

For remote access, verify that the operator's GUI can reach the advertised
endpoint through the approved network/SSH route; if tunneling is needed, forward
the ports required by the pinned client, not just the printed base port. Do not
change server exposure or assume the GUI runs on the non-graphical agent host.
GUI connection failure does not extend the hold automatically; file snapshots
remain the fallback.

Provide an explicit release action scoped to this run (validated campaign token
and directory, atomic message). It only ends inspection and cleans up, never
retries Alloc. Ctrl-C/SIGTERM to the parent also releases and cleans up. Keep
these signal handlers active during the hold instead of retaining the current
exception path's blanket ignore until exit. Both parent and children enforce
the same monotonic deadline, measured from first target detection, with no
client-driven extension. Hold setup cannot extend it.

Use parent-control EOF/parent-death detection so children leave a responsive
hold if the coordinator disappears, even if the SSH connection fails. On lost
parent, children initiate their own bounded cleanup; test this with real fake
subprocesses, not only mocked PID existence. A child-local termination watchdog
must bound cleanup even when the parent is gone. With the parent alive, preserve
the existing shared 10-second cleanup grace, SIGKILL-all-survivors before bounded
reaping, and explicit unreaped-PID failure record. Native/kernel uninterruptible
I/O remains an operational limitation; never certify release just because a
cleanup timer elapsed.

Acceptance: release, timeout, parent signal, parent death and broken GUI all have
bounded outcomes; failure remains failure. Verify known PIDs and barrier/PyRogue
listeners disappear and perform visible-holder checks after teardown. A dead
parent may prevent final parent summary: child artifacts must identify this.
No automatic DAQ restart/reset; restore normal DAQ Configure before acquisition.

## Stage 5 — Offline implementation and independent review gates

Implement in small commits: CLI/status contract; paired coordination; evidence
and client diagnostics; lifecycle hardening and operator documentation. Run
relevant tests at each step, then independent implementation review and repairs
before hardware use (up to three rounds as in the current review skill).

Required offline coverage, without hardware construction:

- Existing 17-test regression suite, default dry-run and host/source/access gates.
- Target exceptions injected in fake RX at the real wrapper's allocation wait,
  with FZP first, ATM first, both failures and an early-success peer; cold last
  trial and warm middle/last trial. Confirm no shutdown/extra trial before release.
- Parent and child ordinary timeouts would expire during a valid hold, but do not;
  the finite hold deadline still triggers bounded cleanup.
- Missing/dead/hung peer, non-target exception, failed snapshot/disk write,
  malformed/stale release, acknowledgment race and failure at the decision edge.
- Late RX after exception, ring overflow, preserved original snapshot, bounded
  diagnostics, cached reads adding no device calls and fixed disk retention.
- Release, expiration, Ctrl-C/SIGTERM, parent exit/SIGKILL and cleanup hang; no
  unintended live fake children/listeners after completion. Model unkillable
  children separately and report limitations without blocking the test runner.
- Software-only Rogue/ZMQ root test with the pinned client: discover endpoint,
  attach and retrieve the planned cached diagnostics, disconnect, release and
  verify sockets close. This supplements, not substitutes for, lifecycle fakes.

Acceptance: reviewer finds no unresolved blocking issues on the latest code
snapshot; records distinguish fake fault injection, software-only server tests
and hardware evidence. A plan review does not satisfy this implementation gate.

## Stage 6 — Hardware rehearsal, then a bounded reproduction campaign

At a later explicitly authorized hardware step, recheck that the whole board
is available that day. Retain `/dev/datadev_0`, FZP lane 0 and ATM IP1 lane 1,
186M timebase and all pinned source checks. Do not carry Monday's test-window
confirmation forward indefinitely.

First implement an explicitly selected `--inspect-after-trial N` rehearsal
mode, mutually exclusive with `--hold-on-error`, also using `--hold-seconds`.
After all cameras successfully complete the named trial, enter the same
post-allocation hold without inventing an exception. Mark this `rehearsal`, with
no target timeout and no claim of reproduction; release ends the campaign rather
than continuing trials. Validate N against the requested trial count. Keep fake
error injection confined to offline tests, never add it to the production path.

Run one successful paired trial into this rehearsal hold. Confirm both PIDs,
server endpoints, diagnostic access and late-RX logging; attach through the
existing servers without opening another hardware root. Exercise explicit
release and inspect processes, sockets, board visibility and final artifacts.
A second short rehearsal can verify automatic expiration; require both cleanups
before any longer campaign. If the real target error occurs during rehearsal,
stop and preserve its evidence with normal cleanup (rehearsal is not the
hold-on-target mode).

Then run a small paired cold/warm smoke comparison with `--hold-on-error`,
followed, only within an agreed time window/trial budget, by a larger reproduction
campaign. Stop at the first target error and retain it for inspection; no
automatic retries. The first warm allocation follows fresh root initialization;
subsequent allocations reuse it. Cold is process/root recreation, not power-cycle.

Success of this feature means the test reliably holds and releases inspectable
roots with preserved evidence. Reproducing issue 205 requires an actual
allocation GCP timeout; a rehearsal or a clean campaign is not evidence of that.
If GUI/server access fails during a real fault, preserve captures and report the
limitation without opening another board client or automatically resetting it.

## References available for implementation/review

- `psdaq/psdaq/debugtools/piranha4_pilot/{pilot.py,test_pilot.py,README.md}` at the
  reviewed pilot commit above. This branch's baseline is evidence-only follow-up
  commit `747580e8d851c10aebfc84fb0197f2dd673b3bfa`.
- `psdaq/psdaq/configdb/piranha4_config.py`: `piranha4_init`,
  `piranha4_connectionInfo`, `piranha4_connectionShutdown`; `barrier.py` lifecycle.
- Pinned release `/sdf/group/lcls/ds/ana/sw/conda2/rel/tmo/lcls2_110326_disable000`.
- Gateway source under `/sdf/group/lcls/ds/ana/sw/conda2-v4/rel/lcls2_submodules_09092026/cameralink-gateway`:
  `firmware/python/cameralink_gateway/_ClinkDevRoot.py`, `software/scripts/devGui.py`.
- PyRogue source under `/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/daq_20250402_r9/lib/python3.9/site-packages/pyrogue`:
  `__main__.py`, `interfaces/_SimpleClient.py`, `interfaces/_Virtual.py`.
- Monday evidence in the main workspace:
  `/sdf/home/m/monarin/lcls2/notes/tmo-piranha-review-20261109/results.md`,
  `verified-results.json`, `post-run-check.json` and the linked campaign directories.
