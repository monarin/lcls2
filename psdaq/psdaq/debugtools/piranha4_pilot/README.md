# TMO Piranha allocation pilot

Prepared for issue 205. **Cold/warm hardware smoke tests passed on November 6 and 9, 2026; the intermittent timeout has not been reproduced.**

See the [routine-check work overview](../../../../notes/tmo-piranha-routine-check/README.md)
for historical results and review history. The opt-in hold implementation is
under local verification; it has not been rehearsed on hardware. The earlier
successful hardware trials validate only the original, non-hold pilot.
This standalone driver invokes the deployed `piranha4_init`, `init_feb`,
`connectionInfo` and `connectionShutdown` functions. It does not start control,
TEB, MEB, Configure, acquisition or recording. It does open the real board,
send startup/diagnostic UART commands, and change timing/trigger state.

## Before a hardware trial

Use an operator-approved test window and authorized execution account on
`drp-srcf-cmp013`. The **entire `/dev/datadev_0` board** must be exclusively
available: stop the DAQ processes holding it, plus devGui, other Rogue/UART
clients and any other board users. Dealloc alone is insufficient. Keep the
normal external timing connection available. Confirm site policy permits a
direct Python process on this node, or arrange an operator-managed allocation.
The harness does not stop jobs, obtain an allocation or reset hardware.

The execution flags attest that you established exclusivity. Preflight checks
visible `/proc` device holders and barrier listeners; it cannot see all other
users' descriptors and therefore cannot prove the board is free. Its advisory
lock excludes concurrent instances of this harness only. Do not attach clients
during ordinary trials. Inspection clients require an acknowledged hold and
the procedure below; generic clients are write-capable, not enforced spy mode.

The driver is pinned to the observed November 3 release and source hashes.
If the deployment changed, stop and review the new baseline; do not bypass
the checks. Default mapping: FZP lane 0, ATM IP1 lane 1, channel 0, 186M,
no XPM-mini. No camera power cycle or extra UART probes are used.

## First pilot: one FZP cold trial

The following commands are for an operator terminal **on `drp-srcf-cmp013`**
after the conditions above are satisfied. The script can be read directly
from Mona's SDF checkout; no installation or production source edit is needed.
Use the copied file path below once preparation is complete. If it is not
readable from the authorized account, copy only this pilot directory to an
operator-approved test directory, not into the production installation.

```bash
source /sdf/group/lcls/ds/ana/sw/conda2/rel/tmo/lcls2_110326_disable000/setup_env_daq.sh
PIRANHA_PILOT=/sdf/home/m/monarin/lcls2_worktree/piranha-routine-check/psdaq/psdaq/debugtools/piranha4_pilot/pilot.py

# Preview only: no hardware imports, children or device opens.
python "$PIRANHA_PILOT" --cameras fzp --mode cold --trials 1

# Actual hardware trial, only with exclusive board access established.
python "$PIRANHA_PILOT" --cameras fzp --mode cold --trials 1 \
  --execute --daq-stopped \
  --output "$HOME/piranha-pilot-fzp-cold-$(date +%Y%m%d-%H%M%S)"
```

This normally takes about 45 seconds plus initialization/cleanup. The default
40-second delay begins after all roots report initialized; it approximates the
observed launch-to-Alloc delay but is not an exact replay. Initialization and
allocation each have a 60-second watchdog. Graceful cleanup has 10 seconds;
a remaining known test PID is then sent SIGKILL and the run fails. Post-kill
reaping is bounded to two additional seconds for the group. If a process remains
unkillable, `unreaped_pids` identifies it: stop and have the operator handle the
stuck device/process before any further test or DAQ restart. No new trial
starts after a reported failure. Ctrl-C stops the test children and saves the
available evidence; an uninterruptible hardware call may require forced exit.

After a successful first trial, change `--cameras fzp` to `--cameras atm` for
one ATM cold trial. Use a new output directory for every invocation. Existing
directories are never overwritten.

## Paired cold and warm comparison

After both smoke trials and review of their logs, the small paired pilot is:

```bash
python "$PIRANHA_PILOT" --cameras both --mode cold --trials 3 \
  --execute --daq-stopped \
  --output "$HOME/piranha-pilot-pair-cold-$(date +%Y%m%d-%H%M%S)"

python "$PIRANHA_PILOT" --cameras both --mode warm --trials 3 \
  --execute --daq-stopped \
  --output "$HOME/piranha-pilot-pair-warm-$(date +%Y%m%d-%H%M%S)"
```

Inspect the cold result before starting warm; do not continue automatically
after a failure. Both modes use separate Python interpreters for the cameras
and the identical ordered allocation dictionary (FZP supervisor, ATM worker).

| Mode | Process/root lifecycle | Delays |
|---|---|---|
| Cold | New process and Rogue root per camera per trial; replay startup commands | 40 seconds after each group initializes; 2 seconds between groups |
| Warm | Reuse both processes/roots; `connectionShutdown()` after every allocation | Initial 40-second settle only; 2 seconds between subsequent allocations |

Cold refers to the processes, **not** a camera reset or power cycle. Warm mode
contains no Configure/acquisition between allocations. `--settle-seconds` and
`--interval` are explicit experiment parameters; use the same values in compared
campaigns and report the actual recorded delays. Subsequent warm trials naturally
have a longer time since root initialization. Do not equate either mode with a
validated reproduction of the intermittent camera fault. Larger trials (maximum
300) require an appropriate test window; three clean trials are only a smoke test.

## Results and recovery

Exit code 0 and `summary.json` with `ok: true` mean every requested call returned
and child cleanup completed, except rehearsal explicitly ends after its selected
trial. Any nonzero exit or `ok: false` means stop and inspect:

- `manifest.json` / `summary.json`: parameters, process groups, PIDs, trial results,
  errors, forced kills and unreaped PIDs. No authentication variables are dumped.
- `process-group-*/{fzp,atm}/provenance.json`: imported module paths/hashes, Python,
  Rogue version and an environment allowlist.
- `ready.json`, `result-NNNN.json`, `exit.json`: identity, delays, returned camera
  information and cleanup status.
- `failure.json`: phase, full traceback and `target_timeout`. Only an exception
  beginning `*** await: prompt not seen:` at the allocation GCP wait is classified
  as the target. Entry/startup, CPA, VT, VV, init and cleanup failures are separate.
- `uart-NNNN.json`, `uart-failure.json`, `uart-final.json`: timestamped original frame bytes in hex,
  TX command markers, wait/clear markers, current line and parsed responses.
  UART character bytes occupy payload offsets 0, 4, 8, ... . A missing parsed
  response does not prove zero received bytes. Ring overflow halts the run;
  the failure snapshot is saved before potentially hanging cleanup. A failure
  while writing capture is recorded as `capture_error`; cleanup still runs.
  Abrupt crashes or hangs before an exception can lose the unflushed segment.
- `process.log`: unbuffered original stdout/stderr, including assigned PyRogue ZMQ server
  ports and firmware/version warnings. Record these listeners and verify they
  disappear at teardown; the harness does not independently certify socket or
  device release after process exit.

Capture hooks observe the existing RX/serial TX methods in the test process;
they do not change waits, parsing, commands, retries or installed source. They
add callback work and can change timing. Initial camera readiness, C++ image
DMA setup and production scheduling remain fidelity gaps. An observed timeout
shows this path failed; raw evidence is needed to interpret the cause.

After **every** run, verify no test processes/listeners/device opens remain.
`root.stop()` does not restore camera/timing settings. Have the operator restore
normal DAQ configuration and verify readiness before returning to operations.
Do not auto-reset or automatically repeat a failed trial.

## Opt-in inspection contract

Use Python 3.9+ and copy the **whole pilot directory**, not just `pilot.py`.
Manifests record hashes of all local runtime modules as well as the existing
deployed-source provenance. No hold mode changes the production UART commands,
waits, parser, polling, server exposure or host/source/exclusivity gates.

These examples are **dry-runs** and do not open hardware:

```bash
python "$PIRANHA_PILOT" --cameras both --mode warm --trials 3 \
  --hold-on-error --hold-seconds 900
python "$PIRANHA_PILOT" --cameras both --trials 1 \
  --inspect-after-trial 1 --hold-seconds 60
```

`--hold-on-error` and `--inspect-after-trial N` are mutually exclusive. A supplied
`--hold-seconds` requires one of them; it must be finite, positive and at most
3600 seconds (default 900). Rehearsal requires `1 <= N <= trials`. It holds only
after every camera succeeds, labels the hold `rehearsal`, and ends the campaign
on release. A genuine error during rehearsal remains a failure and cleans up;
it is never disguised as a successful rehearsal or a synthetic reproduction.

| Event | Enabled-mode behavior |
|---|---|
| All allocations succeed | Parent authorizes deallocation; await results before another trial |
| Exact allocation GCP prompt timeout after init | Preserve failure, retain every selected root, cancel later trials |
| Successful peer finishes early | Park before shutdown, including final cold/warm trial |
| Entry/CPA/VT/VV/init error, loss of original capture | Fail and clean up; no ready inspection |
| Missing/dead/hung peer, missing hold acknowledgment | Inspection unavailable; bounded group cleanup |
| Late ring overflow | Retain original; expose late losses and overwritten intervals |
| Snapshot/client diagnostics publication I/O failure | Inspection unavailable; bounded cleanup |
| Invalid/stale release | Reject/record it; never release another campaign |
| Operator release, hold expiry, parent signal/loss | End inspection and clean up; never retry allocation |

Each worker moves through `initializing`, `ready`, `allocating`,
`allocation_done` or `target_failure`, `held`, `releasing`, then `released`
or `cleanup_failed`. Errors can enter `inspection_unavailable`. Per-camera
`hold.json` has camera/lane/PID, campaign/trial, endpoint, timestamps and release
reason. Parent `hold.json` is `held` **only after every worker acknowledges**
being parked before shutdown. A peer inside `connectionInfo()` may complete its
remaining GCP/VT/VV operations first: this is live state, not a frozen snapshot.
Enabled coordination overhead is recorded in `coordinated_seconds`.

The first detected target sets a shared monotonic deadline; hold setup cannot
extend it. Both parent and workers enforce it independently of ordinary trial
timers. The parent retains the board lock. Inherited, run/trial-validated socket
channels carry control decisions; old result/release files cannot authorize a
new allocation. Parent-control EOF releases responsive orphaned workers.
An independent watchdog interpreter bounds worker cleanup even if native code
holds the Python GIL. Kernel-uninterruptible I/O can still prevent actual exit;
missing `exit.json` or unreaped/forced-kill records mean cleanup is unverified.

## Inspect and release an acknowledged hold

After implementation review, a new authorized exclusive-board window is needed
for the first hardware rehearsal. It is **not authorized by these instructions**.
The parent prints the run's `hold.json`, both existing server endpoints, and an
exact release command. Run release on the same host, in another terminal:

```bash
python "$PIRANHA_PILOT" --release /absolute/run/directory --campaign-id TOKEN_FROM_HOLD_JSON
```

This writes an atomic run/trial-scoped request; it does not claim cleanup has
finished. Wait for final `hold.json`, `summary.json`, and every `exit.json`.
An original target timeout remains nonzero even if delayed RX arrives or later
inspection suggests recovery. GUI disconnection does not release or extend the
hold; the bounded deadline and file evidence are independent of the client.

Use the matching pinned PyRogue environment and the printed cached-only Python
client command. It reads only the software diagnostic variable, never `get()`.
The printed GUI command attaches to the **existing** root. Do not launch
CameraLink `devGui.py`: it constructs a second hardware root and has no `--spy`
option. For SSH routing, forward all three advertised ports without changing
the server's bind address, and use matching local ports for base/base+1/base+2.

Generic GUI/VirtualClient tools can write. Initially avoid ReadAll, register
refresh, polling changes, GCP/VT/VV, UART clear/escape, reset and any hardware
read. `SimpleClient.value()` reads cached values; `get()` requests a read.
Polling is disabled, so caches can be stale. Receiver diagnostics are bounded,
non-atomic software samples, not synchronized device snapshots. Late capture
uses a 4 MiB/20,000-event ring and four rotating snapshots, nominally at most
once per second plus state-boundary snapshots. Original failure files are never
rewritten by inspection. Instrumented UART TX is not a complete register-write
audit. Before any deliberate intervention, preserve a pre-intervention snapshot
and record its UTC time and exact action separately from passive observation.

## Offline checks

```bash
/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/daq_20250402_r9/bin/python -m unittest discover \
  -s /sdf/home/m/monarin/lcls2_worktree/piranha-routine-check/psdaq/psdaq/debugtools/piranha4_pilot \
  -p 'test_*.py' -v
```

Lifecycle tests use explicit fake child processes, including a native call that
retains the GIL during orphan cleanup. Evidence tests also use a real pinned
PyRogue root with software-only nodes and loopback sockets; no gateway/device is
constructed. The fixture selects an available port triplet and disables its
heartbeat; it does not validate production startup or a real GUI/network route.
Missing pinned libraries cause an explicit skip, not proof of compatibility.
The separate prior offline verification exercised the
original receiver/connectionInfo logic with fake hardware. Neither is a real
hardware trial or a proof that the production race will recur.
