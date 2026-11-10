# TMO Piranha allocation pilot

Prepared for issue 205. **Cold/warm hardware smoke tests passed on November 6 and 9, 2026; the intermittent timeout has not been reproduced.**

See the [routine-check work overview](../../../../notes/tmo-piranha-routine-check/README.md)
for results, review history and the staged hold-on-error design. Hold-on-error
is planned and reviewed, but is not implemented in this pilot.
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
lock excludes concurrent instances of this harness only. Do not connect a
GUI/VirtualClient to the PyRogue servers created by the child processes.

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
PIRANHA_PILOT=/sdf/home/m/monarin/lcls2/psdaq/psdaq/debugtools/piranha4_pilot/pilot.py

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
and child cleanup completed. Any nonzero exit or `ok: false` means stop and inspect:

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

## Offline checks

```bash
python3 -m unittest discover \
  -s /sdf/home/m/monarin/lcls2/psdaq/psdaq/debugtools/piranha4_pilot \
  -p 'test_*.py' -v
```

Tests use explicit fake child processes. They exercise orchestration, gates,
capture bounds, failure propagation and watchdog cleanup without importing
hardware libraries. The separate prior offline verification exercised the
original receiver/connectionInfo logic with fake hardware. Neither is a real
hardware trial or a proof that the production race will recur.
