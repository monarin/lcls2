# Piranha standalone reproduction: findings and setup plan

2026-11-06. **Plan only; no test or fix implemented or executed.**

## Verdict and recommendation

**Partly, with high confidence in path feasibility and moderate confidence in reproduction fidelity.** A small Python harness can invoke the actual `piranha4_init()` → `piranha4_init_feb()` → `piranha4_connectionInfo()` path without control, TEB, MEB, recording or ConfigDB queries. It still opens the real board and changes camera/gateway/timing state. Reproducing the intermittent fault is not yet demonstrated. A single camera omits the shared-board worker interaction; even paired Python processes omit C++ DMA readers, scheduling and the real rollcall-to-Alloc interval.

Use a one-camera smoke trial first, then **two fresh Python processes per paired trial**, one per selected TMO Piranha, with the same ordered allocation dictionary and unmodified barrier logic. Reserve the entire board and both cameras during an operator-approved DAQ shutdown. “Standalone” does not mean safe alongside the DAQ.

## Verified evidence

Repository citations below use SDF checkout commit `f3936ee86b35dfc2a93a579c6fb08ac83731eea7`. The latest observed Piranha launch is November 3 at 16:00:11; it does not establish what will be running during a future test. Live configuration was read through nested SSH on November 6. [Issue 205](https://github.com/slac-lcls/lcls2/issues/205) was fetched that day, last updated October 26, with no comments. Its reported launch rates combine cameras and are not independent per-camera probabilities.

1. `psdaq/drp/Piranha4.cc:245–255` constructs the detector and calls `_init()` and `_init_feb()`. `psdaq/drp/BEBDetector.cc:99–150` imports `psdaq.configdb.piranha4_config`, passes detector name, device, lane mask, optional `xpmpv`, default `186M`, verbosity; FEB arguments default to null.
2. `psdaq/psdaq/configdb/piranha4_config.py:127–182`: construct `ClinkDevRoot(dev, laneConfig={lane:'Piranha4'}, pollEn=False, initRead=True, dataDebug=False, enLclsII=True, startupMode=True, pgp4=False, enableConfig=False)`; replace the receiver and connect VC2 **before** `cl.start()`. Imports require Rogue/PyRogue, SURF, CameraLink gateway and its firmware Python dependencies, plus psdaq's import dependencies. `psdaq/psdaq/utils/enable_cameralink_gateway.py:5–16` selects firmware packages through `SUBMODULEDIR`.
3. The deployed gateway `_ClinkDevRoot.py:102–135` opens the PCIe register map and selected DMA VCs. Its `start():278–355` reads/checks firmware and links, then sends **escape, `SPF 0`, startup `GCP`**. These UART writes occur even with `enableConfig=False`; that flag gates later YAML/timing configuration (`385–421`). The constructor/start cannot be used as a read-only probe.
4. Alloc dispatch is `psdaq/psdaq/service/Collection.cc:427–440` → `psdaq/drp/PGPDetectorApp.cc:728–745` → `BEBDetector::connectionInfo()` (`BEBDetector.cc:153–185`) → `piranha4_connectionInfo(cl, alloc_json_str)` (`piranha4_config.py:190–330`). A Python exception causes `BEBDetector::_check()` to abort the DRP (`64–68`). Python alone can reproduce the exception, not that C++ abort.
5. Before the allocation `GCP`, `connectionInfo()` elects a supervisor and initializes barriers (`199–202`); the supervisor clears timing counters, may program Si570, configures timing and resets TxUser, writes TxId, then calls `StopRun()` (`210–262`, `setup_timing():94–119`). `StopRun()` disables trigger MasterEnable and enables event-builder Blowoff (gateway `232–251`). Thus the interference scope exceeds the UART or one lane. Use `xpmpv=None` to match observed commands; the XPM-mini branch is not the proposed test.
6. UART phases are distinct: initial `_await()` (`275`) consumes startup state; preserve the existing 0.1 s sleep, `_clear()`, `GCP()`, allocation `_await()` (`280–286`); then require final response line starting `CPA` (`288–294`); then query temperature `VT` and voltage `VV` (`320–328`). The target signature is the **post-allocation-GCP** `_await()` exception, not every timeout.
7. Receiver `_acceptFrame()` (`63–79`) reads payload positions `0,4,8,…`; CR appends a response except `USER>`, LF clears the current line. `_await()` (`55–61`) requires exact current `USER>` and nonempty response list within 5 s. **`len 0, resp []` does not prove zero received bytes**: control characters, partial lines or a lone prompt may leave no completed response. Raw capture is needed to distinguish transport silence from parsing.
8. `barrier.py:6–25` selects the first active participant on the same hostname as supervisor; on non-GPU hosts it groups participants without distinguishing device. `42–119` uses TCP 5561/5562 when workers exist, and has unbounded waits. `piranha4_connectionShutdown():333–334` only shuts down the barrier: Dealloc does not reconstruct the tree or replay startup commands.

Deployed release `/sdf/group/lcls/ds/ana/sw/conda2/rel/tmo/lcls2_110326_disable000`, commit `6651cbab899a71129dea1dbaa0e7d7466ee97e93`, has a byte-identical `piranha4_config.py` to the inspected checkout. Gateway dependencies come from `/sdf/group/lcls/ds/ana/sw/conda2-v4/rel/lcls2_submodules_09092026/cameralink-gateway`. Relevant dependency files and launch excerpts are preserved with source paths/hashes in [evidence](tmo-piranha-review-20261106/evidence/manifest.json).

## TMO baseline to revalidate before execution

| Setting | Observed value |
|---|---|
| Host/device | `drp-srcf-cmp013`, `/dev/datadev_0` |
| FZP | `fzp_piranha_0`, lane mask `0x1` → lane 0, channel 0 |
| ATM | `atm_piranha_ip1_0`, lane mask `0x2` → lane 1, channel 0 |
| Python environment | `daq_20250402_r9`, Rogue/PyRogue v6.1.3 |
| Timing | `186M`, no `xpmpv` or FEB overrides in observed commands |
| Production collection | `drp-srcf-mon001`, platform 0; never use this collection for the standalone harness |

November 3 launch log headers, lines 1–24, give these settings. The exact FZP command was:

```text
drp -P tmo -C drp-srcf-mon001 -M /sdf/group/lcls/ds/daq/prom/tmo -o /cds/data/drpsrcf -d /dev/datadev_0 -l 0x1 -D piranha4 -p '0' -u fzp_piranha_0 -W 6
```

ATM differs only in mask `0x2` and alias `atm_piranha_ip1_0`. These are provenance, **not proposed launch commands**. Both are selected in `~tmoopr/scripts/tmo.py:44–46`; `tmo_sc.py:163–167` also defines an IP2 camera on mask `0x4`, but it is not selected there. Confirm all board users at the actual test window. November 3 16:00 control logs show cancellation rather than an Alloc exercise; use the 11:40:34 group for successful Alloc timing/reference. That group's ATM log has startup commands at 33–34, Alloc at 116 and supervisor=False/nworkers=1 at 118. FZP's corresponding supervisor role must be recorded for the baseline run.

## Four test levels

| Level | Fidelity and proposed entry | Prerequisites, effects and coexistence | Independent trials / evidence |
|---|---|---|---|
| Offline fake UART | Isolate the actual receiver class with fake base/frame/time; replay packed payloads and exercise `_clear`, `_acceptFrame`, `_check`, `_await`. No gateway construction. | Local source and fake dependency injection; no hardware, timing, ConfigDB, Slurm or DAQ. Safe on SDF while DAQ runs. | Reset fake state each case. Validates timeout/framing/logging only; cannot reproduce hardware silence. Future retries would be a separate fix task. |
| Direct hardware Python | Exact init/init_feb/connectionInfo; synthetic Alloc body. One camera for smoke, paired processes for shared-board baseline. | Deployed environment, accessible board, working PGP/timing, exclusive board/cameras. Opens PCIe/DMA/UART; sends startup commands; writes timing and trigger state. No ConfigDB query, DAQ controller, TEB/MEB or recording required. Scheduling policy may still require a separately authorized Slurm allocation. | Restart every child and reconstruct every tree per cold-launch trial. Capture target exception and raw bytes; C++ abort is absent. Recommended smallest candidate. |
| Minimal Piranha DRP | Existing `drp` construction plus isolated collection protocol emulator sending rollcall/Alloc/Dealloc; no verified stock connection-info-only CLI found. | Adds C++ DMA/buffer/threads and collection sockets to the same hardware effects. Requires correct IDs/envelopes and a nonproduction collection/platform; check memory, port and scheduler needs. No Configure/recording or ConfigDB query required by target path. Cannot coexist on board. | Restart DRPs each trial; closest way to check C++ abort and startup scheduling if Python is inconclusive. Do not launch copied production command: by itself it waits for Alloc. |
| Full DAQ Alloc/Dealloc | Reference environment with actual controller and selected participants. | Full deployment and normal scheduler/jobs; affects production DAQ, board/timing/UART and its other detectors. Restrict to approved downtime. ConfigDB/data recording are not needed merely for Alloc; Configure/BeginRun are outside baseline. | Warm Alloc/Dealloc reuses tree and differs from fresh launch. A cold reference must relaunch processes. Preserve same-launch control, detector and TEB logs; no-error cycles are not proof of a fix. |

Existing candidates: deployed `software/scripts/devGui.py` constructs a hardware root, with timing/configuration options; it does not drive the psdaq allocation function and is not a faithful default reproduction launcher. `psdaq/psdaq/configdb/piranha4_flatfield_cal.py:353–379` demonstrates standalone receiver replacement before start, but performs calibration and changes camera state; do not use it as a test driver. No direct-call reproduction utility or fake-UART suite was found in the searched checkout. January workaround commit `7e50a0292cafe752384cf930f2bc433b7e3c8528` comments out `SEM` and adds sleep; the source comment reports unreproduced automated attempts, but does not identify the old harness or test conditions. Obtain those from the detector expert rather than assuming an existing validated test.

## Setup sequence for a later authorized implementation and run

1. **Freeze baseline and acceptance contract.** Reconfirm release, import paths, submodule hashes, lane/channel mapping, active peers, timebase, firmware versions from existing logs and launch-to-Alloc delays. Preserve environment allowlist only: interpreter, module paths, Rogue version, locale, SUBMODULEDIR and PYTHONPATH; never dump authentication variables. Current logs contain a nonfatal PCIe version-lock warning; record it without firmware updates. Match the deployment first.
2. **Build a separate harness in a test directory.** Leave installed psdaq/gateway files untouched. Parent launches children with `exec`/fresh Python interpreters, not forked preinitialized Rogue state. Child calls the existing functions once, then exits. Use one root per process because module globals (`args`, barrier) cannot represent independent roots. Install observational capture before `piranha4_init` so it sees startup bytes. Keep all command order, waits, timeout and parser semantics unchanged. No retries, extra UART probes, `SEM`, prompt repair or camera reset.
3. **Validate offline first.** Stub external dependencies before loading only the receiver logic: empty stream; prompt only; valid GCP with CPA/prompt; split prompt across frames; CR/LF variations; full reply without prompt; leading NUL/ESC/0xff; delayed prompt beyond timeout. Preserve actual 4-byte lane encoding. Check that logging serializes arbitrary bytes safely and marks capture overflow. Verify synthetic JSON elects one supervisor and expected workers without binding real barrier sockets. These are proposed checks, not tests already performed.
4. **Operational gate before any hardware open.** Mona/operator authorizes exact host, device, participants, duration and state-changing calls. Operator stops DAQ and all other clients using this board; read-only process/job/fd/socket inspection confirms exclusive access, including ports 5561/5562. A mere Dealloc is insufficient: DRPs retain device opens. Confirm scheduler policy and reserve the host/board through the operator. Do not auto-stop jobs or renew credentials. Keep external timing available as in DAQ; no XPM-mini substitution.
5. **One smoke trial per camera, sequentially.** Use exact deployed setup environment and one fresh process. Supervisor=True, nworker=0. Verify complete success through GCP, CPA, VT and VV and saved evidence. Any unexpected failure stops the campaign for review; no automatic recovery reset.
6. **Paired cold-launch campaign.** Launch both children afresh; collect PIDs and exact `socket.gethostname()` values. Once both initialization calls complete, send each the identical ordered Alloc dictionary with FZP first (after confirming reference order) and ATM second. Preserve original barrier code: supervisor=True/False and nworker=1. Record startup-to-Alloc time and interprocess launch skew, matching reference delays first; vary delay only in separately labeled experiments. Do not remove barrier delay or timing setup to speed the run. Start with a bounded 10-pair pilot; if clean and operationally approved, extend to 100 pairs in batches. Optionally extend to 300 only within the approved window. A target failure stops the baseline to preserve evidence for review.
7. **Clean up and hand back.** Every child uses `finally` to call connectionShutdown and stop its constructed root; parent reaps all children. Use an outer watchdog (initial proposal 60 s per phase, calibrated from pilot), since barrier waits can hang. On failure/hang, stop the entire pair, allow bounded graceful cleanup, then terminate only known test child PIDs if necessary; do not kill production processes or reset hardware. A forced exit or cleanup error halts the campaign. Verify test processes/device descriptors/ports are gone. `root.stop()` does not restore camera/timing settings: operator restores through normal approved DAQ configuration and verifies readiness separately.

### Single-child pseudocode (not an executable test script)

```python
# Only in the later approved exclusive hardware window, matching deployed env.
# Capture hooks are installed BEFORE init; they observe without changing logic.
root = None
try:
    root = piranha4_init(detector_name, dev='/dev/datadev_0',
                         lanemask=1 << lane, xpmpv=None,
                         timebase='186M', verbosity=0)
    piranha4_init_feb(None, None)  # observed channel 0 defaults
    # Parent supplies actual child PIDs, exact hostnames and stable order.
    # Single-camera smoke uses only this entry. Pair includes both entries.
    alloc = {'body': {'drp': {
        'fzp': {'active': True, 'proc_info': {
            'host': child_hostname, 'pid': fzp_pid, 'device': '/dev/datadev_0'}},
        # 'atm': same schema with atm_pid, present in paired trials
    }}}
    result = piranha4_connectionInfo(root, json.dumps(alloc))
    save_result(result)
except BaseException:
    save_phase_traceback_and_capture()
    raise
finally:
    # Independent guarded cleanup calls; a failure in one must not skip others.
    piranha4_connectionShutdown()
    if root is not None:
        root.stop()
    flush_capture_and_summary()
```

The harness will implement independent cleanup guards; pseudocode omits them for brevity. If init fails before returning a root, process exit/watchdog provides the final resource boundary; report partial initialization, do not count it as a GCP trial. No `body.ids` is needed for this direct Python call: it is required by the C++ collection dispatch, which is bypassed. For a minimal DRP emulator use real rollcall IDs and the full collection envelope instead.

## Capture and decision rules

Capture full original frame bytes before parser handling, not just reconstructed text: monotonic timestamp, UTC start anchor, trial/PID/lane, phase, payload size, raw hex/base64 and separately decoded bytes at stride 4. Keep a bounded per-process in-memory buffer; explicitly record drops/overflow, and flush outside the callback after completion/failure. Never block the UART callback on file writes, print decoded control bytes directly, or consume frames through a competing DMA endpoint. Instrument the receiver created by init (e.g. an observational subclass installed before init) and preserve its parser call exactly. TX markers must distinguish escape/SPF/startup GCP from allocation GCP, VT and VV. Raw capture changes timing somewhat; record this limitation and compare a later uninstrumented control only if necessary and approved.

Per trial save exception site, command/phase times, `_cur` and `_resp` through ASCII-safe repr/JSON, result model/serial/BiST, stdout/stderr, process exits, release/module hashes, delay/skew, supervisor and worker counts, cleanup status and capture completeness. Existing `_check()` can itself fail under an ASCII locale; preserve and classify that separately rather than silently patching the parser/logger baseline. Capture before the first `_clear()` and retain the whole startup/allocation sequence.

**Reproduction:** after successful initialization and initial startup wait, the allocation `GCP` is sent and its `_await()` raises `await: prompt not seen`, with phase and raw evidence. Distinguish no delivered RX payload, partial/control data, missing prompt and framing mismatch. Startup `_await`, CPA termination, VT/VV, firmware/link, barrier, logging and cleanup failures are separate outcomes, not automatic issue-205 reproductions. A Python traceback is sufficient for reproducing the failing function; claim the DRP abort only if actually observed in a later DRP test.

Report pair-level and per-camera counts separately, including invalid/incomplete trials and first failure index. With independent trials and an assumed 1% rate, 100 trials give about 63% chance of at least one failure; 300 give about 95%. The issue's rates are launch-level and correlated across cameras, so these are planning estimates, not achieved statistical power. Zero failures does not disprove the issue or validate a fix. If the paired Python campaign is inconclusive, review launch timing/previous camera state, then consider the isolated minimal-DRP stage before a full DAQ cold-launch reference.

## Remaining expert checks and task boundary

- Confirm previous automated test harness, launch delays and whether it recreated roots/processes or only repeated Alloc.
- Confirm gateway startup/stop ordering and clean termination under paired failures, shared board users, supervisor ordering and timing-link expectations. The source permits the narrow path; equivalence to the production race remains unproven.
- Decide whether prior Configure/acquisition state matters. Keep normal powered camera state for baseline: no evidence requires a camera power cycle or reset per trial. Resetting would add an untested condition and needs separate authorization.
- Confirm executable harness instrumentation, preflight and scheduler policy in a later implementation review. Exact future launch commands depend on that harness and are deliberately not invented here.

**No production state was changed.** Inspection read source, Git history, GitHub issue, existing logs and configuration only. No device was opened, UART command sent, process/job controlled, credential renewed or reproduction executed. Only local plan, evidence and review artifacts were created.
