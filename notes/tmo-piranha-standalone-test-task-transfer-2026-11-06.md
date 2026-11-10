# Task transfer: Can issue #205 be reproduced without the full TMO DAQ?

## Objective

Investigate [slac-lcls/lcls2 issue #205](https://github.com/slac-lcls/lcls2/issues/205), **“piranha4 DRP aborts at Alloc: GCP reply timeout (`await: prompt not seen`) — TMO and RIX,”** and determine whether we can test or reproduce the failure without launching the whole TMO DAQ.

This transfer is for analysis and test design only. Do not run the proposed test yet.

## Safety and scope

Use the `daq-troubleshoot` and `ssh-connections` skills.

TMO is a production hutch. Perform read-only inspection only:

- Do not start or stop the DAQ, DRPs, Rogue applications, or camera processes.
- Do not submit or cancel Slurm jobs.
- Do not send UART commands to a Piranha camera.
- Do not open `/dev/datadev_*`, a PGP device, or any other hardware endpoint that could contend with the DAQ.
- Do not power-cycle, reset, configure, or otherwise change a camera, KCU/PGP endpoint, or timing system.
- Do not edit the production configuration or source checkout.
- Ask Mona before any hardware-touching or state-changing test.

You may inspect source, Git history, existing tests/scripts, configuration files, process command lines, and historical logs. If production access is useful, use read-only commands only.

## Issue context

The failure is intermittent during `Alloc`. `piranha4_connectionInfo()` clears the UART receive state, sends `GCP`, and calls `uart._rx._await()`. `_await()` polls for up to roughly five seconds and raises if it does not receive a non-empty response ending in the expected `USER>` prompt. The exception aborts the Piranha DRP and DAQ allocation fails.

Important observations from issue #205:

- TMO and RIX both show the same failure on different hosts and camera instances.
- Most failures have `len 0, resp: []`; a few RIX cases received incomplete or oddly framed replies.
- The failure spans multiple releases and predates the current TMO release.
- A relaunch normally succeeds, but that is recovery rather than a fix.
- The issue estimates an occurrence rate of roughly 1–6% of Piranha launches, so a meaningful reproduction test may need on the order of 100 cycles.
- The failure is specifically in the Piranha connection-information path during allocation; determine whether that path can be exercised independently of control, TEB, MEB, recording, and the rest of the DAQ.

Relevant source identified by the issue:

- `psdaq/psdaq/configdb/piranha4_config.py`
- `piranha4_connectionInfo()`
- the UART receive `_check()` and `_await()` implementation

Do not assume the issue summary is complete. Read the current issue and verify conclusions against the source actually available on SDF.

## Investigation

### 1. Trace the exact failing path

In the SDF `lcls2` checkout and, if useful, the currently deployed TMO release, identify:

- the current signature and full call chain for `piranha4_connectionInfo()`;
- how the Piranha DRP invokes it during `Alloc`;
- which objects, configuration values, environment variables, device files, Rogue modules, and hardware links must exist before the `GCP` command is sent;
- whether `connectionInfo()` itself changes camera or gateway state beyond the observed UART request;
- what “cold start” means for this code path: a new DRP process, a newly constructed Rogue tree, a camera reset/power cycle, a DAQ allocation after deallocation, or some combination;
- whether repeated calls in one process faithfully represent repeated cold launches.

Cite exact file paths, symbols, and line numbers or commit hashes.

### 2. Look for existing narrow test mechanisms

Search source, tests, debug tools, deployment scripts, and Git history for:

- Piranha4 standalone programs or devGui launchers;
- scripts that construct the Piranha Rogue/UART objects without a full DAQ;
- tests or utilities that call `piranha4_connectionInfo()` directly;
- prior automated reproduction attempts, including the January 2026 `SEM`/sleep workaround mentioned in the issue;
- a minimal DRP mode that performs connection information or allocation without starting control, TEB, MEB, or recording;
- mock/fake UART tests that could validate retry and prompt parsing separately from reproducing the real hardware silence.

Record whether each candidate is offline-only, hardware-in-the-loop, or equivalent to the real allocation path.

### 3. Compare possible test levels

Evaluate at least these options:

1. **Offline unit test with a fake UART** — useful for retry, timeout, raw-byte logging, and prompt-framing behavior, but probably unable to reproduce camera readiness or transport silence.
2. **Direct hardware-in-the-loop call** — construct only the required Piranha/Rogue objects and invoke the same connection-information path against one TMO camera.
3. **Minimal Piranha DRP launch** — run only the detector process or its connection-info phase, without the rest of the DAQ.
4. **Full DAQ Alloc/Dealloc loop** — the current reference method.

For each option state:

- how faithfully it reproduces the failing cold-start state and code path;
- exact prerequisites and proposed invocation, or precise pseudocode if a safe command cannot yet be given;
- whether it touches the camera, KCU/PGP device, UART, timing, ConfigDB, Slurm, or production DAQ state;
- whether it can coexist with a running DAQ or must be limited to an unscheduled period with the DAQ fully stopped;
- whether one process can perform repeated independent trials or must be restarted each cycle;
- what evidence would count as reproducing issue #205;
- expected logs and how to preserve raw UART bytes safely.

### 4. Check TMO-specific prerequisites read-only

From existing configuration and logs, identify the current TMO Piranha aliases, host, device/lane mapping, deployed release, and the exact arguments used for the affected DRPs. Do not connect to the hardware.

If access from `sdfiana` to the `tmoopr` account is needed, use nested SSH through psbuild so the second connection uses psbuild's host-local Kerberos cache:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o GSSAPIDelegateCredentials=no \
  monarin@psbuild-rocky9-01 \
  'ssh -o BatchMode=yes -o ConnectTimeout=10 -o GSSAPIDelegateCredentials=no \
  tmoopr@tmo-daq "<read-only command>"'
```

Do not renew credentials interactively. If access fails, document the concrete blocker and continue with the source and issue evidence.

## Required conclusion

Give one of these verdicts and explain it:

- **Yes:** the real failure can be tested without the full DAQ, using a narrow path that preserves the relevant cold-start and UART behavior.
- **Partly:** a narrow hardware test can exercise the suspected path, but it differs from real DAQ allocation in an important way.
- **No:** the full allocation environment is required to create the relevant state.
- **Unknown:** specific missing evidence prevents a responsible conclusion.

Recommend the smallest faithful test. Be explicit if “without the whole DAQ” still requires exclusive access to a Piranha camera or its PGP/UART endpoint. Do not equate “standalone” with “safe while the DAQ is running.”

## Deliverable

Write a concise findings report to:

`/sdf/home/m/monarin/lcls2/notes/tmo-piranha-standalone-test-findings-2026-11-06.md`

The report must contain:

- verdict and confidence;
- verified failing call chain with source citations;
- table comparing the four test levels;
- recommended smallest faithful test, with proposed commands or pseudocode;
- production interference and safety requirements;
- whether process restart, camera reset, or DAQ cold start is needed per trial;
- gaps that require confirmation from a detector expert;
- explicit confirmation that no production state was changed.

Do not implement a fix or execute a reproduction test during this task.
