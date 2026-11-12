# piranha4 DRP aborts at Alloc: GCP reply timeout ('await: prompt not seen') — TMO and RIX

Fetched 2026-11-06 from https://github.com/slac-lcls/lcls2/issues/205; updated 2026-10-26T20:47:04Z; comments: 0.

> **Claude (AI reviewer)**, posted via @monarin's account.
> Analysis by Claude from the daq-agent TMO reports (10/19 and 10/26) and the raw TMO and RIX logs; project decisions are @monarin's.
> *Updated 2026-10-26: added 8 more TMO launches (09/18 – 10/25) and the RIX occurrences (06/2024 – 09/2026). The original 10/14 – 10/21 findings are folded into the tables and sections below.*

## What's happening

Piranha4 DRPs intermittently abort during **allocation** while waiting for the camera's reply to `GCP`. DAQ control then reports that the detector did not respond to `alloc`, and the launch has to be restarted.

This is **not specific to TMO or to `drp-srcf-cmp013`**. RIX Piranhas on `drp-srcf-cmp012` fail with the identical exception, and have since at least June 2024.

```text
...UartPiranha4: SendString: gcp
WARNING:root:_check found
Traceback (most recent call last):
  File ".../psdaq/configdb/piranha4_config.py", line 286, in piranha4_connectionInfo
    uart._rx._await()
  File ".../psdaq/configdb/piranha4_config.py", line 61, in _await
    raise Exception('*** await: prompt not seen: len %d, resp: %s' % (len(self._resp), self._resp))
Exception: *** await: prompt not seen: len 0, resp: []
tmo-drp[...]: <C> **** python error
srun: error: drp-srcf-cmp013: task 0: Aborted (core dumped)
```

Status: **Investigating; no verified fix.**

## Summary of occurrences

I searched every Piranha DRP log for `await: prompt not seen`.

| Hutch | Host | Period searched | Launches with the error | Process aborts | Launches using a Piranha | Rate |
|---|---|---|---|---|---|---|
| TMO | `drp-srcf-cmp013` | 2026 (09/17 – 10/26, all SDF logs) | **12** | 20 | 215 | ~5.6% |
| RIX | `drp-srcf-cmp012` | 2024 | 18 | 18 | 1411 | ~1.3% |
| RIX | `drp-srcf-cmp012` | 2025 | 105 | 133 | 3308 | ~3.2% |
| RIX | `drp-srcf-cmp012` | 2026 (01 – 10) | 9 (6 CDS + 3 SDF) | 10 | ~600 | ~1.5% |

The rate is launches with the error divided by launches that started a Piranha DRP. The two rows are not strictly comparable: TMO runs two Piranhas per launch, while RIX runs up to three (`c_piranha_0`, `fzp_piranha_0`, `q_piranha_0`).

Common features in both hutches:

- **Same function, same transition:** always `piranha4_connectionInfo()` → `uart._rx._await()` during Alloc. The traceback line numbers move between releases (254 – 294), but the code path is the same.
- **The camera usually sends nothing back:** `len 0, resp: []` in 147 of 158 RIX logs and all 20 TMO logs.
- **More than one camera often fails in the same launch:** 8 of 12 TMO launches (atm and fzp) and 29 RIX launches.
- **Every release is affected:** RIX `lcls2_061224` through `lcls2_090126` and TMO `lcls2_091426` / `lcls2_101226`. That includes releases with and without the January 2026 `SEM` / 0.1 s workaround.

## TMO occurrence history

Times are local (America/Los_Angeles). Log paths are relative to `/sdf/home/t/tmoopr/daq/logs/2026/`. For `.zst` files, line numbers refer to the decompressed log.

| Launch | Release | Cameras that aborted | Detector log: exception / abort line | Control `alloc failed` | Source |
|---|---|---|---|---|---|
| 09/18 11:07:42 | `lcls2_091426` | fzp | `09/18_11:07:42_drp-srcf-cmp013:fzp_piranha_0.log.zst` 623 / 625 | 11:08:53 | **new** |
| 09/19 08:19:43 | `lcls2_091426` | fzp | `09/19_08:19:43_drp-srcf-cmp013:fzp_piranha_0.log.zst` 625 / 627 | 08:21:40 | **new** |
| 10/08 17:37:40 | `lcls2_091426` | atm, fzp | `10/08_17:37:40_drp-srcf-cmp013:{atm_piranha_ip1_0 627/629, fzp_piranha_0 619/621}.log.zst` | 18:39:51 | **new** |
| 10/10 07:28:13 | `lcls2_091426` | atm, fzp | `10/10_07:28:13_drp-srcf-cmp013:{atm 625/627, fzp 620/622}` | 07:29:31 | **new** |
| 10/12 14:29:36 | `lcls2_091426` | atm, fzp | `10/12_14:29:36_drp-srcf-cmp013:{atm 626/628, fzp 621/623}` | 14:30:39 | **new** |
| 10/14 14:50:23 | `lcls2_101226` | atm, fzp | `10/14_14:50:23_drp-srcf-cmp013:{atm 625/627, fzp 618/620}` | 14:51:07, 14:51:22 | report 10/19 |
| 10/15 17:41:33 | `lcls2_101226` | atm, fzp | `10/15_17:41:33_drp-srcf-cmp013:{atm 626/628, fzp 618/620}` | 17:42:17 | report 10/19 |
| 10/18 07:30:43 | `lcls2_101226` | fzp | `10/18_07:30:43_drp-srcf-cmp013:fzp_piranha_0.log` 625 / 627 | 07:32:17 | report 10/19 |
| 10/21 14:06:14 | `lcls2_101226` | fzp | `10/21_14:06:14_drp-srcf-cmp013:fzp_piranha_0.log` 625 / 627 | 14:07:44 | earlier addition |
| 10/22 15:06:35 | `lcls2_101226` | atm, fzp | `10/22_15:06:35_drp-srcf-cmp013:{atm 619/621, fzp 626/628}` | 15:07:52 | **new**, report 10/26 |
| 10/24 07:41:21 | `lcls2_101226` | atm, fzp | `10/24_07:41:21_drp-srcf-cmp013:{atm 621/623, fzp 626/628}` | 07:43:06 | **new**, report 10/26 |
| 10/25 07:16:09 | `lcls2_101226` | atm, fzp | `10/25_07:16:09_drp-srcf-cmp013:{atm 625/627, fzp 618/620}` | 07:16:47 | **new**, report 10/26 |

The matching `drp-srcf-mon001:control.log` for each launch has `<cam> did not respond to alloc` then `alloc failed to change state` (lines 45 – 80 of each file). In 11 of 12 cases the next launch allocated with both Piranhas. After 09/18 11:07 the next launch failed alloc again, but on `timing_0`, not a Piranha.

## RIX occurrence history

RIX logs are in two places:

- **CDS:** `/cds/home/opr/rixopr/YYYY/MM/` on psbuild. `~rixopr/logs` on CDS contains only `timechart.log`.
- **SDF:** `/sdf/home/r/rixopr/daq/logs/2026/` (09/17 – 10/22/2026).

Launches with the error, by month:

| 2024 | 2025 | 2026 |
|---|---|---|
| Jun 1, Sep 11, Oct 4, Nov 2 | Jan 1, May 3, Jun 17, Jul 3, Sep 20, Oct 26, **Nov 32**, Dec 3 | Jan 3, Mar 2, Apr 1, Sep 3 (SDF) |

(129 launches and 158 detector logs on CDS, plus 3 launches on SDF.)

Latest RIX occurrence (SDF, release `lcls2_090126`):

- Detector log: `/sdf/home/r/rixopr/daq/logs/2026/09/17_11:41:19_drp-srcf-cmp012:q_piranha_0.log.zst`, lines 6747 – 6753 (traceback, `prompt not seen: len 0, resp: []`, abort).
  - Before the timeout, the camera sent `0xff` bytes (`_check found \xff\xff…`). The `logging.warning` call itself then raised `UnicodeEncodeError`.
- Control log: `/sdf/home/r/rixopr/daq/logs/2026/09/17_11:41:19_drp-srcf-mon002:control.log.zst`, lines 68 – 73. `q_piranha_0 did not respond to alloc` at 11:42:31, and again at 11:54:04.

Older RIX logs show the same trace. In a few of them the camera did reply but `_await` still failed: either a full `GCP` reply with no final `USER>` (11/2024), or echoed earlier commands with leading `\x00\x1b` bytes (05/2025).

## Environment and code path

- TMO: platform 0, `drp-srcf-cmp013`; release `/sdf/group/lcls/ds/ana/sw/conda2/rel/tmo/lcls2_101226/install` (10/14 onward), `lcls2_091426` before. FZP Piranha uses lane 0.
- RIX: `drp-srcf-cmp012`; cameras `c_piranha_0`, `fzp_piranha_0`, `q_piranha_0`; releases `lcls2_061224` to `lcls2_090126`.
- `piranha4_connectionInfo()` clears the receive state, sends `uart.GCP()`, then waits for the reply and prompt. See [`piranha4_config.py`, lines 272 – 286](https://github.com/slac-lcls/lcls2/blob/4103fda6638219fad4fef5c258600abcc1345e63/psdaq/psdaq/configdb/piranha4_config.py#L272-L286).
- [`_await()`, lines 47 – 61](https://github.com/slac-lcls/lcls2/blob/4103fda6638219fad4fef5c258600abcc1345e63/psdaq/psdaq/configdb/piranha4_config.py#L47-L61) polls every 10 ms for up to **5 seconds**. It returns only if the current line is exactly `USER>` **and** `_resp` is non-empty. Otherwise it raises, and the DRP aborts.

## Hypotheses

- **Not a cmp013-specific hardware fault.** The same failure on a different host in another hutch, across three camera instances and two years of releases, points to the Piranha4 serial handshake, either camera readiness or how `piranha4_config.py` handles it. This replaces the earlier "shared transport on cmp013" lead.
- **The camera is usually silent, but not always.** In most cases nothing arrives within 5 s (`len 0`).
  - In some RIX cases, the camera did reply but `_await` still failed. In one, it returned the whole `GCP` text without a final `USER>`. In another, the reply was echoes of earlier commands with leading `\x00\x1b` bytes.
  - So at least some failures are about prompt and line framing or timing, not a dead link.
- **Possible mechanism (unverified):** the camera is still busy finishing earlier serial commands or a mode change (`spf 0`, `StopRun`) when `GCP` is sent. It drops or delays the prompt, or ends the reply in a way `_check()` doesn't recognise as `USER>`.
- Nothing has been tried that confirms a fix. The January 2026 workaround (comment out `SEM.set('0')`, add a 0.1 s sleep) is present in releases that still fail at both hutches.

## What's been tried / recovery

- At TMO, the relaunch allocated the Piranhas in every case. Recovery on relaunch is observed; it is not a fix.
- The deployed source has a January 26, 2026 comment about an intermittent await crash that automated testing could not reproduce, plus the speculative `SEM` / sleep workaround above.
- This investigation was read-only. No configuration, camera settings, firmware or processes were changed.

## Next investigation / validation

1. **Don't abort the DRP on the first timeout.** Retry `GCP` (for example clear, re-send, and wait again, up to N times) and log the raw bytes received. This should turn most of these from a launch failure into a delay, and the logged bytes show what the camera actually sends.
2. **Make `_check()` / `_await()` more tolerant:** accept a prompt after leading control bytes, and treat a complete `GCP` reply as success even if `USER>` arrives late.
3. **Fix the logging crash:** `logging.warning(f'_check found {prompt}')` raises `UnicodeEncodeError` on non-ASCII bytes (RIX 09/17). Use `repr(prompt)`. This is the same ASCII-locale problem as the Opal UART tracebacks.
4. To test a fix, run repeated Alloc/Dealloc cycles with each Piranha. The failure rate (1 – 6% of launches) means you need on the order of 100 cycles to see it.

## Report provenance

- TMO: daq-agent reports `19T101327-tmo-8a06439cdf79-report` (10/14 – 10/19) and `26T084630-tmo-2973a1db5b9d-report` (10/22 – 10/26). Neither report covered 09/18 – 10/12. Those, and all RIX cases, come from a direct search of the raw logs on 2026-10-26.
- RIX counts come from a search of all Piranha DRP logs under `/cds/home/opr/rixopr/{2024,2025,2026}/` (on psbuild) and `/sdf/home/r/rixopr/daq/logs/2026/`.

---
*Created by Claude (AI), posted via @monarin's account.*
