# Paired Piranha hardware tests — November 9, 2026

Executed by Codex via SSH as `monarin` on `drp-srcf-cmp013`, after Mona confirmed the DAQ was stopped and no experiment was running in response to the exclusive-board question. Both cameras used `/dev/datadev_0`: FZP lane 0 (serial 12097685), ATM IP1 lane 1 (serial 12102856).

All 3 paired cold trials and 3 paired warm trials passed: 12 successful camera connection attempts. No allocation GCP timeout was reproduced. Both cameras reported BiST Good.

| Mode | Trial | FZP PID | ATM PID | FZP GCP wait (s) | ATM GCP wait (s) |
|---|---:|---:|---:|---:|---:|
| cold | 1 | 3376497 | 3376498 | 1.860 | 1.839 |
| cold | 2 | 3377180 | 3377181 | 1.849 | 1.868 |
| cold | 3 | 3377800 | 3377801 | 1.849 | 1.860 |
| warm | 1 | 3380282 | 3380283 | 1.839 | 1.850 |
| warm | 2 | 3380282 | 3380283 | 1.859 | 1.839 |
| warm | 3 | 3380282 | 3380283 | 1.849 | 1.869 |

Cold created fresh processes/roots for each trial. Warm reused one process/root per camera; its first trial followed initialization, and trials 2–3 reused those initialized roots. Neither mode power-cycled the cameras or performed Configure/acquisition. FZP was the supervisor and ATM the worker in all trials.

Used the pinned November 3 release, Python 3.9.21, Rogue v6.1.3, 40-second initial settle, 2-second inter-trial interval, 60-second watchdog and 10-second cleanup timeout. Script SHA256: `d7c7e249753f29dccf5d713365b79758f8fb60da42fa4e336b5728d2ea10b804`. All source-hash guards passed.

Every child exited with code 0, without forced kills, unreaped PIDs, cleanup errors, or UART capture drops. Between campaigns, a live check found no remaining cold-test PIDs, listeners, or visible board holders. Final live verification at 2026-11-10T00:47:13.186217+00:00 found no test PIDs and no listeners on ports 5561, 5562, 9099–9101 or 9103–9105. No visible device holders remained; 914 process descriptor directories were inaccessible, so this is not a system-wide ownership proof.

This small standalone smoke test does not rule out the intermittent production failure or establish full-DAQ health. No DAQ restart or hardware reset was performed. The pilot changes camera/timing state; normal DAQ configuration must be restored before returning to acquisition.

Evidence:

- [Cold summary](paired-cold-3/summary.json)
- [Warm summary](paired-warm-3/summary.json)
- [Verified per-camera results](verified-results.json)
- [Final live cleanup check](post-run-check.json)

Each campaign directory also contains the complete process logs, provenance, UART captures, and per-trial result files.
