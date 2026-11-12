## Motivation

When the intermittent Piranha allocation GCP timeout occurs, immediate teardown loses the live Rogue state needed to investigate it. This adds standalone single-camera and paired cold/warm allocation checks, plus opt-in inspection that retains both initialized camera roots after that exact timeout.

The coordinated path parks successful peers before cleanup, preserves the original UART failure capture, records bounded late responses, and exposes cached software diagnostics through the existing PyRogue server. Run-scoped release, finite hold deadlines, parent-loss handling and independent cleanup watchdogs bound the inspection. A genuine timeout remains a failed campaign; no automatic retry or reset is performed. Ordinary mode remains the default.

Related to slac-lcls/lcls2#205; this diagnostic tool does not fix or yet reproduce that issue. `--inspect-after-trial` provides a separately labeled successful-trial rehearsal.

This development PR also carries the requested design, independent review records and historical test archive. Start with [the work overview](https://github.com/monarin/lcls2/blob/features/piranha-routine-check/notes/tmo-piranha-routine-check/README.md) and [operator README](https://github.com/monarin/lcls2/blob/features/piranha-routine-check/psdaq/psdaq/debugtools/piranha4_pilot/README.md); executable changes are confined to `psdaq/psdaq/debugtools/piranha4_pilot/`.

## How to test the behavior?

Offline, using the pinned Python 3.9 environment (software-only PyRogue sockets; no camera access):

```bash
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 \
/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/daq_20250402_r9/bin/python \
-m unittest discover -s psdaq/psdaq/debugtools/piranha4_pilot -p 'test_*.py' -v
```

The suite covers early-success peers, both camera failure orders, cold/warm reuse, immutable and late capture, release/expiry, parent signals/death, cleanup hangs including native GIL retention, cached access, and the real pinned software-only PyRogue client.

Dry-run previews (no hardware imports or opens):

```bash
python psdaq/psdaq/debugtools/piranha4_pilot/pilot.py --cameras both --mode cold --trials 3 --hold-on-error
python psdaq/psdaq/debugtools/piranha4_pilot/pilot.py --cameras both --trials 1 --inspect-after-trial 1 --hold-seconds 60
```

Hardware rehearsal of the new hold path is still pending, so this PR is draft. The 31 successful historical camera calls validate only the earlier non-hold pilot. A current exclusive-board window is required before running rehearsal. Generic GUI clients are write-capable, not an enforced read-only spy interface.

Known non-blocking follow-up HOLD-IMPL-007: cancellation before all hold acknowledgments may create a successful peer's `failure.json` with `target_timeout: false`. Do not count every failure file as another camera timeout.

## Checklist

- [x] Updated `CHANGELOG.md`.
- [x] Described the problem and resulting behavior.
- [ ] GitHub Actions checks pass (check current PR status).
- [ ] Closes an issue: intentionally unchecked; issue 205 remains unresolved.
