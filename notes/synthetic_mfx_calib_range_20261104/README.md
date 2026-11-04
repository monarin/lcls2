# Calibration validity extension — November 4, 2026

Mona authorized extending the existing prjdatpsana calibration validity from
runs 1–1 to 1–2. The operation changes only `run_end` in 13 experiment records:
eight Jungfrau constants and five shared `pv_000001` constants. Document IDs,
payload IDs, payloads and every other metadata field remain unchanged. No new
constants were uploaded; shared/source databases and XTC files were not modified.
The original deployment comment still describes the run-1 pilot; the explicit
`run` and `run_end` fields control validity.

`execution/before.json` preserves original metadata; `planned.json`, per-document
intent/HTTP response records, and `after.json` record the update. The script
restricts mutations to PUT requests for the 13 journaled experiment document
URLs, with an exact allowed document body. It refuses a repeated application
once any record no longer has validity 1–1. Ambiguous failures require inspection;
it does not automatically retry or roll back. The service provides no atomic
multi-document transaction or compare-and-swap; a read immediately before each
write detects prior changes but cannot eliminate a concurrent writer race.

`execution/verification.json` records normal psana calibration selection for
both runs (23 detector/type selections per run), checked against the Stage 2
document/data IDs and Stage 1 frozen payload hashes. These experiment records
are not eligible for run 3; ordinary run-3 lookup can still use shared fallback.
The foreign RIX provenance of the PV constants is unchanged; PV raw interfaces
do not consume those constants (see prior audit).

This completes the calibration prerequisite for run 2. It does not generate
run-2 XTC. Seema's own-login acceptance is recorded separately in the Stage 4
collaborator addendum. Her existing Stage 4 check
compares document IDs and calibrated values, so it remains applicable.
Historical Stage 1–4 evidence is unchanged. The original Stage 4 serial driver
asserts run_end=1 and should not be used unchanged to test current validity.

For a fresh read-only check, run with the same pinned Stage 4 runtime and a new
output directory (omit `--apply`):

```bash
bash notes/synthetic_mfx_stage4_20261104/runtime.sh \
  notes/synthetic_mfx_calib_range_20261104/extend.py --output /tmp/calib-range-check
```

Review status and commit identity are recorded in `status.json` and `round-01/`.
No psana core code changed; verification is scoped to live metadata and payload
selection. The first bare `python -m py_compile` command could not run because
`python` was not on PATH; compilation was then successful using the pinned
conda Python. The actual update/verification used the pinned runtime throughout.

Live update and all 46 calibration selections passed. Claude independently
reviewed implementation commit `27c853b2902f7c6e5c5adecec197f27424fddbe7`
and returned PASS with no blocking findings. Both low documentation findings
are addressed in the final evidence records; see `responses.md` and
`round-01/review.json`. The initial transfer rejection was resolved by Mona's
explicit approval; it is retained as history in `round-01/review_blocked.json`.
Seema's subsequently supplied own-login report also passed the Stage 4 reference
comparison; see the Stage 4 collaborator addendum.

Durable archive:
`/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/r0001/calib-range-20261104`.

The original archive retains the initial pending-review state. The final review
and response archive is
`/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/r0001/calib-range-review-20261104`.
