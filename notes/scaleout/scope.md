# Scope

Prepare and independently review a test plan for psana2-gpu scaling on Perlmutter.
Pin PR262 head, create the personal-fork branch, define Lustre baseline and minimal
CPU/GPU resource selection, large-data preparation, and weak scaling. This turn
is planning and read-only discovery; baseline performance is not yet measured.
No production optimization, staging, rebuild or Slurm submission in this turn.
Codex authors; Claude reviews detached commits, up to three rounds, 900 seconds
per call. Accept only a reproducible plan with correct provenance, separate
storage/staging/application metrics, cache/timing/correctness gates, bounded
resource/data sizing, explicit topology and statistical decision criteria.
Publish final plan and compact review records only to monarin/lcls2.
