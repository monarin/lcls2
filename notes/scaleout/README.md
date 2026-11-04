# Perlmutter scale-out campaign

[Final test plan](test-plan.md) — independently reviewed in three rounds,
final verdict **pass**, zero findings on e9cdaff4b00e564160452854896dcfabb30aed5e.
[Baseline manifest](baseline.json) pins PR262 to
5b082e0fdfcefc36ea208c0954aab6548aa6bdb7.

Branch: monarin/lcls2, codex/psana2-gpu-scaleout.
Worktree: ~/lcls2_worktree/codex/psana2-gpu-scaleout.
[Review findings and responses](review/) preserve all seven resolved findings.
[Status](status.json) separates approved planning from unstarted measurements.

Next execution step: freeze baseline/harness builds and verify run442 correctness,
then prepare sufficient private data and measure host Lustre and exact-pattern
read ceilings before GPU resource tuning. No new performance numbers, data copies,
rebuilds or Slurm jobs were produced during planning.

Raw review artifacts remain at
/pscratch/sd/m/monarin/psana2-gpu/scaleout-plan-20261104/.
Use python3.11 for the implementation-review helper on Perlmutter.
