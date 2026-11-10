Prepare a focused PR to slac-lcls/lcls2 features/psana2-gpu using master template.
Base e032e0d562f39ecfee735ebeae8b395f28d4b08e. Port exactly the three runtime and
two unit-test changes from research base1805e1d95..reviewed3c3799829. Preserve all
new target changes (borrowed/owned memory accounting, field access and IPC fixes).
Target already fixes the historical BeginStep expectation; do not port that test.
Include a concise benchmark provenance note and minimal changelog (target lacks
one); no unrelated history, harness or receipt dump. Prepare PR title/body using
master d04c3a114 template. PR publication is not part of this preparation step.
Acceptance: patch applies cleanly; runtime methods and unit tests equal reviewed
implementation; full current-target CPU GPU suite; independent Claude review of
port and draft claims. Prior GPU/benchmark results remain historical evidence,
not claimed as execution on this target. No new GPU benchmark required for this
preparation scope. Default three review rounds. No implementation redesign.
