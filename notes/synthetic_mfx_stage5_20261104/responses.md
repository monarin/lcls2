# Stage 5 review responses

Round 1 PASS at a644240e0dbdc7f2c3a39b71a08c19035da83c9b, no blocking findings.
Full generation started with that exact reviewed generator/validator/helpers;
these files remain unchanged. Source acceptance also passed at that commit.

- S5-ADMIN-01: adopted. Publisher now requires destination and source acceptance
  paths, exact full counts, staging root, 60 equal samples, source report hash,
  and matching reviewed acceptance.py hash for both reports. Journal records
  both artifact paths/hashes before linking. A focused gate test covers seven
  negative variants and one valid report. Publication has not happened.
- S5-ADMIN-02: adopted. Added read-only post_publish.py to compare all 20 run-1
  stats, all 20 published run-2 hard links, and the first calibrated event with
  source. It will be executed after publication; no result is claimed yet.
- S5-ACCEPT-01: deferred. This fixture is intentionally fixed to 30 x 1000;
  acceptance.py is not used for the patched small fixture. Constants and all
  corresponding bounds agree. Preserve the already executed source acceptance
  script and reference provenance. Generalized counts are future scope.
- S5-EVID-01: adopted as evidence-only: small_datasource_command.py.txt is the
  exact executed stdin script, now saved for inspection/reproduction.

## Round 2

PASS at a3852808ada2032a7e6893bbd74bda0a028f8662 with no blocking findings.
S5-ADMIN-01 and S5-EVID-01 resolved. S5-ADMIN-02 code accepted; final execution
and report remain required after publication. S5-ACCEPT-01 deferral accepted.
S5-TEST-01 additional reference-side failure variants are deferred as optional;
existing valid/seven-negative gate tests pass, and actual paired source/full
acceptance reports will be checked before publication. No code changed after
this review. Publication must use this round's full commit SHA and review JSON.

## Final operational evidence

S5-ADMIN-02 is now executed: published_smoke.json passes all 20 final links,
all 20 run-1 file stats and the source-matching first calibrated event. Its
script hash matches the reviewed a3852808ada2032a7e6893bbd74bda0a028f8662 blob.
Publication completed only after full structural and calibrated acceptance
passed; publication.jsonl records both report hashes. No executable code changed
after round 2. The two optional deferrals above remain unchanged.
