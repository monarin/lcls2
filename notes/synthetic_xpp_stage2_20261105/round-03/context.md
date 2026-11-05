# Stage 2 final execution-evidence review

Review 30305b95b3fce95741d345d01eb390b90051d37d, base round2-approved
89801cc72f11aaf4b47bb5b8b6b8227a90f27ffb. NO Python/shell implementation
changed since round2; coordinator confirmed empty git diff for Stage2 *.py.
This is an evidence-only review of completed, already-authorized deployment
and private pilot. All paths below are notes/synthetic_xpp_stage2_20261105/.

Read round-02/review.json + responses.md, then new deployment/journal.json,
deployment/attempt.json, calibration_deploy.log, calibration_readback.json,
pilot_acceptance.json, seam_acceptance.json, completion.json and README.md.
Cross-check with the already-reviewed deployment_plan.json, source_samples.json,
pilot_generation.json and seam_generation.json. Inspect request intent/http/ID
records under deployment/ as needed to confirm the eight paired writes. Raw HTTP
bodies are private; no private binaries/payloads should be read. Read validation
code only as needed for claims; it is unchanged and already approved.

Actual coordinator results:
- One deployment, 16 POSTs, all HTTP200, eight allocated data/doc pairs only in
  cdb_prjdatpsana/jungfrau_000010, run3/run_end4; exact reviewed plan SHA
  0fe56e202307ea7b796db95ae943c76d292c69410b7f7831e036b358f6e47c77.
- Exact metadata/payload readback and normal calibration lookup for run3 AND4
  passed;13 old docs/run1-2 selectionIDs/source-shared metadata unchanged.
- Normal calibrated DataSource pilot:2400events,oneStep,6 sample raw/calibrated
  fingerprints equal source, batch37/SMD29; seam8events,oneStep,repeated3samples
  equal source,batch3/SMD2. No fake steps. Source samples frozen before deploy.
- completion:40 oldfiles/14sourcefilestats unchanged;allreport/currentcode hashes
  checked;private0700dirs/0600XTCsizes preserved;no final run3/4publication.
- 13 focused tests passed before deployment; no code changed/no reason to repeat.

Resolve XPP-S2-08 (actual successful document POST/readback now available) and09
(status at previous commit updated to round2SHA; final reviewSHA will be recorded
in evidence-only followup as usual). Other findings resolved by round2. Check for
inconsistent completion claims/new blockers. The proposal status at
psana/psana/gpu/docs/proposals/synthetic_jungfrau_decompression_cube.md now links
actualStage2records; finalreviewpending wording is intentional until verdict.

Explicit Stage2 instruction authorized scoped DBwrites/privategeneration.
No publicXTC,permissionchanges,logbook,run4generation,GPUkernels or messages.
Stage3 publication is separate; no claim Seema can yet read the private pilot.

Return concise structured review (max8materialchecks). Read-only text review,
no commands, externaltools, mutation, delegation, privatebinaries or recursive
review. Do not re-audit unchanged XTC/parser/helper internals; submodules/gitdir
may be unavailable. Harness enforces exactSHA/cleanliness. Distinguish coordinator
runtime evidence from checks you can independently perform by reading.
