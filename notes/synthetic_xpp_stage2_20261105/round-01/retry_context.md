# Stage 2 code and concrete-plan review (retry after timeout)

Review exact commit 57495428124d135e67d3651a1351d3a0e807b74e in this detached
checkout. Base 07cff96d5079527663344ac60356a574124c18b0. Prior attempt timed
out with NO verdict; no findings or approval to inherit. Use only Read/Glob/Grep,
no commands, changes, external services, private payload reads or delegation.
Return the required JSON with stable finding IDs XPP-S2-xx. Keep the summary
concise and the checks list to the material checks, avoiding exhaustive narration.

Scope: NEW Stage 2 code under notes/synthetic_xpp_stage2_20261105. This is a
pre-deployment review gate, not final Stage 2 acceptance. User explicitly
instructed Stage 2 after it was explained as calibration deployment valid3–4
plus private run3 pilot. Those writes are authorized. No final xtc publication,
source-event sharing, permissions changes, run4 generation or human messages.
No DB write has yet occurred. After code/plan readiness approval, the authorized
plan executes once and normal calibrated acceptance/readback must still pass
and be reviewed; absence of those not-yet-executed reports is expected now.

Read scope.md and README.md, then inspect these bounded NEW files:
fixture.py, generate.py, verify.py, calibration.py, acceptance.py, completion.py,
test_stage2.py and runtime.sh. Check the matching committed reports/logs named
below. Review the new logic and any assumptions on its interfaces. Do not
spend this review re-auditing unchanged low-level XTC implementation: Stage 1
was already independently reviewed and passed at 786938d79. fixture.py pins
Stage 1 helpers by checksum; the wire parser was independently reviewed in
Stage 1 and earlier MFX stages. Its narrow contract is reading XTC headers,
Names schemas, field spans, SMD references, packed timestamps and hashes.
If needed, read named dependencies directly (notes/synthetic_xpp_stage1_20261105/
common.py, inventory.py, plan_layout.py and notes/synthetic_mfx_stage5_20261104/
xtc_fixture.py). Do not search the entire repository for submodule sources;
Smd.cc/ShapesData.hh submodule files are unavailable in this detached checkout.
Report that limitation rather than chasing them through unrelated directories.
The harness has independently checked exact full SHA and cleanliness; gitdir
reads can be denied, which is a disclosed limitation rather than a reason for
repeated gitdir exploration. No core psana changes: core MPI suites not required.

Required pre-deployment evidence, all present:
* pilot_generation/verification.json and pilot_raw_acceptance.json: eight private
  files, 5039830328 bytes, 2400 original 1M events, one step, streams2/3/5/6.
  All retained source/output bytes compared except specifically checked identity,
  outer timestamp and SMD-reference edits. Normal DataSource count2400, sample
  raw hashes match source, SMD_N_EVENTS29/batch37/fake-step0.
* seam_generation/verification.json and seam_raw_acceptance.json: eight-event
  shortened-source cycle (3+3+2), one step, first seam crosses a second boundary.
  Every copied byte checked, normal DataSource passes, SMD_N_EVENTS2/batch3.
* source_samples.json: normal selected source calibration IDs/payload hashes
  equal frozen Stage1 for six samples (0,1,2,10,1199,2399), calibrated values
  finite, raw/calibrated fingerprints ready for destination comparison.
* unit_checks.log: nine focused tests pass, including wrong timestamp/env/run/
  offset/payload rejection; both production schedules match frozen Stage1 maps;
  shortened repetition/rollover/truncation; transport guard one-use exact body
  and endpoint with foreign/repeated/mutated writes rejected.
* deployment_plan.json + calibration_prepare.log: GET-only preflight succeeded,
  exact plan SHA 045f55a2c79656db79938c7fae741e061e7c5d66757e024e0fdec4786826d5f6.
  Eight frozen payloads, 125830928 bytes, fresh docs only in
  cdb_prjdatpsana/jungfrau_000010, run3/run_end4; IDs allocated anew and source
  refs only in extpars provenance. No source/shared/name-map writes. Original13
  destination docs, run1/2 selection IDs and protected metadata fingerprints
  saved. Deploy enforces exact plan/script/payload hashes and unchanged baseline,
  refuses any prior attempt, and writes request intents before each allowed POST.
  Verify code checks normal run3/4 lookup, payloads, metadata, validity exclusions,
  old selections/docs and protected source/shared metadata after upload.
* All commit hooks passed. Initial failures are transparently preserved under
  initial_attempt: helper path priority (fixed by sys.path restoration), and
  requests json=None body hashing (fixed and covered by the guard test). No
  failed test sent a DB request. Tiny fixture regenerated after helper fix.

Privacy: all binary source/output paths are outside Git in owner-only storage;
DO NOT read them. Evaluate supplied metadata/hash evidence, with the usual
limitation that you cannot rerun tests or recompute binary hashes. Project quota
for future run4 is unknown; this pilot's bounded5GB allocation actually succeeded.
No exhaustive pixel-science or real compression performance claim.

Focus on genuine new-code risks: HTTP scope and uncertain-write recovery,
calibration document construction/selection, allowed byte changes, source-map
consistency, repetitions/timestamp rollover, validation independence and evidence
claims. Assess whether the code and exact plan are ready for the authorized
execution, without treating future post-deployment checks as already passed.
