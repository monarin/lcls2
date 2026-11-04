# Operational continuation

Generation session 43373 was launched once, with stdout/stderr in generation.log.
Do not restart over its staging directory. Writer: make_pilot.py; exact reviewed
code at a644240e0, unchanged in final reviewed commit a3852808ada2032a7e6893bbd74bda0a028f8662.
Staging: /sdf/data/lcls/ds/prj/prjdatpsana/xtc/.synthetic-mfx-r0002-stage5-20261104
Expected files: 20, total bytes 1019012911780, 30373 records per stream.
Complete: both code reviews PASS, full generation/validation/acceptance PASS,
20 files published, and final calibrated smoke plus unchanged-run-1 checks PASS.

The generation, validation, acceptance and publication operations below have
completed successfully. These commands are a historical record, not a restart
recipe. Do not rerun generation or publication over the existing files.

```bash
bash notes/synthetic_mfx_stage5_20261104/runtime.sh \
  notes/synthetic_mfx_stage5_20261104/verify_pilot.py \
  --root /sdf/data/lcls/ds/prj/prjdatpsana/xtc/.synthetic-mfx-r0002-stage5-20261104 \
  --report notes/synthetic_mfx_stage5_20261104/verification.json

PS_SMD_N_EVENTS=1001 bash notes/synthetic_mfx_stage5_20261104/runtime.sh \
  notes/synthetic_mfx_stage5_20261104/acceptance.py --kind destination \
  --root /sdf/data/lcls/ds/prj/prjdatpsana/xtc/.synthetic-mfx-r0002-stage5-20261104 \
  --reference notes/synthetic_mfx_stage5_20261104/source_samples.json \
  --report notes/synthetic_mfx_stage5_20261104/destination_acceptance.json

bash notes/synthetic_mfx_stage5_20261104/runtime.sh \
  notes/synthetic_mfx_stage5_20261104/pilot_admin.py publish \
  --staging /sdf/data/lcls/ds/prj/prjdatpsana/xtc/.synthetic-mfx-r0002-stage5-20261104 \
  --verification notes/synthetic_mfx_stage5_20261104/verification.json \
  --review notes/synthetic_mfx_stage5_20261104/round-02/review.json \
  --commit a3852808ada2032a7e6893bbd74bda0a028f8662 \
  --journal notes/synthetic_mfx_stage5_20261104/publication.jsonl \
  --acceptance notes/synthetic_mfx_stage5_20261104/destination_acceptance.json \
  --reference notes/synthetic_mfx_stage5_20261104/source_samples.json

bash notes/synthetic_mfx_stage5_20261104/runtime.sh \
  notes/synthetic_mfx_stage5_20261104/post_publish.py \
  --staging /sdf/data/lcls/ds/prj/prjdatpsana/xtc/.synthetic-mfx-r0002-stage5-20261104 \
  --reference notes/synthetic_mfx_stage5_20261104/source_samples.json \
  --report notes/synthetic_mfx_stage5_20261104/published_smoke.json
```

All operation logs and reports are saved alongside this file. The final
reviewed code is unchanged. See README.md and completion_checks.json for
results and provenance, and results_SHA256SUMS for the durable archive manifest.
Any later implementation/fix in this session must still use the review loop.
