# Operational continuation

Generation session 43373 was launched once, with stdout/stderr in generation.log.
Do not restart over its staging directory. Writer: make_pilot.py; exact reviewed
code at a644240e0, unchanged in final reviewed commit a3852808ada2032a7e6893bbd74bda0a028f8662.
Staging: /sdf/data/lcls/ds/prj/prjdatpsana/xtc/.synthetic-mfx-r0002-stage5-20261104
Expected files: 20, total bytes 1019012911780, 30373 records per stream.
The code review is complete (both rounds PASS). Generation is still in progress
as of this note. Source calibrated acceptance is complete, source_samples.json.

After the writer exits successfully, confirm generation.json is complete and
file sizes total exactly the preflight estimate; save a compact generation
summary, keeping the large full trace in staging and durable results, not Git.
Then run these individually, inspecting success before dependent operations:

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

Redirect each operation to its own log. Publication is authorized by the user;
no extra confirmation is needed. If any step fails, inspect before retrying;
never replay publication blindly. Record post_publish.py's Git/blob provenance,
all final ACLs, durable results with SHA256SUMS, final status/README/proposal,
and an evidence-only commit/push. No run-2 MPI/GPU benchmark is part of this
request. No DB writes or logbook registration. Review loop remains required
for any executable fix; max three rounds, two already completed.
