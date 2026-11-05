# XPP Stage 2: run-3 pilot and calibration 3–4

Stage 2 execution and all acceptance checks passed. Final independent evidence
review is pending. Eight experiment-only calibration documents are deployed for
runs 3–4; the private run-3 pilot contains 2,400 events in one native step.
Stage 3 publication to `/sdf/data/lcls/ds/prj/prjdatpsana/xtc/` is excluded.

Private run-3 pilot directory:
`/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_xpp_stage2_private_20261105/pilot/`.
The root and subdirectories are mode 0700; XTC files are explicitly mode 0600.
Inherited group ACL entries have no effective directory access under this mask;
preserve that mask and the explicit modes. No existing permissions were changed.

The pilot stores 2400 JF1M events in one step across source streams 2/3/5/6:
timing/triginfo, Jungfrau, EPICS and BLD. Eight files total 5,039,830,328 bytes.
No 16M materialization or bin payload is stored. The original source-event data
remain unchanged, with experiment/run identity, closing timestamps and SMD
references rewritten as specified by Stage 1. The eight-event `seam/` fixture
uses a shortened three-event source cycle twice plus a two-event prefix and
forces a second rollover. It is only a validation fixture, never a published run.
The production pilot/full schedules are checked against both frozen Stage 1 maps.

`generate.py` writes exclusively into new directories. `verify.py` compares every
retained datagram against the actual source, permitting only timestamp, runinfo
and SMD-reference edits; it checks each field's new value, exact totals, EOF and
the enabled one-step state machine. `acceptance.py` uses normal serial DataSource
with fake steps disabled, non-dividing batch sizes, all selected small detectors,
complete event counts, and source raw/calibrated sample comparisons. Calibrated
checks use CPU-v3 with cmpars=None, matching Stage 1. Scientific equivalence to
real decompression or a physical cube is not claimed.

`calibration.py prepare` performs GET-only checks and saves a concrete eight-doc
plan, source/shared metadata fingerprints, original 13 destination documents and
run-1/2 selection IDs. `deploy --plan-sha HASH` permits only one explicitly
hashed POST at a time, only to the destination experiment's GridFS and
jungfrau_000010 collection. Each request has a durable intent before sending,
then a response/ID record. Any prior deployment attempt makes another deployment
refuse; partial/uncertain requests require inspection, not an automatic retry.
Readback verifies normal run-3 and run-4 resolution, all payload hashes, fresh
destination metadata, exclusion outside 3–4, unchanged old documents/selections
and unchanged source/shared metadata. No detector-shared DB is written.

The user's explicit Stage 2 instruction authorizes this calibration deployment
and private generation. Collaborator-facing event-payload publication remains a
separate stage. Pilot allocation is bounded to about 5 GB and successful writes
established capacity for this pilot; full-run quota confirmation remains a
Stage 4 prerequisite. No logbook registration or human messages are included.

Validation commands are recorded in logs; do not blindly replay writers or
exclusive report creation. Use runtime.sh for the pinned Python/psana install.
The thirteen focused tests cover disallowed/mutated HTTP writes, one-shot request
consumption, timestamp/offset edits, header/payload/run corruption, both production
maps and repeated/truncated source cycles across a second boundary.

Initial diagnostics are preserved in `initial_attempt/`: a test import selected
the Stage 1 calibration module because the temporary helper path was not removed;
fixture.py now restores the caller's module priority. The initial tiny fixture
is preserved separately, and the final one was generated with the corrected
code. A subsequent guard test caught requests' automatic `json=None` argument
being hashed instead of the binary body. The guard was fixed and the read-only
deployment plan regenerated before any remote mutation. No failed test caused a
DB request to be sent. The source reference samples remain valid across the
import-path fix; their acceptance driver was unchanged.

Review uses the requested Codex/Claude loop, at most three rounds. Round 1 reviews
code, the actual private pilot and raw DataSource evidence, and the dry-run plan.
After it passes, the authorized plan is executed once and complete calibrated
acceptance/evidence are reviewed again. No core psana implementation is changed.

Round 1 passed with seven nonblocking findings. `round-01/responses.md` records
six fixes and the intentionally fresh destination metadata. Round 2 reviews
these fixes and the regenerated plan before the authorized deployment.

If an upload stops, preserve the deployment directory and private response bodies.
Inspect each intent, HTTP result, returned GridFS/document ID and completed pair;
use GETs to reconcile the destination collection and `fs.files`, then retrieve
candidate data IDs and compare payload hashes to the frozen plan. A data POST
without a matching document may leave an orphan blob; a lost response may instead
hide a successful write. Never delete evidence, rerun deploy, or infer absence
from a missing response. Any repair needs a concrete scoped plan after readback;
the current writer offers no automatic cleanup/retry.

`fixture_contract.json` is a metadata sidecar: native uint16 JF1M input and a future
`repeat16-v1` expansion into 32 panels. Its fly-scan bin rule uses the global logical
event index modulo 30, independent of the single native DAQ step. Neither GPU
expansion nor cube kernels are implemented in this stage.

## Execution results

- Sixteen POSTs returned HTTP 200: eight data blobs plus eight documents. Every
  intent, safe HTTP result, allocated ID and completed pair is in `deployment/`.
  Raw HTTP bodies are kept only in the private results directory.
- `calibration_readback.json` verifies all eight types for both runs 3 and 4,
  exact metadata/payload hashes, unchanged 13 existing documents and run-1/2
  selections, and unchanged source/shared metadata. New records do not match
  runs 1, 2 or 5 (shared fallback outside 3–4 is still possible).
- `pilot_acceptance.json`: 2,400 events, one step, raw shape (2,512,1024), all six
  raw/calibrated samples identical to source. Batch size 37 and SMD batch 29.
- `seam_acceptance.json`: eight events, one step, repeated raw/calibrated samples
  identical, with a second-boundary rollover. Batch size 3 and SMD batch 2.
- `completion.json`: all 40 run-1/2 files and 14 source-file stats unchanged;
  output/manifests and owner-only access checked; no run-3/4 final XTC published.

Example for the owner using an activated psana environment:

```python
from psana import DataSource

ds = DataSource(
    exp="prjdatpsana", run=3,
    dir="/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_xpp_stage2_private_20261105/pilot",
    live=False,
)
for run in ds.runs():
    jf = run.Detector("jungfrau1M")
    for step in run.steps():
        count = sum(1 for evt in step.events())
        print(count)  # 2400
```

Set `PS_FAKESTEP_FLAG=0`. The directory remains private: collaborator publication
and access validation are Stage 3. Run 4 has calibration ready but no generated
XTC yet. No GPU decompression/cube implementation or throughput claim is made.
