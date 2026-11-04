# Stage 4 synthetic MFX pilot acceptance

Calibrated serial comparisons and all nine MPI cases passed on
`work/synthetic-mfx-scan`. Round 1 passed at `24c927cab7b76a18df9e48bf799df6f6651fec8b`;
nonblocking review corrections are undergoing follow-up review. The Stage 3 files and Stage 2 calibration
records are read-only inputs. The complete Stage 4 gate also requires a test
under the collaborator's own login; Mona's checks cannot substitute for it.

## Validation

`serial_compare.py` reads the original first 60 events and then the complete
pilot with normal live calibration selection (no calibration transport or
selection overrides). It verifies all 23 selected detector/type records against
the frozen source IDs/payload hashes or the deployed destination IDs and validity.
It compares detector identity/segments, all exposed non-config detector event
fields, available EPICS interfaces, and segment presence/damage across all
60 timestamps. Six JF calibrated samples span both sides of each step boundary
and the tail: indices 0, 18, 19, 38, 39, 59. Calibration uses explicit
`cversion=3, cmpars=None` on both inputs. Fingerprints include dtype and shape;
only floating NaN payload bits are normalized. Finite values are compared
exactly; matching NaNs occupy identical positions. Per-gain status masks are
checked, and complete assembled images at indices 0 and 59 are compared by
fingerprint. PNG previews are stride-8 visualizations with a 1–99% display scale.
A separate calibration-skipped serial `run.events()` pass checks all timestamps
and step values after the calibrated `run.steps()` pass.

`run_matrix.py` launches nine bounded local CPU MPI cases sequentially, with
600-second timeouts per case. Four cases each use 1 SMD0 + 1 EB + 2 BD and
1 SMD0 + 2 EB + 4 BD. `(PS_SMD_N_EVENTS, batch_size)` pairs are `(19,7)`,
`(20,20)`, `(21,19)`, `(61,21)`; these straddle step boundaries and include a
short final batch. Both `run.steps()` and `run.events()` are checked in every
case, including exact aggregate timestamp multiset, step assignment, per-rank
order, topology, and full raw JF fingerprints. These eight cases skip calibration
to isolate transition delivery. The ninth repeats `(20,20)` with four ranks and
normal calibration enabled for the steps pass, comparing all six calibrated
samples against serial. Per-rank step counts need not match serial counts.

`test_acceptance.py` checks that the acceptance logic rejects duplicated events,
a missing tail, wrong step assignment and changed data/dtype/shape, while accepting
unordered rank aggregation and equal NaNs. A narrow regression check ensures
only the known missing-encoder accessor error is recorded, not arbitrary errors.

## Runtime and reproduction

Use `bash notes/synthetic_mfx_stage4_20261104/runtime.sh SCRIPT ...` from the
repository root. It selects the same installed Python 3.11/psana build as Stage 3,
records imported module paths/hashes, limits CPU threads, and rejects inherited
nonzero `PS_FAKESTEP_FLAG`. This is acceptance for that recorded runtime, not a
claim about every psana release. The normal calibration client works unchanged.

```bash
bash notes/synthetic_mfx_stage4_20261104/runtime.sh \
  notes/synthetic_mfx_stage4_20261104/serial_compare.py --kind source \
  --report notes/synthetic_mfx_stage4_20261104/source.json --plots /tmp/stage4-images
bash notes/synthetic_mfx_stage4_20261104/runtime.sh \
  notes/synthetic_mfx_stage4_20261104/serial_compare.py --kind pilot \
  --reference notes/synthetic_mfx_stage4_20261104/source.json \
  --report notes/synthetic_mfx_stage4_20261104/pilot.json --plots /tmp/stage4-images
bash notes/synthetic_mfx_stage4_20261104/runtime.sh \
  notes/synthetic_mfx_stage4_20261104/run_matrix.py
```

Reports and the matrix directory must not already exist. Large per-event JSON
reports are ignored by Git and archived in project results; compact summaries,
code, logs and review records are committed. Do not regenerate or redeploy the
fixture when repeating acceptance.

## Observations and limitations

- A preliminary `PS_SMD_CHUNKSIZE=262144` read failed on the original source
  Configure with an oversized-dgram error: the EPICS SMD Configure is 299,652
  bytes (larger than 262,144). The runtime message incorrectly labels that raw
  byte count as MB. The acceptance uses psana's standard
  268435456-byte buffer. This is not a claim that small byte buffers are supported;
  event-count batch sizes are varied independently. Original failure log retained.
- Source `MfxUsbEncoder01.raw.descriptions()` raises a TypeError when encoder
  segments are absent. The comparison calls the normal accessor, records exactly
  that known exception only when segments are missing, and requires identical
  behavior in the pilot. No psana core fix or silent substitution is applied.
- PV constants retain the previously documented foreign RIX provenance; their
  raw APIs do not consume calibration. Matching source behavior does not establish
  physical validity of those camera/QADC constants.
- The pilot samples one short interval. Performance and GPU behavior are outside
  Stage 4. Run-2 generation/calibration remains Stage 5 and is not authorized here.

## Collaborator handoff

`collaborator_check.py` is standalone and must run under Seema's own login in
her activated psana environment. It lists and opens all twenty final files,
loads run 1 with calibration, verifies JF calibration document IDs and compares
one calibrated event against the Stage 4 reference. It records login, UID,
groups, host, runtime, input hashes and results. Output goes to her writable
working directory. Do not run under Mona and claim collaborator acceptance.

## Serial results

All 60 per-event records compare equal, including 79 detector field interfaces
and all 267 EPICS aliases. Of the detector fields, 72 return a value in at least
one event; the encoder fields and five MfxUsrDio fields have no values in this
interval. The encoder descriptions exception occurs in all 60 source and pilot
events. These absences match; this fixture cannot test populated encoder/DIO data.
All EPICS aliases return at least one value. All six calibrated arrays are
float32 `(32,512,1024)` with 16,777,216 finite pixels and no NaNs. The assembled
images at events 0 and 59 compare exactly, including shape and dtype. Preview
inspection shows the same ring pattern and detector/module gaps. Per-gain status
counts remain 16,732 / 11,917 / 12,768 masked pixels, matching the normal status
mask API. All 23 source and destination calibration selections/payloads pass.

The first pilot-driver attempt used `step.timestamp`, which is not an accessor
in this runtime; it was corrected to `step.evt.timestamp` and the complete pilot
check reran successfully. The source reference was unaffected (it uses
`run.events()`); `history/serial_compare_source.py.txt` preserves its exact
script bytes/hash. Failed probe logs are retained and are not counted as passes.

Durable reports and four previews are under:

```text
/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/r0001/stage4-20261104/
```

For Seema, after activating her normal psana environment, run from a writable
working directory under her own login:

```bash
python /sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/r0001/stage4-20261104/collaborator_check.py \
  --reference /sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/r0001/stage4-20261104/pilot.json \
  --output collaborator-report.json
```

Return that JSON (or the error output). Reading the script/reference also tests
access to the results area. No collaborator execution has been claimed yet.

All nine MPI cases passed both iteration paths (18 distributed passes), including
the six calibrated samples in the four-rank case. No duplicated/missing timestamps,
misplaced steps, raw mismatches or calibrated mismatches were observed. This
checks the recorded CPU runtime and pilot; it is not throughput evidence.

All nine MPI logs retain UCX unmatched tag-receive descriptor warnings at
shutdown, after the report was saved and PASS printed. The active PML was UCX;
the launch option `--mca btl self,vader` did not select the PML or constrain that
UCX transport. All nine launchers exited zero and all assertions passed; this is not a claim of
warning-free MPI shutdown. Calibration-empty warnings are from the explicitly
calibration-skipped passes. No MPI runtime or psana shutdown fix is included.
`provenance.json` verifies all eleven run reports against the committed code
actually used, with the preserved source script noted above.

## Review provenance clarifications

All nine MPI cases used the working-tree `pilot.json` generated before the matrix,
SHA256 `c5abc67a6825f065229f7e9f043150bd87ee4474f1484634dab87a2a4c3f42e5`.
The same bytes are archived in results and identified in `serial_summary.json`.
The historical MPI reports do not embed this input hash; a future MPI driver
revision should record the reference hash directly. No retrospective edit to
those reports is made. The reviewer cross-checked all three recorded BeginStep
timestamps against the Stage 1 schedule and original first BeginStep; they match.
An explicit schedule assertion can be added in a future driver revision.

The collaborator script now prints Python executable/version and hashes/paths
for psana, Dgram, Jungfrau and UtilsJungfrau before data access, and includes them
in a successful report. If `Calibrated sample differs` occurs under a different
psana build, return the error and printed environment details: the comparison is
inconclusive for access acceptance and does not by itself establish a fixture
failure. The initial one-event calibration probe is superseded by the archived
full pilot comparison (eight JF types, six calibrated samples).
