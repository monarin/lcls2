# XPP run-311 Stage 1 preflight

In progress. Source inventory, calibration snapshot, representative samples,
layout computation and administrative checks have passed. The full Jungfrau
frame audit and independent review remain pending. This is read-only preparation:
no run-3/4 XTC files, database records or permission changes have been made.

Source: `/sdf/data/lcls/ds/xpp/xpp101605526/xtc/`, run 311, 151,152 events.
Private snapshot (mode 0700):
`/sdf/scratch/users/m/monarin/gpu-validation/synthetic-xpp/stage1-20261105/`.
SMD/configuration/transition copies, indexes, calibration bytes, source mapping
arrays and catalog snapshots stay there, outside Git and public results. Scratch
is working storage; preserve on an approved private durable destination before
purge. Compact manifests, scripts, hashes and review evidence live here.

## Findings established so far

- All seven SMD streams have exactly the same 152,419 timestamp/service pairs:
  151,152 events, 1,260 SlowUpdates and seven opening/closing transitions. The
  source has one BeginStep and one Enable; every event/SlowUpdate is enabled.
- Retain whole source streams 2 (timing/triginfo), 3 (JF), 5 (EPICS) and 6
  (BLD/beam monitors/encoder). Exclude streams 0 (alvium_1), 1 (epix100) and
  4 (alvium_tt). Keep source stream IDs; four bigdata/SMD pairs, one chunk each.
  No mixed-stream image filtering is needed for this layout.
- BeginStep/EndStep/Enable/Disable/EndRun templates in retained streams are
  empty 24-byte datagrams. There is no scan Names payload to preserve or add.
  BeginRun carries runinfo in all four retained streams; its existing field
  widths permit in-place destination experiment/run replacement. No Enable
  chunkinfo payload needs renaming.
- SMD L1Accept parent Src is reconstructed as 0; the JF bigdata parent Src is
  its source node ID (3). Timing/env match. The writer must preserve these
  different XTC headers. `xtcdata/xtcdata/xtc/src/Smd.cc` explains this behavior.
- Calibration resolves to `jungfrau_000010`, distinct from run-1/2 MFX
  `jungfrau_000009`. Eight payloads totaling 125,830,928 bytes are frozen:
  five pedestal-family types from source experiment run 130; geometry from
  detector-shared run 148, gain/offset from detector-shared run 137. Exact IDs,
  origin experiments, types/shapes and hashes are in calibration.json.
- Normal calibrated DataSource sampling independently matched all eight live
  selections/payload hashes. Twenty-two source events cover the pilot cutoff,
  full-source endpoints, run-4 tail cutoff and evenly spaced interior samples.
  All calibrated outputs are finite; threshold >5 counts range from 0 to 3,973
  of 1,048,576 pixels. Two invalid gain-code-2 pixels occur in every sample.
  Status masks mark 3,109 / 3,338 / 3,524 pixels for the three gain planes.
  These characteristics are retained and reported, not silently repaired.
  Calibration uses CPU-v3 with `cmpars=None`; common-mode equivalence is not
  claimed. Sample statistics do not establish full-run pixel-value quality.
- Run-3/4 file and logbook names were unused at inspection, not reserved.
  All 13 destination calibration documents still match the completed run-1/2
  validity-1–2 snapshot; all 40 existing files have recorded baseline stats.
  Filesystem free bytes are recorded in admin_checks.json. The Weka quota
  query requires authentication, so project quota remains unconfirmed.

## Exact proposed output contract

| Run | Events | Records per stream | SlowUpdates | Total XTC + SMD bytes |
| --- | ---: | ---: | ---: | ---: |
| 3 | 2,400 | 2,428 | 21 | 5,039,830,328 |
| 4 | 400,000 | 403,342 | 3,335 | 839,913,818,464 |

Both schedules pass the one-step/one-Enable state machine. Sizes assume whole
retained streams and byte-length-preserving identity/timestamp/SMD-offset edits;
no extra bin payload, scan Names or image materialization in the XTC. Stage 2
must reproduce these totals or explicitly revise the contract.

Run 4 uses two complete 151,152-event sequences and a 97,696-event prefix.
Repetition shifts are integer nanoseconds: the source Disable-minus-Enable
interval, 1,259,698,068,452 ns per copy. Original first-interval timestamps are
retained; each later event/SlowUpdate receives the same cycle shift on every
stream. Embedded detector counters/timestamps are unchanged source values.
Only one initial Configure/BeginRun/BeginStep/Enable and final
Disable/EndStep/EndRun are present. Closing template timestamps are 1, 2 and
3 microseconds after the last retained body datagram. No seam transitions.
The minimum adjacent datagram gaps are 285 ns (pilot) and 4 ns (full), positive
and normalized across seconds. Mapping arrays and boundary examples are in
layout.json and the private snapshot.

## Validation and reproduction

Run each command individually from the repository root, inspecting its output:

```bash
bash notes/synthetic_xpp_stage1_20261105/runtime.sh notes/synthetic_xpp_stage1_20261105/inventory.py
bash notes/synthetic_xpp_stage1_20261105/runtime.sh notes/synthetic_xpp_stage1_20261105/calibration.py
bash notes/synthetic_xpp_stage1_20261105/runtime.sh notes/synthetic_xpp_stage1_20261105/admin_checks.py
bash notes/synthetic_xpp_stage1_20261105/runtime.sh notes/synthetic_xpp_stage1_20261105/audit_jungfrau.py
bash notes/synthetic_xpp_stage1_20261105/runtime.sh notes/synthetic_xpp_stage1_20261105/samples.py
bash notes/synthetic_xpp_stage1_20261105/runtime.sh notes/synthetic_xpp_stage1_20261105/plan_layout.py
bash notes/synthetic_xpp_stage1_20261105/runtime.sh notes/synthetic_xpp_stage1_20261105/test_preflight.py
```

The snapshot/report writers refuse existing final report names. These commands
record how checks were run, not an instruction to overwrite/replay completed
artifacts. Inventory supports verification of partial snapshots after interruption.
The runtime pins the previous psana installation, serial mode and fake-step=0.
The wire parser is reused from reviewed Stage 5 and pinned by SHA256 in common.py.
Eleven focused tests cover malformed/missing/damaged frames, differing SMD Src,
repetition/prefix schedules, second rollover and prohibited HTTP writes.

The full JF audit reads bounded 64 MiB blocks and records block hashes, checks
every L1 header, both segment ShapesData extents/shapes/damage and Data lengths,
and reports missing/damaged frames. It does not interpret every pixel value.
Other source streams are indexed from frozen SMD: non-SlowUpdate transition
bytes and first/last SlowUpdate copies are checked, remaining SlowUpdate lengths
are inferred and bounded by following offsets/EOF. Full retained-payload source
comparison remains a Stage 2 generation/validation requirement.

Retained diagnostics: the first inventory attempt was deliberately terminated
because sparse reads from large files were slow; verified partial snapshots
were resumed. An initial admin query incorrectly included MongoDB internal
GridFS collections and returned HTTP 500; those collections are now excluded.
The first JF audit rejected the expected SMD-vs-bigdata Src distinction; the
corrected check follows the source SMD generator and separately validates
bigdata Src while preserving actual damage. No scientific data were altered.

## Later-stage prerequisites

Private-source sharing authorization for collaborator-accessible payloads and
constants is still required; this stage neither grants nor assumes it. Quota
confirmation remains separate from free-space observation. These are later
deployment/generation prerequisites, not reasons to stop read-only preflight.
No database deployment, synthetic file publication, collaborator message or
GPU benchmark is included in Stage 1. Implementation/fixes use the requested
independent review loop, with at most three rounds.
