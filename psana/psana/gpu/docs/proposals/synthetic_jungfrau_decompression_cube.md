# Synthetic JF 1M input for expansion and fly-scan cubes

**Superseded for future work:** use the [MFX five-stream 10x expansion plan](synthetic_mfx_expand10.md). The completed XPP stages below remain historical evidence; further XPP fixture generation/publication is paused by the November 5 direction change.

Status: Stage 2 complete, November 5, 2026, with independent Claude review
passing at `30305b95b`. Run 3 exists privately (2,400 events, one native step), and eight
experiment-only calibration types are deployed for runs 3–4. Neither run 3 nor
run 4 is published in final `xtc/`. This replaces the conversational proposal to
use XPP run 216 and repeat its native steps.

## Source and destination

Use `xpp101605526`, run **311**, detector `jungfrau1M`, from:

```text
/sdf/data/lcls/ds/xpp/xpp101605526/xtc/
```

The candidate inspection counted 151,152 L1Accept records and one BeginStep in
stream 0. All seven cataloged bigdata files were present with matching sizes.
Configure and a first-event read identified two Jungfrau modules, each storing
`uint16 (1, 512, 1024)`: 1,048,576 pixels total. These are recorded-event counts,
not a completed detector damage/availability audit. Stage 1 below rechecks and
archives the evidence before generation.

| Destination | Events | Steps | JF pixel bytes, excluding overhead |
| --- | ---: | ---: | ---: |
| `prjdatpsana`, run 3 | First 2,400 source events | 1 | 5,033,164,800 (5.03 GB) |
| `prjdatpsana`, run 4 | 400,000 | 1 | 838,860,800,000 (839 GB) |

Run 3 is a prefix of the source's single step; 2,400 is a small-fixture size,
not a claim that run 311 has 2,400-event native steps. Run 4 uses two complete
151,152-event source sequences followed by the first 97,696 events. It repeats
source payloads rather than requiring 400,000 distinct experimental exposures.

Store only the original JF 1M pixel payload per event. Retain timing, existing
step metadata and selected useful small scalar detectors. Exclude unrelated
camera/ePix image payloads. Inventory mixed streams and field dependencies
before deciding which streams can be copied and which require repacking.
Record the exact file layout and total byte estimate before writing.

## One-step timing and transition contract

Each output run has one Configure/BeginRun and one BeginStep/EndStep pair,
with a clean enabled acquisition interval and final Disable/EndRun. Reuse
each stream's source transition templates and preserve required control/XTC
fields while updating identity, timestamps and applicable payloads.

There are **no transitions at source-repetition boundaries** in run 4. Do not
copy the source's closing/opening run or step transitions between repetitions.
Retain relevant SlowUpdates in their original order within each copied interval;
shift their datagram timestamps with that interval. Source preflight must check
for any additional Enable/Disable intervals and define their treatment before
the writer is finalized.

Preserve source event timestamps in the first interval where possible. Use a
shared integer-nanosecond offset for subsequent intervals, then repack the XTC
seconds/nanoseconds representation. Ensure every output event is unique and
strictly ordered, all streams agree, and each event is inside the enabled step.
Choose closing-transition timestamps after the retained interval using the
verified source templates and available spacing. Truncating a run requires
closing transitions even though no internal artificial steps are introduced.

Rewrite experiment/run identity consistently, rebuild all SMD offsets/chunk
references, and record output-to-source event/interval provenance. Embedded
detector frame counters/timestamps remain source payload values; distinguish
them from the new outer datagram event identity.

## Calibration: destination validity 3–4

Freeze the constants that normal psana selects for **source run 311**, including
any detector-shared fallback selection. Do not reuse the earlier run-216
calibration proposal or extend the MFX records for destination runs 1–2.

Deploy only the required selected constants to the destination experiment DB,
with validity **3–4** and detector identity/collection mapping verified against
the generated Configure. Preserve source/shared DBs and existing run-1/2
records. Record source/destination IDs, metadata, payload hashes and a deployment
journal. Avoid deploying irrelevant constants for retained scalar detectors.

Build fresh destination documents from the frozen payloads and their encoding
metadata (`ctype`, `dtype`, applicable `data_*` fields), with experiment
`prjdatpsana`, run 3 and run_end 4. Let the destination assign new document/data
IDs. Do not copy source DB object IDs, reference fields, or the transient
`tsec_id`/`tstamp_id` selection fields. Preserve source IDs and payload hashes
only as clearly named provenance. Verify the detector identity independently.

Verify normal destination lookup for both runs 3 and 4, including payload
equality and validity-range selection. Recheck run-1/2 selection remains
unchanged. Constants represent the stored **1M** detector. The GPU simulation
will expand matching constants consistently with the pixels when needed;
synthetic 16M constants are not deployed to the calibration DB.

## Expansion and fly-scan benchmark contract

The stored Configure describes the real 1M payload. A fixture manifest records
the proposed simulation contract `repeat16-v1`: two input panels expand to 32
output panels, with output panel `p` taking input panel `p % 2`. This is a
simulation of reduced input volume and GPU expansion work, not a real codec or
evidence of achievable 16:1 compression.

The future GPU task reads 1M, physically materializes 16M, optionally calibrates
according to the benchmark's stated scope, and accumulates into persistent
fly-scan bins. Bin identity is independent of the single DAQ step. Initially
use a deterministic, interleaved 30-bin mapping from the logical output event
index, with provenance sufficient to reproduce it. This is a synthetic binning
policy, not a physical scan coordinate. The implementation must supply aligned
bin identity through a supported task input; it must not depend on BD-local
iteration counts or an undeclared scan-payload API.

Retain sums across the whole run and export at EndRun, with explicit stream
ownership and final merging. Do not substitute a broadcast view, compute only
1M and multiply, or transfer full images to the CPU per event. The managed
persistent-state lifecycle remains proposed; this dataset plan does not claim
that cube accumulation or decompression is implemented.

At an assumed aggregate 35 kHz, 400,000 events provide 11.43 seconds total.
Reserve up to 50,000 events for warmup and target 350,000 timed events (10 s),
extending warmup/test length if measurements do not stabilize. Exclude warmup
from timed sums/counts or reset the benchmark state safely before measurement.
Report actual warmup, GPU/BD topology, physical reads/cache state, input bytes,
execution batch sizes, expansion/calibration scope and aggregate export cost.

## Staged execution

Stage 1 completed on November 5, 2026, with independent review passing at
`786938d79`. All 151,152 source events have both JF segments and zero XTC
damage. Exact output sizes are 5,039,830,328 bytes (run 3) and 839,913,818,464
bytes (run 4). Calibration snapshots are frozen privately; no destination
constants or synthetic XTC were written during Stage 1. See the
[Stage 1 results](../../../../../notes/synthetic_xpp_stage1_20261105/README.md)
for manifests, validation, private archive and review records. Project quota
confirmation and source-sharing authorization remain later-stage prerequisites.

Stage 2 generated and verified the 5,039,830,328-byte private run-3 pilot.
Normal calibrated DataSource passes for all 2,400 events and six source-matched
raw/calibrated samples, plus an eight-event repetition/second-rollover fixture.
All eight calibration types resolve correctly for runs 3 and 4. Existing runs
1–2, source files and source/shared calibration metadata are unchanged. See
[Stage 2 results](../../../../../notes/synthetic_xpp_stage2_20261105/README.md)
for private location, journals, acceptance and review records. Final independent review passed at `30305b95b`; publication remains Stage 3
and run-4 generation Stage 4.

1. **Read-only source preflight.** Freeze file/configuration/transition inventory,
   exact source-event mapping, selected detector fields and calibration manifest.
   Audit missing/damaged JF frames and inspect representative ADC/gain/mask/
   threshold distributions. Resolve any source-quality issue explicitly rather
   than silently changing counts. Compute exact pilot/full sizes and verify
   destination collisions, capacity and relevant quota information.
2. **Calibration and run-3 pilot.** Implement/review the scoped writer,
   experiment-only calibration deployment and validators. Deploy validity 3–4
   with journal/readback. Generate run 3 in private staging. Verify every retained
   payload against its source, all stream timelines, the one-step state machine,
   schemas, identities, offsets and EOF. Run normal DataSource acceptance for
   exactly one step/2,400 events and compare representative raw/calibrated
   samples to source run 311. Validate a small repeat/truncation fixture as well,
   because a prefix-only pilot does not exercise run-4 repetition boundaries.
3. **Publish and inspect run 3.** Publish only after required checks and review
   pass, using no-overwrite operations and a journal. Check final-path reads,
   calibration and collaborator access. Use explicit `dir=` and `live=False`
   until registration is separately arranged; set `PS_FAKESTEP_FLAG=0`.
4. **Run 4 after pilot acceptance.** Generate the 400,000-event single-step
   fixture, validate repetition seams and all output records/offsets, and run
   full one-step DataSource counting with representative calibrated samples
   including both seams and the tail. Verify unchanged run-1/2/3 artifacts and
   calibration selection, then publish with the same gates and final readback.
5. **GPU workload and performance.** Implement expansion and persistent fly-bin
   accumulation in separately reviewable scope. Validate a small known mapping
   and sums/counts before measuring throughput. Scientific equivalence to an
   experimental cube is not required; skipped work, incorrect bin assignment,
   lost events or invalid numerical output must still be detected. Use run 216
   later for a separate step-transition overhead comparison if useful.

Use the session's implementation → independent Claude review → response/fix →
re-review loop for executable changes. Preserve exact reviewed commits and
compact evidence alongside the existing synthetic-data records. The original revision
was planning-only; subsequent explicit Stage 1 and Stage 2 instructions authorized
the completed preflight, scoped calibration deployment and private generation.

## Sharing and scope

The XPP source is private and its observed ACLs exclude Seema. Read-only source
preparation can proceed, but authorization to share its payloads and constants
through `prjdatpsana` must be established before collaborator-accessible
publication/deployment. A renamed or repeated copy is still derived from the
private source. No permission changes or external messages are part of this
plan revision. The later explicit Stage 2 instruction authorizes the scoped
calibration deployment and private generation; event-payload publication remains
a separate Stage 3 action.

The completed MFX runs 1–2 and their historical plans/results remain unchanged.
See [persistent task state and XPP cubes](persistent_task_state_xpp_cube.md)
for the lifecycle proposal and prior fixture evidence.
