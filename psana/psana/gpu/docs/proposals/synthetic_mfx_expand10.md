# MFX five-stream input with synthetic 10x GPU expansion

Status: proposed, November 5, 2026. This is the active dataset direction.
Stop further real-codec exploration and the XPP repeat16 fixture path. Preserve
those results and existing datasets; this plan does not generate or publish data.

## Layout and calibration

Use public `mfx101555026` run 10, the same source as prjdatpsana runs 1–2:
`/sdf/data/lcls/ds/prj/public01/xtc/`. Keep the five original JF DRP streams and
all 32 physical panels, with their event payload order:

| Stream | Physical panel IDs |
|---|---|
| s005 | 3, 7, 0, 1, 2 |
| s006 | 19, 15, 11, 31, 27, 23 |
| s007 | 18, 14, 10, 6, 30, 26, 22 |
| s008 | 16, 12, 8, 4, 28, 24, 20 |
| s009 | 17, 13, 9, 5, 29, 25, 21 |

These are five JF-bearing files, not five equal-sized pieces and not a change
to detector segmentation. Retain the five ancillary source streams for the
pilot, including timing and small detectors; account for their bytes separately.
Validate actual Configure membership and L1Accept child order before generation.

Keep the frozen MFX calibration payloads and physical panel indexing unchanged.
The image layout is `(32, 512, 1024)`; gain-dependent constants retain their
actual `(3, 32, 512, 1024)` layout, dtype and values. Geometry and other constant
types retain their original representations. Do not shrink, tile or interpolate
constants. Preserve the source detector identity; verify that the new synthetic
algorithm still resolves the intended calibration collection.

Runs 1–2 remain unchanged. Runs 3–4 already have XPP constants and are not to be
repurposed. Initially use private scratch and frozen MFX constants. Propose new
runs 5 (pilot) and 6 (large) only after checking collisions and choosing their
calibration validity range. Before public use, verify normal DataSource lookup
against those frozen bytes; do not rely on shared-DB fallback or assume run-1/2
validity covers the new runs. No DB changes are part of this planning task.

## Stored representation: sample10-v1

For each original panel and event, flatten the raw uint16 image in C order and
store `raw[::10]`. This retains 52,429 complete words per panel, including their
ADC and gain bits. Expand each stored word into ten adjacent output words,
trimming the final group to exactly 524,288 pixels, then reshape to `(512,1024)`.
The last group has eight pixels. Sampling never crosses a physical panel.

Use an explicitly synthetic XTC algorithm, proposed `sim10` version `1.0.0`,
with a rank-one UINT16 `seed` field per original panel Names record. Preserve
physical segment identity in Configure and the corresponding NamesId references
in event ShapesData. Record encoding version, factor 10, original shape and
original dtype as validated configuration metadata. The final schema and its
NamesId allocation require a writer/reader preflight. Do not label the short
array as ordinary `raw` or claim that its stored shape is a full panel.

Rebuild Configure/ShapesData extents, event sizes, and all SMD offsets/sizes.
Preserve event identity, damage and transitions consistently across all streams;
retain the existing source steps and do not insert artificial scan steps.
A truncated pilot must still close its enabled interval, step and run correctly.

This is deterministic synthetic expansion, not reversible compression of the
original image. Words are taken from the same event and panel, which supplies
realistic ADC/gain values, but spatial correlations and pixel-to-pedestal pairing
change. Check distributions and finite calibrated output on a pilot. Scientific
agreement with the original event is not an acceptance criterion.

## GPU work and reference

Conceptual pseudocode (not a claim that these APIs already exist):

```python
# Writer, independently for each physical panel
seed = original_raw.reshape(-1)[::10].copy()

# GPU, for every event and all physical panels
raw[i] = seed[i // 10]                 # i = 0 .. 512*1024-1
calib = calibrate(raw, full_constants) # original physical panel indexing
cube[global_event_index % nbin] += calib
```

Read the smaller payloads from all five JF streams. Materialize a full contiguous
uint16 GPU image for all 32 panels, then calibrate every pixel to float32 and
accumulate fly-scan bins. Do not substitute a broadcast view or compute only the
seeds and multiply the result. Bins use a recorded global output-event index,
not a BD-local counter, and remain independent of native step boundaries.

The current user calibration example expects dense uint16
`(events, segments, rows, columns)` input and full physical constants. The
synthetic input schema needs an explicit reader/task adapter and an expansion
kernel before that calibration arithmetic can be reused. Neither the adapter
nor run-persistent cube state is implemented by this plan. Follow the
[persistent task proposal](persistent_task_state_xpp_cube.md) for ownership and
final merging; retain sums through steps and export once at EndRun.

Keep compressed input buffers alive until expansion completes; retain expanded
raw through calibration and calibrated output through cube accumulation. Tie
slot reuse to CUDA completion. Budget seed, full raw, calibrated output,
constants and cube buffers across all BDs sharing a GPU. A float32 full-image
cube bin costs 67,108,864 bytes; 30 bins cost 2.013 GB per independent accumulator,
before counts and scratch. Persistent buffers and user output allocations need
explicit accounting, not just the framework input-slot budget.

For correctness, build a CPU reference with `repeat(seed, 10)[:524288]` for each
panel, then apply the identical calibration options and constants. Require exact
CPU/GPU expanded raw equality and the established calibration comparison. Check
cube sums/counts with a documented floating-point accumulation tolerance, correct
event/bin identity and no lost events, including across step boundaries. Compare
against the expanded synthetic image, not the original source image.

## Size and target rate

| Item | JF payload only |
|---|---:|
| Stored per panel | 104,858 bytes |
| Stored per event, all five streams | 3,355,456 bytes (3.355 MB) |
| Expanded raw per event | 33,554,432 bytes (33.554 MB) |
| Calibrated float32 per event | 67,108,864 bytes (67.109 MB) |
| 1,000-event pilot | 3.355 GB |
| 400,000-event fixture | 1.342 TB |

The payload reduction is 9.99996x rather than exactly 10x because panel sizes
are not divisible by ten. Metadata, transitions, SMD and ancillary streams add
to these numbers; total files will not shrink by exactly 10x. Calculate actual
bytes and quota headroom before generation, including staging/publication copies.
At 35,000 events/s, the JF seeds alone require **117.44 GB/s aggregate input**.
Expansion writes 1.174 TB/s of raw output and calibration produces 2.349 TB/s
of float32 output, in addition to reads/cube traffic. These are workload byte
rates, not measured throughput or a promise that a given GPU count can sustain it.

## Stages

1. **Preflight and tiny fixture.** Freeze source/calibration manifests, define
   the synthetic schema, verify panel order and size math, and implement/review
   a roughly 10-event scratch writer plus CPU expansion reference. Validate all
   stream timelines, Names mappings, SMD offsets, seed/source equality and EOF.
2. **GPU correctness pilot.** Implement/review reader adapter, materialized
   expansion and calibration integration. Compare every pixel for the tiny
   fixture, then scale to approximately 1,000 events. Include a separate native
   step-boundary sample and exercise stream/batch buffer reuse. Measure actual
   files and memory before scaling.
3. **Cube and performance.** Implement/review persistent sums and deterministic
   bin assignment; validate against CPU. Prepare 400,000 events only after the
   pilot and capacity checks. Source run 10 has 36,037 events, so the large
   fixture must repeat source intervals with unique, aligned output timestamps;
   specify and test the transition/repetition seams before writing it. Use the
   existing integer-nanosecond timestamp machinery, preserve valid source step
   structure, and record output-to-source mapping. At 35 kHz, 400k events give
   11.43 s, including a tentative 50k warmup and 350k timed events. Report actual
   topology, warmup, GDS/cache state, bytes read and final cube export/merge cost.

Use the session's implementation/review loop at each stage. Publication, new
run assignment and calibration deployment follow their own concrete preflight;
this revision authorizes planning only, with no production DRP/codec changes.

Evidence: [five-stream codec probe](../../../../../notes/mfx_real_codec_20261105/README.md),
[raw size probe](../../../../../notes/raw_compression_factor_20261105/README.md),
[frozen calibration manifest](../../../../../notes/synthetic_mfx_stage1_20261103/manifest.json),
and [current calibration example](../../examples/jungfrau_calibration.py).
