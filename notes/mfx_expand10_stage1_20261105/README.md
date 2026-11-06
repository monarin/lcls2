# MFX sample10 Stage 1: ten-event scratch fixture

The first 10 original MFX run-10 events are stored with every tenth uint16 pixel
of every physical JF panel. The 32 panels retain their five original JF streams
(s005–s009), panel IDs, detector IDs, frame counters, detector timestamps and
hot-pixel fields. All five ancillary streams are retained. One native step is
truncated and closed with the source Disable/EndStep/EndRun templates; no internal
steps are inserted, and all original event timestamps are unchanged.

Artifact directory (private scratch, not published):

```text
/sdf/scratch/users/m/monarin/gpu-validation/mfx-expand10/20261105-stage1-r2/xtc/
```

Files retain the **source identity** `mfx101555026`, run 10, in this isolated
directory. They are synthetic and must not replace original source files.
No destination run number is allocated. No public files, calibration documents
or existing run-1/2/3/4 datasets were changed.

| Quantity | Result |
|---|---:|
| Events / native steps | 10 / 1 |
| JF streams / total streams | 5 / 10 |
| Bigdata + SMD files | 20 |
| All file bytes | 40,280,528 (40.28 MB) |
| JF seed bytes | 33,554,560 |
| CPU-expanded JF bytes, all events | 335,544,320 |
| Source/seed/expansion panel comparisons | 320 |

## Stored schema

Each original JF `raw` Names record becomes `sim10` version 1.0.0, preserving
NamesId, detector identity and physical segment. Its pixel field becomes
`seed: UINT16[52429]`. Other event fields remain unchanged. Original detector
configuration payloads remain byte-identical. Each JF stream has an additional
`sim10meta.config` record in Configure with factor 10, rows 512, columns 1024,
XTC dtype code 1 (UINT16), and encoding version 1. Metadata segment equals the
stream index. Its node-0 NamesId is checked free in both BD and SMD Configure.

CPU reference: `np.repeat(seed, 10)[:524288].reshape(512, 1024)`. The last stored
word supplies eight output pixels. ADC/gain bits are retained verbatim in each
sampled word; this is a synthetic workload, not lossless image reconstruction.

## Checks and reproduction

`verification.json` records exhaustive output hashes, original payload hashes,
Configure changes, non-JF payload preservation, per-panel scalar preservation,
source/seed equality, independent expansion indexing, original event timestamps,
strictly ordered identical ten-stream timelines, transition state, all SMD
references, complete EOF, and normal serial DataSource traversal of all 320
panel-events. Per-panel gain-state counts are recorded for source and expanded
images. Six unit checks cover word/gain-bit preservation, the partial last group,
invalid seed shape/type, source field rewrite/scalars, missing panels, invalid
extents, and packed timestamp second rollover (some checks cover multiple cases).

```bash
bash notes/mfx_expand10_stage1_20261105/runtime.sh \
  notes/mfx_expand10_stage1_20261105/test_fixture.py

bash notes/mfx_expand10_stage1_20261105/runtime.sh \
  notes/mfx_expand10_stage1_20261105/fixture.py --root /absolute/new/scratch/xtc

bash notes/mfx_expand10_stage1_20261105/runtime.sh \
  notes/mfx_expand10_stage1_20261105/verify.py \
  --root /absolute/new/scratch/xtc --report /absolute/new/scratch/report.json
```

The output directory and report must not exist. The runtime wrapper uses the
same frozen Python3.11 psana installation as the reviewed MFX pilot, sets
`PS_FAKESTEP_FLAG=0` and `PS_PARALLEL=none`, and is recorded in `provenance.json`.
Serial DataSource uses `exp='mfx101555026', run=10, dir=<scratch xtc>, live=False,
skip_calib_load='all'`. There is no `run.Detector('jungfrau').sim10` adapter yet;
the validator inspects decoded event Dgrams after the normal EventBuilder and
bigdata reads. GPU expansion and calibration are **not tested or implemented**
in Stage 1. The full frozen MFX calibration manifest remains the reference for
Stage 2, including `(3,32,512,1024)` gain-dependent arrays. No compressed/expanded
calibration constants are created.

The initial `20261105-stage1-r1` attempt stopped before completing stream 0 due
to an overly strict source BD/SMD header comparison. Source timestamps and env
must agree, while other BD/SMD header fields can legitimately differ. The writer
now compares source identity and the validator separately requires each output
header's env/src/damage/type to match its own source. That failed attempt/log
remain in scratch; accepted output is only `stage1-r2`. The first verification
passed; a second pass tightened the metadata-schema assertion without rewriting
the data. Both logs are retained in scratch.

Plan: [five-stream MFX expansion](../../psana/psana/gpu/docs/proposals/synthetic_mfx_expand10.md).
Independent review records are under `review-01/`. Large arrays are not saved;
the expanded reference is regenerable from the retained seed files.
