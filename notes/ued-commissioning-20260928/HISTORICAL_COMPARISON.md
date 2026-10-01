# Comparison with the July 8 UED experiment-data study

> Historical September 2026 record. For current repeat-test entry points and archived evidence, see [the repeat-test guide](../epixquad/README.md). Process IDs, running-state statements and launch paths below describe that session.

The September standalone FH/FM results and persistent FL mismatch pattern
match the earlier production experiment-data findings. In particular, all
352 persistent FL mismatches are the previously identified UED column pair;
there are zero new persistent coordinates. The additional first-frame FL
mismatches remain a separate startup observation.

## Recovered evidence and mode identification

The July 8, 2026 study is in task `019ee1fd-f7a9-7c31-aae1-d7aea492e4a5`
(“Setup our daq to run with the new test module for epix10ka”). The final
report is at history line 8411, timestamp 2026-07-08T17:48:04.528Z. Its
underlying tool outputs at lines 8383, 8391, 8399 and 8400 record the actual
run results and coordinate intersections. Relevant outputs and the nine
fixed-mode run summaries are preserved in
[history-fixed-gain-evidence.json](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-commissioning-20260928/history-fixed-gain-evidence.json), with the
original history path and line numbers. This comparison recovered that
recorded analysis; it did not reread the historical XTC files.

Modes were identified from each run's **saved Configure data**:

| Mode | asicPixelConfig | trbit | Expected raw top2 |
| --- | --- | --- | --- |
| FH | 0xc | 1 | 0 |
| FM | 0xc | 0 | 0 |
| FL | 0x8 | ignored | 1 |

Today's writer records those same pixel codes/trbit settings. Its saved
software/configuration establishes the intended mode; the raw top bits alone
still cannot distinguish FH from FM. This is consistent with the existing
[write/readout mapping note](../../psdaq/psdaq/debugtools/epixquad1kfps/PIXEL_GAINMODE_WRITE_READOUT.md)
and the fixed-mode selectors in `psana/psana/detector/UtilsEpix10ka.py`.

## Direct gain-bit comparison

All historical samples below checked 50 full raw frames per run.

| Mode | Historical experiment/runs | Historical observation | September standalone |
| --- | --- | --- | --- |
| FH | ued1015980 r98; ued1015999 r470 | All top2=0; zero FP | Same: zero FP in all 50 frames |
| FM | ued1015974 r143; ued1015980 r20; ued1015999 r518/r521 | All top2=0; zero FP | Same: zero FP in all 50 frames |
| FL | ued1015999 r75/r185/r318 | Same 352 FN pixels in every frame | Exactly the same persistent 352; extra first-frame-only mismatches |

The historical FL runs each had 17,600 FN occurrences (352 × 50), with an
intersection and union both equal to 352 pixels across all three runs.
The common coordinates were:

- DAQ raw: segment 2, rows 176–351, columns 289 and 290.
- ASIC programming: ASIC 8, rows 0–175, columns 97 and 98.
- Bank coordinates: ASIC 8, bank 2, rows 0–175, columns 1 and 2.

All coordinates are zero-based. A set comparison against today's saved
[persistent CSV](fl-persistent-mismatch-candidates.csv) gives **352/352
overlap, zero additions, zero missing pixels**. The saved mixed-map
ued1015980 run-2 FN CSV also has **68/68 coordinates inside this same set**.
Thus the old mixed-map FN subset and today's fixed-FL column failures are
spatially consistent.

Today's raw files were decoded again for the comparison. FH/FM each have
27,033,600 top2=0 samples and no other top2 codes across 50 images. FL frames
2–50 have exactly 352 top2=0 pixels per frame; every other pixel is top2=1.
No top2=2 or 3 samples occur in any of these three captures.

## Difference and interpretation

Today's FL frame 1 has 315,612 FN pixels: the known 352 plus 315,260 pixels
that do not mismatch again. Those extra pixels were absent from the recorded
historical fixed-FL samples. Keep their evidence, but do not add them to the
persistent known-bad baseline on this basis.

Today's first-frame raw14 mean is about 13,278 in every mode. Means over
frames 2–50 are FH=2815.711, FM=2775.997, FL=2754.868. The ordering is
consistent with earlier FH/FM/FL pedestal observations, but temperatures,
timing, exposure and calibration context are not normalized here. This is
not a quantitative analog-gain or pedestal-equivalence test. The July 6
pedestal-proximity classifications also changed with the calibration source;
they should not override the Configure-based fixed-mode identification.

The 352 pixels can now be treated as a **historically corroborated bad-gainbit
baseline** for later gain-map accounting, retaining provenance and reporting
both total and outside-baseline mismatch counts. This does not establish
their actual analog gain response or imply that they should be removed from
all image/calibration analysis. A repeat FL write remains useful as part of
the planned repeatability sequence, but is not needed merely to establish
that these columns were already seen in UED data. The startup behavior still
needs its own investigation.

Reproduce the offline comparison with `compare_fixed_gain_history.py` in
the DAQ Python environment and this checkout's psdaq on PYTHONPATH.
[fixed-gain-history-comparison.json](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-commissioning-20260928/fixed-gain-history-comparison.json)
records exact histograms, source checksums and coordinate comparisons. No
detector access or live setting changes were made for this comparison.
