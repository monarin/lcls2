# DAQ clock at 10 and 360 Hz — September 30, 2026

The UED production module produced full images at an internal trigger setting
of 360 Hz with `AsicRoClkT=3`, `AsicRoClkHalfT=1`, and ADC pipeline delay 68.
No historical extra gain-map FP pixels appeared. This is a successful bounded
burst, not a sustained-throughput qualification or proof of the old failure's
cause. No PCIe timing or timeout adjustment was needed for these captures.

## Results

All captures used `drp-ued-cmp003`, `/dev/datadev_0`, lane 0, the pinned test
bundle, MapFHL with the unchanged expand2 map and the DAQ map-writing helper.
The code commit was `6f5698eef`. The explicit clock override was applied after
baseline loading and verified before each phase. First, attempt 11 passed the
10 Hz full-frame and known-bad-list checks. Only then was attempt 12 started.
Attempt 12 loaded the baseline and wrote the map once, then changed only the
camera trigger rate between phases; readout was quiesced/drained between phases.

| Capture | Internal rate | Full images | Camera sequence | Assessed FP / FN per image |
| --- | ---: | ---: | --- | --- |
| Attempt 11, prerequisite | 10 Hz | 51 | 205–255 | 0 / 68 |
| Attempt 12, phase 1 | 10 Hz | 51 | 256–306 | 0 / 68 |
| Attempt 12, phase 2 | 360 Hz | 87 | 307–393 | 0 / 68 |
| Attempt 12, phase 3 | 10 Hz | 51 | 394–444 | 0 / 68 |

Every sequence is consecutive. Camera sequence-counter increments match saved
full-image counts. All four camera line-buffer error counters remained zero,
ADC stayed done/pass, all saved Rogue records had error 0, and native file
framing/decode checks passed. All assessed FN pixels are the same 68 selected
pixels in the production module's 352-pixel persistent list. There are no
assessed FP/FN outside that list and no overlap with the historical extra FP.

First-frame handling follows the agreed policy: preserve every frame but
exclude the first full image of each capture/phase from the separate
assessment. This gives 50, 50, 86 and 50 assessed images. Attempts 11 and
12 phase 1 each have the familiar first-frame 237 FP / 22158 FN transient.
**The first images after changing to 360 Hz and back to 10 Hz already have
0 FP / 68 known FN.** Thus the transient did not recur merely on rate-phase
restart without reprogramming the map in this test.

## Rate and timing interpretation

At 360 Hz, trigger readback was source 3, period 434028 ticks and
359.99981568009434 Hz. The measured camera minimum trigger period also became
434028. Host image-arrival rate over the short file was about 228.16 Hz,
compared with about 10 Hz in each low-rate capture. All 87 generated camera
sequence numbers were saved, so there is no observed image loss in this burst.
The difference is consistent with buffering/recorder overhead; this capture
does not isolate that overhead or establish sustained 360 Hz host throughput.
The collector performs image decoding in the receive callback, which is a
candidate to examine before a longer throughput qualification.

The requested frame count is a stopping threshold, not an exact file size.
Polling plus in-flight records and shutdown drain produced 87 images for the
51-image 360 Hz request. All extra images remain in the file and analysis.
There were nine 48-byte timing-only records at 360 Hz, and 514/513/513 in the
10 Hz captures. The only full-image sizes were 1095248 and 1095288 bytes;
there were **no 1616- or 1656-byte abbreviated image records** like June 29.

PCIe event-builder timeout stayed at 144675 and bypass at 0. Timing rates
remained at the previous profile values. Event-builder per-input timeout
counters are nonzero even in successful captures; they must not be presented
as camera image-loss counts. At 360 Hz, final `DataCnt[1]=87` matches the saved
image count, while final `TimeoutDropCnt[0:3]` values are 70, 9, 96. The complete
before/after snapshots in [results.json](results.json) preserve startup counts
and distinguish these from counter increments. No timeout tuning was attempted.

The historical failure was on rdsrv421 and its exact register state is unknown.
These results therefore establish current UED behavior with the DAQ clock;
they do not prove that clock period 6 was the cause of that failure. This test
also does not validate full DAQ triggering, pedestal quality, or calibration.

## Final state and evidence

Both wrapper capture/restoration exits were zero; every phase has an empty
cleanup-error list. [Postflight](postflight.json) confirms readout off,
source-3 idle triggering restored to 10 Hz, ADC good and frame count 444 stable
across two samples. **Clock period 3 and MapFHL remain loaded.** PCIe stays in
standalone local-timing mode with trigger masters off and blowoff on. Full
reinitialization/Configure is still required before production DAQ use.

[results.json](results.json) records checksums, clock/trigger readbacks,
counters, sequence ranges, gain summaries and record flags. Each capture has
a `*-review.json`, `*-after-first-mismatches.csv`, `*-payloads.json`, lossless
`*-test.json.gz`, and software/console records here. Raw data, input copies,
full mismatch arrays and separate after-first arrays remain in the ignored
`../attempt11-mapfhl/` and `../attempt12-mapfhl/phaseNN/` directories.
[archive_review.py](archive_review.py) reproduces the offline reviews without
hardware access. Launch/execution/restoration logs are archived by attempt
number in the parent directory. The clock-option implementation passed
[26 offline checks](../clock3-offline-tests.log) before hardware execution.

Remaining comparisons include MapFML at the new clock/rate, interleaved
repeated programming, a longer recording-throughput check, and full
DAQ/external-trigger equivalence. The bundle remains a test bundle.
