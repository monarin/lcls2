# FM / FL 50-frame tests — 2026-09-28 PDT

Historical cross-check: [the July 8 experiment-data study](HISTORICAL_COMPARISON.md)
already found these exact 352 FL pixels in three UED runs. FH/FM also match
that study. The extra first-frame FL mismatches remain a separate observation.

Both authorized captures completed on drp-ued-cmp003 at 10 Hz with the same
pinned bundle, UED baseline, normal PROM check and all 16 ASICs used for FH.
No DAQ or devGui was started. FM passed the gain-bit check; FL has a startup
transient and two persistent mismatching columns. FL is not a clean gain-bit
pass, despite successful acquisition and shutdown.

| Mode / attempt | Full images | FP occurrences | FN occurrences | Unique mismatch pixels | Cleanup |
| --- | ---: | ---: | ---: | ---: | --- |
| FM / 03 | 50 | 0 | 0 | 0 | Clean |
| FL / 04 | 50 | 0 | 332,860 | 315,612 | Clean |

Each file contains 41 image payloads of 1,095,248 bytes, 9 of 1,095,288 bytes,
and 503 timing-only 48-byte records. All Rogue error flags are zero. Native
file framing is complete. Each raw file is 54,791,328 bytes. See the
[machine-readable results](fm-fl-results.json), [FM record](test-attempt03.json)
and [FL record](test-attempt04.json).

## FL mismatch evidence

Offline frame-by-frame analysis reproduces the saved FP/FN arrays exactly:

- Frame 1 has 315,612 FN pixels.
- Frames 2–50 each have the same 352 FN pixels.
- 315,260 pixels mismatch only in frame 1.
- 352 pixels mismatch in all 50 frames: **ASIC 8, programming columns 97 and
  98, rows 0–175**. In DAQ raw coordinates these are segment 2, columns 289
  and 290, rows 176–351. All coordinates are zero-based.

For those two columns, the expected low-gain bit is 1 but the observed bit is
0. The [persistent candidate CSV](fl-persistent-mismatch-candidates.csv)
contains the 352 coordinates, 50/50 occurrence counts and fractions. Source
and subset checksums, selection rule and source capture are in
[fm-fl-results.json](fm-fl-results.json). This is evidence of reproducibility
within one capture, not proof of permanent bad pixels or of the cause.

The original 315,612-row candidate CSV, candidate mask and provenance remain
under `attempt04-fl/mismatch-records/`; they include every first-frame
mismatch. No candidates were suppressed from the reported totals. Tracked
[FL metadata](mismatch-attempt04-metadata.json) records their checksums.
FM has an empty CSV/mask and [zero-candidate metadata](mismatch-attempt03-metadata.json).

The [per-frame report](attempt04-frame-mismatches.json) includes the file record
index, full-image index and counts for each ASIC. Reproduce it offline with
`review_frame_mismatches.py ATTEMPT OUTPUT_JSON` in the recorded DAQ Python
environment and this checkout's psdaq on PYTHONPATH.

## ADC startup behavior and next test

Both FM and FL again have first-frame segment means around 13.2–13.3k ADC,
followed by roughly 2.6–2.9k. This resembles the FH startup behavior. No ADC
pixels were zero. Per-frame ADC statistics are retained in
[FM payload review](attempt03-payload-review.json) and
[FL payload review](attempt04-payload-review.json). Gain bits alone cannot
distinguish FH from FM or validate analog gain/pedestal quality.

The historical comparison now corroborates ASIC 8 columns 97/98 as an existing
bad-gainbit baseline. Carry those 352 pixels into gain-map accounting with
mode/source provenance and retain both total and outside-baseline counts.
A repeat FL capture remains useful for write repeatability; investigate startup
behavior separately before accepting gain-map results.
Do not promote all first-frame-only candidates to permanent exclusions.
No extra live repeat, gain-map test, ADC training or PROM programming was run.

## Final hardware state

Both wrappers restored the prior idle camera settings with verified identity:
TrigSrcSel=3, TrigEn=true, AutoTrigEn=true, period=15625000 (10 Hz), readout off.
ADC remained done/pass. The independent [postflight](postflight-fm-fl.json)
shows frame count 200, unchanged across two samples, and image rate zero.
The existing monitor IOC remains running.

The final loaded map is FL and the UED acquisition baseline remains loaded.
PCIe remains in standalone local timing configuration; sampled timing master
enables are off and EventBuilder blowoff is on. Returning to production DAQ
requires full detector initialization plus Configure and readback verification,
as described in the [restoration review](../ued-settings-20260928/README.md).

Raw data, copied inputs and complete mismatch artifacts remain under
`attempt03-fm/` and `attempt04-fl/`, excluded from Git. Test records, software
manifests, console/launch logs, restoration readbacks, summary JSON and the
persistent candidate CSV are retained in Git. The bundle remains in test
status; gain-map validation and superproject promotion are still pending.
