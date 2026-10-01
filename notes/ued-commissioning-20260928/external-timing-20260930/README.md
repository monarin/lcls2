# External UED timing: attempt 17

Captured September 30, 2026, about 21:06 PDT on drp-ued-cmp003, after Mona
powered on the detector and authorized capture. Operator enabled groupca Run
after READY. The same expand2 map and DAQ gain writer were used, with
AsicRoClkT=3 and AsicSyncInjEn=True. Camera auto triggering was disabled;
external timing used XPM0/group0 and the DAQ run/readout routing.

**No extra FP/FN appeared.** Of 490 full images, 489 were assessed after
excluding the first by the established policy. Every image, including the
first, had zero FP and exactly 68 FN, all in the 352-pixel known-bad list.
None of the 43 extra FP coordinates from full DAQ run 3 reappeared.

| Acquisition | Assessed images | FP outside known-bad | FN outside known-bad |
| --- | ---: | ---: | ---: |
| Internal 360 Hz, SYNC=True, attempt 15 | 86 | 0 | 0 |
| External UED timing, SYNC=True, attempt 17 | 489 | 0 | 0 |
| Full DAQ, 360 Hz, run 3 | 4627 | 43 | 0 |

Recorded timing timestamps give **359.959 Hz** over 1.35849 seconds. Camera
sequence numbers run from 16576 to 18043 with exactly +3 for all 489 steps,
consistent with the 1080-Hz camera run trigger and 360-Hz readout. There are
no camera sequence gaps beyond that expected selection. Timing link stayed
up with zero recorded CRC/decode errors, and all four LineBufErr counters
were zero. Two 104-byte timing-only records follow the full images; their
absence of camera data is retained in the record histogram, not counted as
good images. EventBuilder reported two drops on input 1.

This shows that external DAQ timing, at these settings and with quiescent
standalone map programming, was insufficient to reproduce the 43 FPs. It
does not establish equivalence to the entire DAQ Configure/start sequence.
That sequence, including trigger/readout state during map programming and
remaining baseline differences, remains a candidate for investigation.

## Recorder failure and offline recovery

The original capture is correctly retained as **failed**: its old size
allowlist rejected the new 1,095,344-byte external packets and stopped at
the 512-MiB byte limit, rather than at 51 recognized images. Cleanup and
idle restoration succeeded. The file contains 490 complete camera records
and two timing-only records, with no truncated Rogue records or Rogue error
flags. The original zero-valid-image/zero-mismatch summary must not be used
as a gain-bit result.

The recovered review follows EventBatcher.hh and EpixQuad.cc:

- Batcher v1 header is 8 bytes. TDEST 0 timing is 32 bytes at offset 8.
- TDEST 2 camera is 1,095,232 bytes at offset 48; image words start at 80.
- TDEST 3 metadata is 48 bytes at offset 1,095,288, followed by its tail.

The old length-only formula would incorrectly skip 136 bytes because it
mistakes trailing metadata for an extra prefix. The decoder now parses the
external batcher tails, locates TDEST 2 and skips its 32-byte camera header.
It rejects malformed framing, missing/short camera data and EOFE flags.
The external profile and offline reviewers now recognize this packet size;
the two existing internal packet formats retain their decoding behavior.

Offline tests in daq_20250402_r9 on drp-ued-cmp003: **50 passed** across
test_epixquad_external_payload.py, test_epixquad_standalone.py,
test_epixquad_capture_writers.py and test_epixquad_external_timing.py.
The new tests exercise trailing metadata, reordered subframes, independent
panel-coordinate expectations, framing corruption and live-counter filtering.
An initial local run lacked SUBMODULEDIR and failed the existing profile
resolution test; the complete run in the sourced DAQ environment passed.
The corrected recorder has not been used for another hardware capture.

## Evidence and replay

[summary.json](summary.json) records timing and exact run-3 FP comparison.
[review.json](review.json) contains all per-image gain counts and source hashes.
[after-first-mismatches.csv](after-first-mismatches.csv) preserves all 68
remaining FN coordinates. The NPZ files retain all-frame and assessed counts.
Original metadata, profiles, YAMLs and logs are compressed here; original
capture status and mismatch artifacts in attempt17-mapfhl were not changed.

Native file: `../attempt17-mapfhl/capture.dat` (536,722,704 bytes), SHA256
`0b846238c4d315e153665f495ad6a2aa76ffb4b639c620a9b6aa2762c5faca46`.
It remains on the shared filesystem, excluded from Git.

From the sourced test environment, with source psdaq prepended to PYTHONPATH:

```bash
python notes/ued-commissioning-20260928/review_gain_map.py \
  notes/ued-commissioning-20260928/attempt17-mapfhl \
  /tmp/ued-attempt17-new-review --redecode-failed
python notes/ued-commissioning-20260928/external-timing-20260930/archive_review.py
```

The review output must be a new directory. The explicit redecode flag permits
the failed capture's old mismatch arrays to differ, records that difference,
and writes only a separate review. The archiver verifies native/test/map
hashes and saves timing evidence and comparison with the 43 run-3 FPs.

## Stopped state

Capture exited 1 (byte limit); restoration exited 0. Readbacks restored
TrigSrcSel=3, AutoTrigPer=15625000, AutoTrigEn=True, TrigEn=True and
**RdoutEn=False**. ADC calibration was done/pass. The operator was told to
uncheck groupca Run; the script does not write XPM PVs. PCIe retains external
test configuration; full DAQ must configure again before use. No ConfigDB
changes were made. Another hardware capture is not needed for this result.
