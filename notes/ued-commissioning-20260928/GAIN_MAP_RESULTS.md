# Production UED gain-map comparison — 2026-09-28 PDT

The full historical expand2 pattern was captured on drp-ued-cmp003 in both
MapFML (attempt 05) and MapFHL (attempt 06). **Neither capture has FP/FN
outside the production persistent list after the first image.** This initial
pair did not reproduce the historical production-module edge-row FP pattern.

## Conditions and accounting

The unchanged map selects 22,169 FL pixels, including 68 of the 352 known
bad-gainbit pixels. It is the expand2 map used for ued1015980 runs 2/3/4 and
the rdsrv421 comparison. Its exact input is archived as
[gain-map-expand2.npy.gz](gain-map-expand2.npy.gz), with uncompressed SHA256
`1739094ffe6e80a7eae0496961730399c0f1dce077f7426e445137fac4434894`.
[Input review](gain-map-input-review.json) records source paths, selected count,
coordinate round-trip validation and the historical CSV checksum.

Each attempt recorded 51 complete images at 10 Hz. Per Mona's instruction,
**only frame 1 is excluded from acceptance**, leaving 50 assessed images.
All data and original all-frame FP/FN counts remain preserved. No low-gain
pixels were removed from the commanded map for this comparison.

The write path is the current standalone implementation: load the saved UED
baseline, set trbit, fill each ASIC with the H/M background, then individually
write the selected FL pixels. It does not write every background pixel
individually, and this run does not establish equivalence to the complete
historical DAQ configuration/write sequence. Injection is disabled; normal
PROM checking and ADC done/pass gates remain enabled.

## Findings

Counts below apply to **each of frames 2–51**, with the same coordinates
throughout both captures:

| Mode | FP | FN | FP outside persistent list | FN outside persistent list |
| --- | ---: | ---: | ---: | ---: |
| MapFML | 0 | 68 | 0 | 0 |
| MapFHL | 0 | 68 | 0 | 0 |

Both have 3,400 assessed FN occurrences (68 × 50), all on ASIC 8 columns
97/98 where the unchanged map requests FL. These exactly match the 68 FN
coordinates in the historical production run-2 CSV. The 50-frame FP/FN arrays
are identical between the two current modes.

The first image in each capture has 237 FP and 22,158 FN pixels, including
the known 68. Across all 51 images, the unfiltered records report 237 FP
occurrences and 25,558 FN occurrences. The 237 FP and 22,090 FN outside the
persistent list occur only in frame 1 and are retained but excluded from
acceptance under the stated policy.

Every accepted full image decoded, every saved record had Rogue error=0,
native file framing was complete, both shutdowns were clean, and both camera
idle restorations succeeded. See [machine-readable results](gain-map-results.json),
[MapFML frame-by-frame review](gain-map-attempt05-review.json) and
[MapFHL frame-by-frame review](gain-map-attempt06-review.json).

## Comparison with task history

The saved production analysis of **ued1015980 runs 2/3/4, MapFML**, found
43 FP pixels on raw rows 175/176 and 68 FN pixels with this expand2 map. The
43 FPs are outside the current persistent list. None recur in either current
capture's 50 assessed frames. The actual historical
[run-2 mismatch table](historical-expand2-run2-mismatches.csv) is preserved for
coordinate comparisons.

On the rdsrv421 test module, the corrected expand2 MapFHL/MapFML comparisons
had one known bad FP and 22 known bad FN pixels, with no additional production
edge-row FPs. These are different module-specific bad coordinates; its mask
was not applied to the production data. The original task excerpts and the
user's MapFHL/MapFML labeling correction are in
[gain-map-history-evidence.json](gain-map-history-evidence.json).

Today's production result therefore resembles the rdsrv421 outcome **after
each module's own known-bad accounting**, despite the earlier production data
having additional FPs. It does not prove that the historical problem is fixed
or identify its cause. Software/configuration versions, programming order,
configuration history and timing differ from the historical DAQ session.
This is one write/capture per mode, not a repeatability study.

The next useful tests are repeated identical writes, then the larger expand4
pattern and a controlled programming-path/order comparison if the historical
FPs still do not appear. Neighbor-removal tests need a reproducible FP target.
Keep analog response and full production DAQ acceptance separate from these
gain-bit checks. No bundle promotion or additional live tests were performed.

## Final state and retained evidence

Independent [postflight](postflight-gain-map.json) confirms readout off, the
prior source-3 internal idle triggering at 10 Hz restored, ADC done/pass, and
frame count 302 stable across two samples with rate zero. The monitor IOC
remains running. The last loaded map is MapFHL; PCIe remains in local timing
test configuration with trigger masters stopped. Full detector initialization
plus Configure/readback verification is required before returning to DAQ.

- Raw data/copied maps/all-frame mismatch masks: `attempt05-mapfml/` and
  `attempt06-mapfhl/` on the shared workspace (excluded from Git).
- Full test records including selected-pixel lists:
  [attempt 05](test-attempt05.json.gz), [attempt 06](test-attempt06.json.gz).
  These are lossless gzip copies of the original JSON files.
- Assessed coordinate CSVs: [MapFML](gain-map-attempt05-mismatches.csv),
  [MapFHL](gain-map-attempt06-mismatches.csv). Within-baseline and historical
  membership are explicit fields, not deleted rows.
- Original software manifests, console/launch logs and restoration readbacks
  are retained by attempt number. Data/map/review checksums are in the JSON
  records.
- `review_gain_map.py` verifies its all-frame counts against the recorder,
  checks detector identity against the production baseline, and separately
  exports post-first-frame counts. Verification used the earlier real FL
  capture plus a synthetic wire-format capture with a first-frame-only FP,
  a persistent outside-baseline FP and a persistent known FN; see
  [validation](gain-map-review-validation.txt).

The first launch failed before hardware access because the original map path
was not mounted on drp-ued-cmp003. The retry used a checksum-identical shared
copy; the initial launch log is retained as `launch-attempt05-map-unavailable.log`.
