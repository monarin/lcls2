# DAQ map-helper comparison — 2026-09-30 PDT

The first DAQ-writer expand2 capture passed in both MapFML and MapFHL at
10 Hz internal triggering. **Both assessed mismatch arrays exactly match
their standalone controls.** None of the historical 43 extra FP pixels
recurred. This comparison used the DAQ map helper inside the standalone
recorder; it did not run the full DAQ or its complete Configure sequence.

## Controlled conditions

Attempts 09/10 used `--gain-writer daq --rate-hz 10 --frames 51` on
`drp-ued-cmp003`, `/dev/datadev_0`, PGP4 lane 0. The loaded function was
`psdaq.configdb.epixquad1kfps_config._apply_raw_pixel_map_program` from this
checkout, with source-file SHA256
`5a3347c509be6031df5a1f249a4ceb42efd50ea2c7d4214ed39253d8a405e9c7`.
Source path/hash verification passed before hardware startup.

The unchanged expand2 map selects 22,169 FL pixels. Its checksum, both YAML
input checksums, all capture-script hashes, all test-bundle submodule commits
and local tracked diffs, and the bundle patch-script checksum match the
September 30 standalone controls (07/08). The checkout commit at capture was
`88e9f0777`; changes since the controls were result documentation only.

Preflight confirmed the same camera identity, ADC done/pass, readout off,
source-3 10 Hz idle settings and stable frame count 102. The monitor IOC
remained running, with no competing detector DRP, GUI or capture observed.
See [preflight](preflight-daq-writer-20260930.json) and its
[process/readback log](preflight-daq-writer-20260930.log).

## Findings

Each capture recorded 51 full images. Per the agreed policy, frame 1 remains
in the data and all-frame mismatch exports but is excluded from the separate
50-image assessment.

| DAQ attempt | Mode | Standalone control | FP per assessed frame | FN per assessed frame | FP/FN outside persistent list |
| --- | --- | --- | --- | --- | --- |
| 09 | MapFML | 07 | 0 | 68 | 0 / 0 |
| 10 | MapFHL | 08 | 0 | 68 | 0 / 0 |

Every frame from 2 through 51 has exactly these counts. All 68 FN pixels are
within the production module's 352-pixel persistent list, at the selected
positions on ASIC 8 columns 97/98. Each capture has 3,400 assessed FN
occurrences. Full assessed FP/FN arrays and known-bad masks match the controls
element by element. Historical FP-coordinate overlap is zero; FN-coordinate
overlap is the same 68 pixels.

Frame 1 in each DAQ-writer capture has 237 FP and 22,158 FN, with 237 FP and
22,090 FN outside the persistent list. All-frame totals are 237 FP and 25,558
FN occurrences. These transient observations are preserved, not added to the
persistent mask.

The map-programming intervals, including adapter enable/cleanup, were 13.54 s
(MapFML) and 12.83 s (MapFHL). Camera readback was source 3, period 15,625,000
ticks and 10 Hz for both; observed host image-arrival rates were approximately
10.004 and 10.003 Hz. All saved records had Rogue error=0, all full images
decoded, native file framing passed, and both cleanup-error lists were empty.
There were 513 and 512 timing-only 48-byte records respectively. Both wrapper
capture and restoration exit codes were zero.

## Interpretation and remaining tests

Using the current DAQ map-writing sequence alone **did not reproduce** the
historical production FP pattern under these 10 Hz standalone conditions.
This weakens the hypothesis that the write-order difference alone explains
the discrepancy in this setup. It does not establish the historical cause,
validate the full production DAQ path, or rule out interactions with rate,
trigger routing, other configuration steps, firmware/software history or
configuration history.

This is one DAQ write per mode. The planned repeated-programming comparison
and rate tests remain pending. Recover the historical run/readout trigger
settings before selecting a rate comparison; no rate sweep, external-timing
change, firmware change or test-bundle promotion was performed here.

## Final state and retained evidence

[Postflight](postflight-daq-writer-20260930.json) confirms ADC done/pass,
readout off, verified source-3 10 Hz idle settings restored, and frame count
204 stable across two samples with rate zero. The last map is MapFHL. PCIe
remains in standalone timing configuration with trigger masters disabled;
full reinitialization and Configure/readback verification are required before
returning to production DAQ operation.

- [Machine-readable comparison](daq-writer-20260930-results.json) records input,
  script and bundle equivalence, exact-array comparisons, data hashes and state.
- Frame-by-frame reviews: [MapFML](gain-map-attempt09-review.json),
  [MapFHL](gain-map-attempt10-review.json).
- Assessed coordinates: [MapFML](gain-map-attempt09-mismatches.csv),
  [MapFHL](gain-map-attempt10-mismatches.csv).
- Lossless test records: [09](test-attempt09.json.gz), [10](test-attempt10.json.gz).
- Payload reviews, software manifests, launch/console logs, wrapper execution
  records and restoration readbacks are archived by attempt number.
- Raw files, copied inputs, all-frame masks and `review-after-first/` arrays
  remain in `attempt09-mapfml/` and `attempt10-mapfhl/`, excluded from Git.
  Reviews verified detector identity and raw-file hashes, framing and mismatch
  counts against the recorder before producing the assessed results.
