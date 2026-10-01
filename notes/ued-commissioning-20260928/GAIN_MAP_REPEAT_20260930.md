# Standalone gain-map reproduction — 2026-09-30 PDT

> Historical September 2026 record. For current repeat-test entry points and archived evidence, see [the repeat-test guide](../epixquad/README.md). Process IDs, running-state statements and launch paths below describe that session.

After Mona powered the production detector on again, both standalone expand2
captures reproduced the September 28 accepted results. **Neither capture has
FP/FN outside the established persistent list after the first image.** The new
DAQ writer has not been exercised on hardware.

## Preflight

Read-only checks on `drp-ued-cmp003` found the same Rocky 9.6 host, KCU1500
PGP4 firmware 4.2.0, camera firmware and four carrier IDs as before. PGP RX/TX
and remote-ready flags were true, with zero reported link-down/error counts.
ADC was done/pass and AsicMask was 0xffff. The existing monitor IOC (PID 238794)
remained running; no competing capture, devGui or detector DRP process was
observed. The nonprivileged fuser command reported no visible users; process
inspection and successful VC1 access supplied the additional ownership evidence.

Camera readout was off, with source-3 internal idle triggering at 10 Hz. These
settings matched the saved restoration target exactly. Frame count was zero
and unchanged across two samples following the power cycle. PCIe timing was
already in the previous standalone configuration, with trigger masters disabled
and no timing CRC/decode errors. See the [probe](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-commissioning-20260928/probe-repeat-20260930.json),
[settings](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-commissioning-20260928/preflight-repeat-20260930.json), and
[host/process inspection](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-commissioning-20260928/preflight-repeat-20260930-host.log).

## Captures and comparison

| Attempt | Mode | Recorded / assessed images | FP per assessed frame | FN per assessed frame | Outside persistent list |
| --- | --- | --- | --- | --- | --- |
| 07 | MapFML | 51 / 50 | 0 | 68 | 0 |
| 08 | MapFHL | 51 / 50 | 0 | 68 | 0 |

Both used `--gain-writer standalone --rate-hz 10`, the same expand2 selection
of 22,169 FL pixels, the same UED baseline, and the pinned test bundle. The
selected-map, camera YAML and PCIe YAML checksums match attempts 05/06. All
submodule commits/local tracked diffs and the bundle patch-script hash also
match. The standalone writer source is byte-identical; capture/accounting code
is now commit `c539c20f5` with the writer selector and trigger readbacks.

The entire assessed FP and FN occurrence arrays and known-bad masks are
identical to the respective earlier captures, not just the aggregate counts.
The same 68 selected pixels from the 352-pixel persistent list account for all
3,400 FN occurrences per mode. None of the historical 43 extra FP coordinates
recurred. Every assessed frame individually has FP=0 and FN=68.

Camera readbacks confirmed source 3, period 15,625,000 ticks and 10 Hz. Host
image-arrival rates were about 10.005 Hz (MapFML) and 10.003 Hz (MapFHL).
Both captures had 51 complete decoded images and 513 timing-only 48-byte
records, all with Rogue error=0. Native file framing passed, both cleanup-error
lists were empty, and both wrapper capture/restoration exit codes were zero.

Frame 1 remains preserved and excluded only from the separate acceptance
analysis. MapFML frame 1 had 175 FP and 22,096 FN (previously 237/22,158);
MapFHL frame 1 had 237 FP and 22,158 FN, matching its earlier counts. This
variation does not extend into frames 2–51 and does not identify a cause.

## Final state and next step

The independent [postflight snapshot](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-commissioning-20260928/postflight-repeat-20260930.json) confirms
ADC done/pass, readout off, the verified source-3 10 Hz idle settings restored,
and frame count 102 stable across two samples with rate zero. MapFHL remains
loaded; PCIe remains in standalone timing configuration with trigger masters
stopped. Full reinitialization and Configure/readback verification are still
required before returning to production DAQ operation.

This adds one independent write per mode after a power cycle. It confirms the
standalone control result is reproducible in this session. It does not prove
the historical FP issue fixed. The next comparison is the prepared DAQ map
helper at the same 10 Hz settings; no DAQ-writer or rate-sweep test was run here.

## Retained evidence

- [Machine-readable comparison](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-commissioning-20260928/gain-map-repeat-20260930-results.json), including
  exact-array equality, input/software comparisons, data hashes and final state.
- [Attempt 07 review](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-commissioning-20260928/gain-map-attempt07-review.json) and
  [attempt 08 review](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-commissioning-20260928/gain-map-attempt08-review.json), with every frame's counts.
- [MapFML coordinates](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-commissioning-20260928/gain-map-attempt07-mismatches.csv) and
  [MapFHL coordinates](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-commissioning-20260928/gain-map-attempt08-mismatches.csv).
- Lossless full records: [07](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-commissioning-20260928/test-attempt07.json.gz), [08](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-commissioning-20260928/test-attempt08.json.gz).
- Payload reviews, software manifests, console/launch logs, wrapper execution
  records and restoration readbacks are archived by attempt number.
- Raw captures, copied inputs, all-frame masks and post-first-frame count arrays
  remain under `attempt07-mapfml/` and `attempt08-mapfhl/`, excluded from Git.
  Each `review-after-first/` was produced with `review_gain_map.py`, which checks
  detector identity, file hash/framing and its counts against recorder outputs.
