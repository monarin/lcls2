# UED 50-frame commissioning — 2026-09-28

Latest: [September 30 DAQ-writer comparison](DAQ_WRITER_RESULTS_20260930.md)
passed in MapFML and MapFHL at 10 Hz. Both assessed mismatch arrays exactly
match the [same-session standalone controls](GAIN_MAP_REPEAT_20260930.md):
zero FP and 68 known FN per frame, with no extra historical FP pixels.
ADC is healthy and readout is off after restoration. Repeated programming
and rate comparisons in the [plan](DAQ_WRITER_COMPARISON.md) remain pending.

The original [expand2 MapFML/MapFHL results](GAIN_MAP_RESULTS.md) have no mismatches
outside the persistent list across 50 assessed images per mode, after excluding
frame 1 as instructed. The historical production FP pattern was not reproduced.

Follow-up: [FM and FL tests](FM_FL_RESULTS.md) are complete. FM has zero
gain-bit mismatches; FL has first-frame mismatches and 352 persistent candidates
on ASIC 8 columns 97/98. That report records the latest hardware state.
The [historical comparison](HISTORICAL_COMPARISON.md) confirms those same
352 persistent pixels in three previously analyzed UED experiment runs.

The bounded FH commissioning capture passed on `drp-ued-cmp003`, using
`/dev/datadev_0`, PGP4 lane 0, all 16 ASICs, internal camera triggering at
10 Hz, and local XpmMini timing. No DAQ or devGui was run. Normal PROM
compatibility checking succeeded with bypass disabled; no ADC training or
PROM programming was performed.

| Attempt | Full images | 48-byte timing-only records | Result |
| --- | ---: | ---: | --- |
| 01 | 50 | 1014 | Acquisition/analysis succeeded; shutdown drain timed out |
| 02 | 50 | 502 | Completed, clean drain, no cleanup errors |

Both attempts contained 40 image payloads of 1,095,248 bytes and 10 of
1,095,288 bytes. Every record had Rogue error=0. Every full image decoded;
FH gain-bit false-positive and false-negative pixel/occurrence counts were
zero, including control-row counts. The `invalid_records` field counts
the timing-only records because they do not qualify as full images; it
does not imply damaged image payloads. Native file framing was checked
through EOF without a truncated record.

The first attempt exposed an ordering error in recorder cleanup: stopping
camera readout leaves PCIe timing-only events flowing. Cleanup now disables
the timing TriggerEventBuffer master enables before draining queued records,
then calls StopRun and closes the writer. It attempts every timing buffer
even if one disable fails, and still attempts this step if camera quiesce
fails. The second live attempt verified the corrected order. All 14 offline
tests passed, including native Rogue writer/reader integration and failure
cleanup. See [offline-tests.log](offline-tests.log).

## Scope and remaining observations

This establishes full-image transport, decoding, FH gain-bit behavior, normal
startup and bounded shutdown. It does not establish pedestal quality or
complete gain-map acceptance. Both captures have a high first-frame ADC
level: segment means around 13.2–13.3k, dropping to roughly 2.6–2.9k on
subsequent frames. No ADC pixels were zero. The cause of the startup
transient has not been established; retain it and examine settling in the
next test rather than silently excluding that frame from accounting.

The following plan was recorded after FH; see the follow-up report above for
the completed FM/FL tests and revised next check.
Every upcoming fixed-gain capture will also export per-pixel FP/FN coordinates,
counts/fractions, a candidate mask, and detector/mode/source provenance for
later gain-map known-bad-list review. Candidates are observations, not automatic
exclusions. Both original FH captures were exported offline under their own
`mismatch-records-v1/` subdirectories without modifying the original records;
each has zero candidates across 50 images. See
[backfill locations/checksums](mismatch-backfill.json),
[attempt 01 metadata](mismatch-attempt01-metadata.json),
[attempt 02 metadata](mismatch-attempt02-metadata.json), and
[16-test verification](mismatch-tests.log).

Keep the bundle in test status until that validation is complete. After
successful testing, commit/review the validated superproject gitlinks and
required patch recipe so others can reproduce the bundle, then promote the
exact tested pins under the production naming convention. This initial
commissioning is not that promotion.

## Restored state and return to DAQ

Camera identity matched before both restorations. Both readbacks restored
the prior idle state: TrigSrcSel=3, TrigEn=true, AutoTrigEn=true,
AutoTrigPer=15625000 (10 Hz), RdoutEn=false. ADC remained done/pass.
The [postflight snapshot](postflight.json) independently confirms this;
camera frame count was 100 and unchanged across two samples, with rate=0.
The monitor IOC was left running.

Restoration covers idle trigger/readout settings, not a full configuration
rollback. The UED BEAM acquisition baseline and FH map remain loaded. PCIe
is still set for local timing: ClkSel=1, UseMiniTpg=true; HwEnable=false,
both sampled trigger master enables=false, and EventBuilder.Blowoff=true.
Timing link is up, CRC/decode counters are zero. A **full detector
reinitialize plus Configure and readback verification** is required before
production DAQ use; see the [restoration source review](../ued-settings-20260928/README.md).

The four missing readout fields in the snapshot are absent from the pinned
Python model, as in preflight. The first postflight invocation omitted
QT_QPA_PLATFORM=offscreen and failed at Qt startup; its log is retained as
`postflight-no-display.log`. The corrected read-only snapshot succeeded.

## Reproducibility and artifacts

Bundle: `/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1`,
superproject `4a28be03a7aa77373f0795dae7a1e0d003d4bc3c`, ePix
`44a7c1cb9ed1869e75b16e24bb610e7e3ee53698`, PCIe apps
`ccbe3de52fa7f6c09dac0216372121bcfbbc49e5`. Python 3.9.21 / Rogue 6.1.3,
DAQ environment `daq_20250402_r9`. Each software manifest records all 106
repositories, local patches, script hashes and the lcls2 diff at execution.
The profile note was updated after capture to record the observed payloads;
the executed profile copies remain unchanged in the attempt directories.

- [Attempt 01 record](test-attempt01.json), [software](software-attempt01.json),
  [payload review](attempt01-payload-review.json), [restoration](restore-attempt01.json).
- [Attempt 02 record](test-attempt02.json), [software](software-attempt02.json),
  [payload review](attempt02-payload-review.json), [restoration](restore-attempt02.json).
- `execution-attemptNN.json` records the command and capture/restore exit codes;
  both restoration exits were 0. Attempt 02 acquisition plus drain took 5.34 s.
- `run_once.py --attempt N` is the host-gated, bounded commissioning wrapper;
  `restore_camera_idle.py` uses the saved preflight identity/settings.
  It must be invoked with the recorded environment and headless Qt setting.
- `review_payloads.py CAPTURE OUTPUT_JSON` reproduces the offline record
  histogram, file checksum and per-image ADC statistics.

Raw data and complete per-attempt YAML/profile/map artifacts are retained
under this directory in `attempt01-fh/` and `attempt02-fh/`, excluded from
Git. Attempt 02 `capture.dat` is 54,791,312 bytes, SHA256
`691286f39244b2600642661fbc983ecbea992a20610129710f55af0a29642e3c`.
Attempt 01 is 54,819,984 bytes, SHA256
`ca995df98471ee8481c85ec2396a11c8d8a0a3aeca15409abca1d17c852c143a`.
Git preserves the summaries/manifests/logs; raw-file availability depends on
retaining these shared-filesystem directories.
