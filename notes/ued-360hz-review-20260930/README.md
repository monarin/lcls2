# UED standalone 360 Hz settings review — September 30, 2026

The standalone baseline omits a readout-clock override applied by full DAQ
Configure. Match and validate that override at low rate before attempting
360 Hz. This is a confirmed configuration difference, **not a proven cause
or cure of the historical 360 Hz failure**. This investigation made no
register or ConfigDB writes and did not acquire data.

## Confirmed settings

Read current UED `BEAM/epixquad1kfps_0` twice at 19:00 UTC; both responses
matched each other and the September 28 snapshot. Canonical JSON SHA256:
`c0d91aa6e7485ab1a2a59eea7c8a823698aeca55c5cc32b9f876b7c66f95d73a`.
See [configuration summary](config-summary.json), [complete snapshot](beam-config.json.gz)
and [live readback](registers.json) from `drp-ued-cmp003`, `/dev/datadev_0`, lane 0.

| Setting | Current standalone / database | Full DAQ or implication |
| --- | --- | --- |
| `AcqCore.AsicRoClkHalfT` | ConfigDB 3; baseline writes `0xAAAA0003`; live 3 | Legacy half-period register sets both halves to 3 |
| `AcqCore.AsicRoClkT` | Absent from ConfigDB/baseline; **live 6** | Full Configure finally writes **`0xAAAA0003`**, giving full period 3 |
| `RdoutCore.AdcPipelineDelay` | ConfigDB and live 68 | Keep 68; source comment saying 61 is stale |
| `AsicR0ToAsicAcq`, `AsicAcqWidth` | Both 15625 ticks | Each 100 microseconds at 156.25 MHz; the source's “100ms” comments are wrong |
| `user.gate_ns`, `user.start_ns` | 100000, 94000 | Gate is 100 microseconds; start is used for external timing alignment |
| Internal camera trigger | Source 3; auto/trigger enabled; period 15625000 | Idle state is 10 Hz, with readout disabled; 360 Hz period rounds to 434028 ticks |
| PCIe event builder timeout | 144675 ticks | About 0.926 ms; standalone and DAQ init use the 1080 Hz value |
| PCIe EVR channel rate selections 0 / 4 | 4 / 3 | YAML's nominal 100 Hz / 1 kHz selections; changing camera rate does not change these |

The decisive write is in
[epixquad1kfps_config.py](../../psdaq/psdaq/configdb/epixquad1kfps_config.py),
`epixquad_config`, line 1183 at lcls2 commit `a722c83f7`, after `config_expert`.
The deployed UED release also has this override at line 1181:
`/sdf/group/lcls/ds/ana/sw/conda2/rel/ued/lcls2_091026/install/lib/python3.9/site-packages/psdaq/configdb/epixquad1kfps_config.py`.
Our `--gain-writer daq` invokes the DAQ pixel-map helper, **not full Configure**,
so the successful 10 Hz writer comparison did not apply this override.

The pinned ePix repository is
`/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1/epix-quad-1kfps`,
commit `44a7c1cb9ed1869e75b16e24bb610e7e3ee53698`. Its
`firmware/common/EpixQuadCommon/rtl/AcqCore.vhd` lines 224–258 confirm
the half/full-period distinction and `0xAAAA` write protection. The reviewed
AcqCore and RdoutCoreBram RTL have no differences from camera firmware git
`43361ef5`, reported by the live FW 0x02030000 camera.
With 6.4 ns system ticks, period 6 is 38.4 ns (26.04 MHz); period 3 is
19.2 ns (52.08 MHz), using unequal 1- and 2-tick halves. After a full-period
write of 3, expect full-period readback 3 and legacy half-period readback 1.

A nominal readout-clock budget alone does not establish why 360 Hz failed:
`4 * 48 * 178` readout clocks at period 6 take about 1.31 ms, versus a
2.78 ms frame interval at 360 Hz. Acquisition, buffering and transport must
also work. Do not describe the faster clock as a demonstrated requirement
for 360 Hz based only on this comparison.

Other ConfigDB fields were reviewed: `OverSampleEn=1` and `OverSampleSize=3`
are absent/commented out in this pinned Python model and matching RTL;
do not add blind writes. The standalone baseline deliberately disables
`AsicSyncInjEn` (database value 1). Do not change injection while testing
rate effects. ConfigDB `TriggerDelay=2618`, partition and XPM rate selection
are not a recipe for an internal 360 Hz camera trigger; DAQ computes timing
delays/partition during configuration. The DAQ `epixquad_internal_trigger`
helper explicitly disables readout, so it is not by itself a recording setup.

## Historical failure and remaining uncertainty

The [preserved user report](history-360hz.json) from June 29 is for the test
module on **rdsrv421**, FW 2.3.0, FL. The 360 Hz file
`/tmp/data_20260629_150150.dat` contained 1960 channel-1 records:
171 of 48 bytes, 1453 of 1616 bytes, and 336 of 1656 bytes; no full images
decoded. The preceding 1080 Hz capture also had no full images.
This is not evidence of gain-map FP/FN at those rates because there were
no images to assess. The report does not establish the exact register
state used then.

Short records do not prove that the reader needs a packet assembler.
Matching `RdoutCoreBram.vhd` enters its footer early on buffer/time errors
(lines 909–917); header, footer and TPS alone total 1600 camera bytes.
The observed sizes are consistent with such an abbreviated frame plus
batcher overhead, but historical raw payload/error-bit analysis would be
needed to confirm that interpretation.

The independent PCIe timing rates and timeout warrant instrumentation.
An event-builder timeout can indicate missing input streams; it is not
automatically a camera readout deadline or proof of image truncation.
Do not simply scale the timeout or put `360` into the enumerated `RateSel`.

## Recommended next experiment (not performed)

1. Add an explicit, recorded DAQ-clock option that applies keyed
   `AsicRoClkT=0xAAAA0003` **after** the baseline, retaining pipeline delay 68.
   Record both clock readbacks so a later half-period write cannot silently
   undo the override. Preserve the existing period-6 results as controls.
2. With the same map/writer, acquire and assess a bounded 10 Hz control at
   period 3 first. Confirm ADC health, full image sizes, sequence continuity
   and the known-bad-list comparison, excluding frame 1 as agreed.
3. If that passes, hold the clock/map fixed and run bounded 10 → 360 → 10 Hz
   phases. The recorder already selects source 3, sets `AutoTrigFreqHz`, and
   enables trigger/auto-trigger/readout. Verify the actual period readback.
4. Record camera buffer/error counters and PCIe per-input data/timeout/drop
   counters before/after each phase, plus all short-record sizes and error
   flags. Check host throughput (roughly 394 MB/s of image payload at 360 Hz).
   If missing timing streams correlate with failures, vary timing/timeout
   separately, preserving the initial result. A 360 Hz frame-period timeout
   would be about 434028 ticks, but its suitability is not yet established.

Readout remains off, frame count 204 stable, ADC done/pass. PCIe remains in
the existing standalone local-timing state with sampled master enables off
and event-builder blowoff on. Full reinitialize/Configure is still required
before returning to production DAQ.

## Evidence collection

[fetch_config.py](fetch_config.py) and [read_settings.py](read_settings.py)
are read-only evidence collectors. Source `scripts/setup_ued_epixquad_test.sh`
before the ConfigDB collector; this supplies the normal environment/auth.
Initial environment/route failures are retained in `config-fetch*.log`;
[config-fetch-setup.log](config-fetch-setup.log) records the successful fetch.
[registers.log](registers.log) preserves the live read output. Four unavailable
readout fields are reported as errors because the pinned model lacks them;
the relevant clock, trigger, pipeline and ADC reads succeeded.
