# UED internal-trigger test preparation — 2026-09-28

Mona selected internal triggering for the standalone test and requested that
we preserve the settings or verify that DAQ configuration restores them. The
UED profile is now `ready_for_test`: this permits an initial commissioning
capture, not production acceptance. No detector/timing settings were changed
in this preparation step; no images were acquired.

## Evidence and selected settings

Read-only observations are in [registers.json](registers.json), with complete
console output in [registers.log](registers.log). The first attempt used an
incorrect PyRogue relative-node lookup; [that log](registers-path-error.log) is
retained. The corrected probe read all requested existing nodes; four readout
fields are absent because their declarations are commented out in the pinned
Python model, not because a hardware transaction timed out.

| Item | Observation / selected test value |
| --- | --- |
| Device | drp-ued-cmp003, /dev/datadev_0, lane 0; KCU1500/PGP4 v4.2.0 |
| Existing PCIe timing | ClkSel=0 (119 MHz), ModeSel=1, UseMiniTpg=false, XpmMini.HwEnable=false; RxLinkUp=1; zero CRC/decode counters |
| Existing trigger routing | DAQ buffer 0 uses XPM, partition 0, delay 8577; run buffer 4 uses EVR event code 6, delay 11186; both master enables true |
| Existing camera | TrigSrcSel=3, TrigEn=true, AutoTrigEn=true, AutoTrigPer=15625000 (10 Hz); RdoutEn=false |
| ADC / ASICs | Done=true, failed=false; mask 0xffff |
| Existing image stream | FrameCnt=0 and FrameSize=0 in two samples one second apart; readout is disabled, so this does not establish an image fault |
| Test timing | Local XpmMini plus camera internal auto-trigger at 10 Hz; intentionally different from the production external timing path |
| PCIe YAML | Pinned defaults_LCLS-II.yml, used specifically for the local-XpmMini test |
| EventBuilder | Bypass=0, VcDataTap=0, pause threshold=32, timeout=144675 match the live snapshot; retained for initial test, not derived from 10 Hz |
| Camera baseline | Current UED BEAM ConfigDB export, adapted and reviewed offline; see below |
| PROM validation | Enabled (prom_bypass=false); normal Top startup must pass its calibration/identity check; no automatic bypass or training |
| Image acceptance | 1095248 or 1095288 bytes, from the existing full-image decoder contract; these sizes have **not yet been observed on this powered-on UED module** |

The normal runtime checks remain enabled. A short commissioning capture must
establish PROM startup compatibility, completed images, payload format and
successful decoding before the gain-map sequence. Unexpected payloads do not
count toward the frame target. Stop and retain the record on any failure.

## Camera baseline

Fetched `ued / BEAM / epixquad1kfps_0` using ConfigDB read calls. The exact
response is preserved in [beam-config.json.gz](beam-config.json.gz); the
uncompressed JSON SHA256 is
`c0d91aa6e7485ab1a2a59eea7c8a823698aeca55c5cc32b9f876b7c66f95d73a`.
The revision-key endpoint returned HTTP 403, while configuration reads were
allowed. Two successive full configuration responses matched. The content
hash identifies this snapshot; no revision key is claimed. No serial-DB
composition or relative configuration was required. See
[config-summary.json](config-summary.json).

The latest matched DAQ logs are the September 25, 15:11:22 group under
`/sdf/home/u/uedopr/daq/logs/2026/09/`. The detector log records the same principal
acquisition settings at lines 1729–1751; control reached Configured and the TEB
received both contributors' Configure transitions. The session subsequently
unconfigured and was terminated. This establishes configuration provenance,
not clean gain-map data or a completed image-validation test.

Key differences from today's power-on/idle state:

| Register | Live snapshot | UED BEAM baseline |
| --- | ---: | ---: |
| AsicR0Width | 100 | 30 |
| AsicR0ToAsicAcq | 100 | 15625 |
| AsicAcqWidth | 100 | 15625 |
| AsicAcqLToPPmatL | 0 | 1000 |
| AsicRoClkHalfT | 5 | 3, written with 0xAAAA key |
| AdcPipelineDelay | 74 | 68, written with 0xAAAA key |
| DummyAcqEn | true | false |

The old bench YAML writes `AsicRoClkT` and pipeline delay 61. It is not used as
the UED camera baseline. The generated
[UED YAML](../../psdaq/psdaq/debugtools/epixquad1kfps/baselines/ued_beam_20260928.yml)
uses the current `AsicRoClkHalfT` field, pipeline delay 68, and the configured
ASIC registers. It omits AsicMask writes, nonexistent model fields, scope/ADC
test commands, PixelDummy, and the same ASIC-3 fields explicitly skipped by
`epixquad1kfps_config.apply_dict`. Every exclusion is recorded in
[baseline-review.json](baseline-review.json).

Test overrides disable trigger/readout during loading and disable charge
injection. The gain writer programs the selected gain map after baseline load.
SACI device enable and IsEn are set before bulk YAML loading, as in DAQ
configuration; the shared helper attempts to disable every entered ASIC even
on partial failure. This avoids relying on bulk-write ordering to enable SACI.
No PROM, reset, ADC training, or power-cycle command is in the baseline.

## Restoration / return to DAQ

Both the working source and the deployed
`/sdf/group/lcls/ds/ana/sw/conda2/rel/ued/lcls2_091026/install/lib/python3.9/site-packages/psdaq/configdb/epixquad1kfps_config.py`
were inspected. The September 25 DRP command specifies `timebase=119M`.

A **full detector reinitialize plus Configure** reapplies the functional
production path:

1. `epixquad_init` starts DevRoot with standAloneMode=false when xpmpv is absent.
   DevRoot calls ConfigLclsTimingV2, whose ConfigLclsTimingBase sets
   UseMiniTpg=false. epixquad_init subsequently selects ClkSel=0 for 119M and
   resets the timing receiver.
2. It selects run buffer 4 / EVR event code 6 and DAQ buffer 0 / XPM. Configure
   recalculates delays and partition from the current ConfigDB/user settings.
3. `epixquad_external_trigger` writes AutoTrigEn=0, TrigSrcSel=0, TrigEn=1 and
   RdoutEn=1. `config_expert` reapplies the camera/ASIC configuration and map.

This is a source-code check, not a post-test hardware verification. Enable and
Disable are no-ops here; Configure alone must not be substituted for the full
initialization path when returning from local XpmMini. No DAQ was started.

This is also **not a complete register-for-register rollback**. In particular,
DAQ external-trigger setup does not restore AutoTrigPer or explicitly clear
XpmMini.HwEnable; switching UseMiniTpg=false deselects local timing. The test's
10 Hz period matches the measured prior period, so it need not change. If a
later test changes that rate, restore the saved period explicitly. Clear the
local-mini enable and verify timing source/clock, buffer routing, camera mode,
ADC state and monitoring when returning to production.

For an idle detector after standalone tests, preserve monitoring: the current
idle camera state is source 3, TrigEn=true, AutoTrigEn=true, period 15625000,
RdoutEn=false. The recorder's normal cleanup disables camera triggering/readout;
restoring that prior idle state is a separate post-test action, with readback.
The monitor IOC remains running on VC3 and must not be asked to reconfigure
while a capture owns VC1. Its configure operation writes trigger settings.

## Validation

- 13 offline unit/integration tests passed, including SACI baseline enable/cleanup.
- The real PyRogue baseline loader passed against emulated memory for all 16
  ASICs, including keyed clock/pipeline values and trigger/readout-off checks.
- The UED headless dry run records the new profile, copied baseline and complete
  recursive software manifest; no hardware roots are opened by that dry run.
- Live work consisted only of register reads and process/configuration inspection.
