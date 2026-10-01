# DAQ gain-map write diagnosis: IsEn overwritten before programming

Follow-up: [run 7](../isen-ordering-test/README.md) tested the original DAQ
writer with only IsEn ordering corrected. All 1,086 frames are clean outside
known-bad and all 44 prior FP addresses read FH twice afterward. The historical
diagnosis and candidate below describe the evidence before that validation.
Its replay and candidate target the pre-fix source at commit `5c1800656`;
use the follow-up regression script to test the now-corrected source.

The full DAQ Configure path violates its own programming-state requirement:
it sets ASIC `IsEn=True`, applies expert ConfigDB values containing `IsEn=0`
for every ASIC, then programs the map without reasserting `IsEn=True`.
This is a confirmed software sequencing defect and the leading explanation
for the extra FL pixels. Its responsibility for the observed 43/44 row-0 FPs
still needs an isolated hardware test. No hardware, DAQ state, ConfigDB or
installed Python files were changed during this diagnosis.

## Evidence

In `psdaq/psdaq/configdb/epixquad1kfps_config.py` at source HEAD `5c1800656`:

1. `config_expert`, lines 1019–1024, explicitly requires `IsEn=True` while
   configuring and sets it on all 16 ASICs.
2. Line 1035 calls `apply_dict` with the complete expert camera configuration.
   `apply_dict` does not exclude or override `IsEn`.
3. The saved `ued/TEST` key-1406 configuration has `IsEn=0` on all 16 ASICs.
   Its SHA256 is recorded in [replay.json](replay.json).
4. Lines 1043–1050 invoke `_apply_raw_pixel_map_program`. That helper does not
   set `IsEn`; it assumes its caller established the programming state.
5. The subsequent cleanup sets `IsEn=False` for acquisition, which is the
   appropriate place for that running value.

The failing-run detector log confirms the expert writes, including:

```
30_21:59:03_drp-ued-cmp003:epixquad1kfps_0.log
1794: Setting cbase.Epix10kaSaci0.IsEn to 0
2799: Setting cbase.Epix10kaSaci15.IsEn to 0
2826: Using user.pixel_map_raw write path ... unique_values=[8, 12]
2827: Raw pixel_map write complete in 13.513 s (22169 pixel updates)
```

Its later normal Configure repeats the same sequence at lines 3034–4067.
The log directory is `/sdf/home/u/uedopr/daq/logs/2026/09/`.
[Extracted evidence](isen-log-evidence.txt) includes all 16 writes and the
successful run-6 log for comparison. These log messages record requested
register writes; they are not independent hardware readbacks during programming.

## Why standalone passed, including its DAQ-helper variant

`capture_gain_writer.program` sets `IsEn=True` immediately before invoking
the DAQ map helper. It does not run the full `config_expert` function, so
no subsequent ConfigDB application overwrites that setting.

The original standalone `_write_gain_mode` also sets `IsEn=True` immediately
before programming. The new run-6 adapter calls that same helper *after*
the expert ConfigDB application, thereby correcting the state incidentally.

Our initial run-6 comparison identified map order and camera gating, but
missed this third difference. Run 6 alone therefore cannot attribute the
improvement to any one of these three changes. Post-run YAML comparisons
also miss this defect: all paths restore `IsEn=False` after writing.

## Offline reproduction

[replay.py](replay.py) executes the checkout's actual `config_expert`,
`apply_dict`, trigger helpers and raw-map helper using AST extraction to
exclude hardware imports. It uses the complete saved TEST configuration,
the actual expand2 mask, and a register recorder with an ideal address model.
The successful paths call the actual standalone adapter/helper or capture
wrapper. No Rogue server, hardware, network or ConfigDB connection is used.

| Path | IsEn during 16 matrix fills and 22,169 FL writes | Ideal raw-map mismatches |
|---|---|---:|
| Original full DAQ Configure | 0 | 0 |
| DAQ helper under standalone wrapper | 1 | 0 |
| Standalone adapter inside DAQ Configure | 1 | 0 |
| Proposed IsEn-only correction | 1 | 0 |

The proposed correction preserves the original map-command sequence exactly;
the recorded command-sequence SHA256 values agree. All 540,672 intended raw
pixel codes match the mask in every replay. This rules out a deterministic
address/value mismatch in these tested write algorithms; it does not simulate
electrical ASIC behavior or prove how IsEn=0 affects row addressing.

Reproduce from the repository root:

```bash
env PYTHONPATH=psdaq \
  /sdf/group/lcls/ds/ana/sw/conda2/inst/envs/ps_20241122/bin/python \
  notes/ued-full-daq-20260930/gain-writer-diagnosis/replay.py \
  > notes/ued-full-daq-20260930/gain-writer-diagnosis/replay.json
```

This regenerates the report and candidate patch; it does not apply the patch.

## Scope of the causal conclusion

The stored DB map and recorded Configure map were correct. All 44 run-4 FP
addresses instead read actual ASIC code 8 (FL) twice, despite requesting 12
(FH), and all are ASIC row 0. The DMA/pre-write/disk comparison found the
same pixel words throughout. Correcting the 44 ASIC memory locations made
run 5 clean; full standalone programming without those corrections made run
6 clean. These observations locate the failure in detector programming,
before file writing. They do not yet identify the ASIC mechanism that creates
row-0 FL settings while IsEn is zero.

## Smallest decisive hardware check

[candidate-isen-only.patch](candidate-isen-only.patch) reasserts `IsEn=True`
after expert settings, immediately before the map write. It is reviewed by
the offline replay but **not applied or deployed**. It preserves the original
DAQ writer, pixel ordering, camera gates, timing, DB values and cleanup.
The running opt-in standalone adapter would need to be deselected for this
test; otherwise it would conceal the difference.

For isolated validation, use the original DAQ writer with only this correction,
perform a fresh full Configure (no targeted overwrite), collect a short 360 Hz
run, analyze every frame against expand2 and the unchanged known-bad list,
then read the 44 locations after EndRun. Do not add a pre-run ASIC readback:
that access sequence itself changes ASIC configuration state. Expected success
is zero FP/FN outside known-bad, 68 known-bad FNs per frame, and all 44 requested
FH addresses reading 12 with trbit 1. A repeat/reversion comparison can establish
whether IsEn alone explains reproducible failure and recovery.

Do not change ConfigDB's running IsEn value to 1 as a workaround. The software
must own the temporary programming state and restore the acquisition state.
Production hardening should also ensure cleanup and trigger handling on write
failure; that is separate from this minimal causal test.
