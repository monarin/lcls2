# UED run 2 with the intended submodule bundle — September 30, 2026

**The extra FP pattern persists with the intended test bundle.** Experiment
`uedc00107`, run 2 has 44 FP pixels outside the 352-pixel known-bad list and
68 FN pixels inside it in every one of 3,428 assessed frames. The first frame
has the same pattern and is retained separately, excluded from assessment.

All 43 run-1/historical FP coordinates remain, plus one new FP at zero-based
raw **segment 1, row 175, column 170**. The 68 FN coordinates are unchanged.
Every mismatch occurs in every assessed frame. Keep the known-bad baseline
unchanged; the extra FP coordinates are evidence of the discrepancy.

| Acquisition | Assessed frames | FP outside known | FN inside known | Rate |
| --- | ---: | ---: | ---: | ---: |
| Run 1, older submodule release | 4898 | 43 | 68 | 539.9707 Hz |
| Run 2, intended test bundle | 3428 | 44 | 68 | 539.9911 Hz |

The new DRP header confirms:

- `TESTRELDIR=/sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq/install-ued-gainmap-20260930`
- `SUBMODULEDIR=/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1`

The source log is
`/sdf/home/u/uedopr/daq/logs/2026/09/30_18:22:15_drp-ued-cmp003:epixquad1kfps_0.log`.
Its matching control and TEB logs show Configure, BeginRun, Enable, and Disable
for the same launch. The control log reports Pause after Disable; no EndRun
was present in the analyzed data snapshot.

The Configure map exactly matches expand2 (22,169 FL pixels), pixel codes 8/12,
and trbit 1 on all sixteen ASICs: MapFHL. Recorded carrier IDs match the
production known-bad baseline. A comparison of every non-private leaf in
`epixquad1kfpshw` found **all 1,920 fields identical between runs 1 and 2**,
including elementwise equality of array fields. This is recorded configuration
evidence, not hardware readback of every programmed pixel.

The old bundle therefore does not explain the existence of the 43 recurring
FPs by itself. One acquisition per bundle cannot establish that the additional
single pixel was caused by the bundle change. Recent standalone tests with
both gain writers had zero extra FPs; the full DAQ tests here also differ from
the 360 Hz standalone test in trigger rate and configuration/trigger sequence.
A full DAQ comparison at 360 Hz with this same bundle and map is the next
useful rate check. No timing, configuration, or hardware changes were made
during this review.

## Snapshot and reproduction

Analyzed the readable transferred file:
`/sdf/data/lcls/drpsrcf/ffb/ued/uedc00107/xtc/uedc00107-r0002-s001-c000.xtc2.inprogress`.
The local DRP file named in the log is
`/u2/pcds/pds/ued/uedc00107/xtc/uedc00107-r0002-s001-c000.xtc2`.

The FFB file contained 3,754,426,368 bytes. The complete 3,753,457,252-byte
prefix held 3,429 full, undamaged image events. The remaining 969,116 bytes
belong to an incomplete trailing datagram and were not interpreted. This is
a bounded in-progress snapshot, not whole-run completeness or loss validation.

The unchanged [reviewer](../review_xtc_gainbits.py) was run on SDF:

```bash
cd /sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq
PYTHONPATH="$PWD/install/lib/python3.9/site-packages" PS_PARALLEL=none \
  /sdf/group/lcls/ds/ana/sw/conda2/inst/envs/ps_20241122/bin/python \
  notes/ued-full-daq-20260930/review_xtc_gainbits.py \
  /sdf/data/lcls/drpsrcf/ffb/ued/uedc00107/xtc/uedc00107-r0002-s001-c000.xtc2.inprogress \
  /tmp/ued-run2-gainbits-new-review
```

Use a new output directory and the finalized filename after transfer finishes.

- [review.json](review.json): Configure checks, counts, rate, scope and hashes.
- [after-first-mismatches.csv](after-first-mismatches.csv): all 112 coordinates.
- [frames.csv](frames.csv): per-frame gain-bit counts and timestamps.
- [counts.npz](counts.npz): full-coordinate count arrays and baseline/input masks.
- [run1-comparison.json](run1-comparison.json): exact coordinate-set comparison.
- [config-comparison.json](config-comparison.json): saved hardware-config comparison.

Output checksums were independently verified after analysis. The coordinate
comparison confirms no run-1 mismatch disappeared and exactly one FP was added.
