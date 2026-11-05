# Real SLEEK compression of MFX Jungfrau data

This is a scratch feasibility fixture, not a published XTC run or a throughput
benchmark. The unchanged DRP SLEEK compressor consumes **calibrated float32**.
Its decoded output feeds a cube directly; applying raw calibration again would
be incorrect. Original uint16 events remain the calibration-kernel reference.

## Sources and representation

- DRP master: `9f7391c091d21afe585165d5ee72ca057ed98e4a`.
  `subprojects/packagefiles/sleek/app/SingleCompressorLossy.{cu,hh}` is compiled
  unchanged, called by `encode_bridge.cu` using the DRP state/index/size-prefix
  convention. This exercises the compressor used by `SleekReducer`, not the
  entire running DRP, its persistent CUDA graph, or GPU direct I/O.
- SLEEK wrap revision: `00ff130d8d3182204b6b63fc22d5c85ad6b5f2e1`.
  The CPU and GPU decoder translation units are included unchanged with their
  CLI `main` symbols renamed. Both are called through small test ABIs. Serial
  calls on CUDA device 0 only; concurrent codec instances are not supported by
  this harness (upstream kernels use a global chunk counter).
- Input: first requested events of `mfx101555026`, run 10, from
  `/sdf/data/lcls/ds/prj/public01/xtc`. Original event timestamps are retained.
- Calibration: byte-verified Stage-1 database snapshot, injected into a fresh
  normal `Detector.raw` before calibration-cache initialization. Pedestals,
  gain and offset match that snapshot by SHA256. This is an offline test,
  **not** a new live DB selection or deployment for run 3/4/5.
- Settings: CPU calibration v3 versus the existing user-owned GPU calibration
  example, offsets enabled, masks and common mode disabled. The known source
  pixel-status issue is not exercised by this arithmetic test. No scientific
  acceptance of these settings or of the error bound is implied.

Five concatenated blob files, one per original Jungfrau DRP:

| Source stream | Ordered physical panels in each decoded blob |
|---|---|
| s005 | 3, 7, 0, 1, 2 |
| s006 | 19, 15, 11, 31, 27, 23 |
| s007 | 18, 14, 10, 6, 30, 26, 22 |
| s008 | 16, 12, 8, 4, 28, 24, 20 |
| s009 | 17, 13, 9, 5, 29, 25, 21 |

`report.json` indexes each blob by file, byte offset, length, timestamp, panel
order, decoded shape/type, hashes and per-pixel results. Each blob is exactly
the compressor output (including its decoded-size header), without the DRP's
external compressed-length prefix or the standalone SLEEK CLI's extra framing.
The `.sleek` files are **not XTC2**, and cannot be passed directly to DataSource
or the standalone CLI. Use the index and decoder test ABI. They contain only
Jungfrau payloads; other detectors and XTC transitions are not packaged here.

## Checks and current result

Scratch root:
`/sdf/scratch/users/m/monarin/gpu-validation/mfx-real-codec/20261105-r1/`.

The one-event gate (`gate01`, job 43382364) passed across all 32 panels:
GPU calibration matched CPU v3 bitwise, both decoders agreed bitwise, and every
decoded pixel was within the configured absolute bound 3.0 (observed max 2.0).
The five blobs totaled 14,013,832 bytes, compared with 33,554,432 raw bytes and
67,108,864 decoded float32 bytes. The 1,000-event trial is job 43382526 (`pilot01`);
its final results and readback are recorded separately when complete.

**Separate decoder failure:** `probe_edges.py`, job 43382608, tests repeated
`[0, 1, -3, 1e9, -1e9, +inf, -inf, nan]`. The CPU decoder passed; the unchanged
GPU decoder differed from the CPU on 81,920 of 131,072 values, with finite
absolute errors up to 1e9 and corrupted nonfinite values. `edges01.json` retains
the original diagnostic, including hexadecimal input/output witnesses.
The initial diagnostic double-counts some nonfinite mismatches; use the exact
bit-mismatch count and witnesses rather than summing its mismatch categories.

Source inspection identifies two shadowed `offs` variables in the GPU decoder:
the compressed-chunk byte offset in `d_decode` is passed to `d_iSLEEK` instead
of the quantization offset; `d_iSLEEK` also passes a local block index offset
to `deQuantize` in one path. The above-threshold/lossless branch needs the
quantization offset. This is a supported diagnosis, not a validated codec fix.
No codec source has been modified, and a passing MFX sample does not qualify
this decoder for arbitrary float32 inputs. Resolve this before production use.

The trial intentionally retains one raw/calibrated witness and hashes all
later references rather than duplicating the original dataset. `verify_saved.py`
reads every stored blob back, checking hashes, framing, file/index coverage,
shape, panel order and timestamp alignment. It does not infer a pass from the
Slurm exit code alone. The numerical trial validates all pixels in every blob.
Two CPU unit tests check framing rejection and finite/nonfinite error detection.

## Reproduction

`bash build.sh NEW_SCRATCH_DIRECTORY` builds the same unchanged sources and
bridges with CUDA 12.8, targeting A100 (`sm_80`). The local initial attempt using
system GCC failed on OpenMP declarations; the successful build uses the
`xpp_drp_gpu_311` environment's compiler. Full trial inputs, sources, binaries,
driver copies and hashes are retained under the scratch root.

Run `trial.py --root BUILD_DIRECTORY --output NEW_OUTPUT_DIRECTORY --events N`
through `notes/synthetic_xpp_stage2_20261105/runtime.sh` in an A100 allocation
(account lcls, partition ampere, one GPU, four CPUs, 24 GB host memory).
Set `LD_LIBRARY_PATH` to CUDA 12.8 `lib64` and the `xpp_drp_gpu_311` environment
`lib` before that runtime wrapper prepends the pinned psana install library.
Use an absolute scratch `CUPY_CACHE_DIR`. The wrapper enforces
`PS_FAKESTEP_FLAG=0`, `PS_PARALLEL=none`, and one OpenMP thread.
Output directories must be new; no existing results are overwritten.

Run `probe_edges.py --root BUILD_DIRECTORY --output NEW_JSON` under the same
GPU/runtime setup. Its process exit indicates that the diagnostic completed;
the JSON's CPU/GPU results determine whether decoding passed.

Readback: `verify_saved.py PILOT_DIRECTORY --output NEW_JSON` through the
pinned runtime. Unit tests: `python -m unittest discover -s
notes/mfx_real_codec_20261105 -p test_trial.py` in that environment.

## Next dataset stage

After resolving the decoder limitation, package the proven payloads into XTC2,
with `sleek` uint8 arrays and explicit decoded shape, ordered physical-panel
IDs, codec revision and error bound. Add a reader adapter, preserve the desired
small detectors and native transitions, rebuild SMD offsets, and validate the
DataSource/decompress/cube path. Use original raw events separately for
calibration regression tests. Run identity and any DB deployment remain
unassigned; existing run data and calibration ranges are unchanged.
