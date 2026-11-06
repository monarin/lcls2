# Small lossless raw compression measurement

One retained original MFX run-10 event (timestamp 4909050823936603206), all
16,777,216 uint16 pixels, split into the five original Jungfrau DRP groups.
Both codecs use 8 KiB blocks. Every input group matches the previously validated
raw SHA256, and every decoded word matches the original exactly, including gain
bits. See `result.json` for versions, hashes, per-stream bytes and totals.

| Encoding | Bytes | Decimal MB | Raw/compressed factor |
|---|---:|---:|---:|
| Original uint16 | 33,554,432 | 33.554 | 1.000 |
| LZ4 | 33,710,037 | 33.710 | 0.995 |
| Bitshuffle + LZ4 | 27,940,162 | 27.940 | 1.201 |

Bitshuffle+LZ4 saves about 16.7% of the raw bytes; ordinary LZ4 expands them
slightly. For this same event, the earlier **lossy calibrated-float** SLEEK test
stored 14,013,832 bytes, or 2.394x relative to raw. Its 2.70x figure was the
average of 1,000 events. These are different accuracy requirements, and this
single raw event does not establish a run-wide lossless compression factor.

This measurement uses CPU bitshuffle 0.5.2 and python-lz4 4.4.5 (LZ4 1.9.4).
It is not nvCOMP output, a GPU throughput measurement, or a new XTC dataset.
Reported sizes include four bytes of compressed-length framing per 8 KiB block
and exclude external shape/index metadata and XTC headers. Bitshuffle's native
output already includes that framing; the plain LZ4 count adds the equivalent
four bytes per block. The two packages may bundle different LZ4 implementations.
Inspection of upstream
[src/bitshuffle.c](https://github.com/kiyo-masui/bitshuffle/blob/master/src/bitshuffle.c)
(file blob SHA `f641b0f933c26cd800b7e97007921360119c83a6`) confirms four-byte
block lengths. The library version and binary hash in the result identify the
actual runtime used here.

Dependencies were installed only under
`/sdf/scratch/users/m/monarin/gpu-validation/raw-compression-factor/20261105-r1/packages`,
using `pip --no-deps --only-binary=:all: --no-cache-dir` and the existing Python
3.11 environment. The source raw witness remains in the previous codec trial's
scratch directory; no new raw/compressed data copies are retained.

Reproduce with `OMP_NUM_THREADS=1` and Python from
`/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/xpp_drp_gpu_311/bin/python`:

```bash
python notes/raw_compression_factor_20261105/measure.py \
  --packages /sdf/scratch/users/m/monarin/gpu-validation/raw-compression-factor/20261105-r1/packages \
  --raw /sdf/scratch/users/m/monarin/gpu-validation/mfx-real-codec/20261105-r1/pilot01/raw-event0.npy \
  --output /absolute/scratch/path/to/new-result.json
```

The output file must be new. Validation is the full lossless round trip and
reference hashes; no new unit tests or GPU jobs were needed for this size check.

Independent Claude review of `a98c01fafa9ee2d254bee35aa95c7ab1e298d1b5`
passed with no findings; see [review record](review-01/review.json).
