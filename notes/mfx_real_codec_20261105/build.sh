#!/usr/bin/env bash
# Reproduce the bridge build in a NEW scratch directory. No source patching.
set -euo pipefail
umask 077
codec_root=${1:?usage: build.sh NEW_SCRATCH_DIRECTORY}
codec_notes=$(cd -- "$(dirname -- "$0")" && pwd)
codec_repo=$(git -C "$codec_notes" rev-parse --show-toplevel)
codec_drp=9f7391c091d21afe585165d5ee72ca057ed98e4a
codec_sleek=00ff130d8d3182204b6b63fc22d5c85ad6b5f2e1
mkdir -- "$codec_root"
codec_root=$(cd -- "$codec_root" && pwd)
git clone https://github.com/burtscher/SLEEK.git "$codec_root/SLEEK"
git -C "$codec_root/SLEEK" checkout --detach "$codec_sleek"
for name in SingleCompressorLossy.cu SingleCompressorLossy.hh; do
    git -C "$codec_repo" show "$codec_drp:subprojects/packagefiles/sleek/app/$name" > "$codec_root/$name"
done
/sdf/group/lcls/ds/tools/cuda12.8/bin/nvcc \
    -shared -Xcompiler -fPIC -O3 -arch=sm_80 -std=c++17 -include cuda_runtime.h \
    -I "$codec_root" -I "$codec_root/SLEEK" \
    "$codec_notes/encode_bridge.cu" "$codec_notes/decode_gpu_bridge.cu" \
    "$codec_root/SingleCompressorLossy.cu" -o "$codec_root/libcodec_gpu.so"
/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/xpp_drp_gpu_311/bin/g++ \
    -shared -fPIC -O3 -fopenmp -std=c++17 -I "$codec_root/SLEEK" \
    "$codec_notes/decode_cpu_bridge.cpp" -o "$codec_root/libcodec_cpu.so"
sha256sum "$codec_root/SingleCompressorLossy.cu" "$codec_root/SingleCompressorLossy.hh" \
    "$codec_root/libcodec_gpu.so" "$codec_root/libcodec_cpu.so" > "$codec_root/build-sha256.txt"
