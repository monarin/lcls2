// Include upstream decoder unchanged, exposing its actual kernel via a test ABI.
#define main upstream_sleek_gpu_main
#include "lossy_compressors_gpu/sleek_single_decompressor_lossy.cu"
#undef main
#include <cstring>

extern "C" int codec_decode_gpu(const byte* input, byte* output,
                                long long* output_size, float bound,
                                cudaStream_t stream) {
  unsigned bits;
  memcpy(&bits, &bound, sizeof(bits));
  const int exponent = int((bits >> 23) & 255) + 24;
  const int offset = int((unsigned(exponent) << 23) - (1u << 23));
  cudaDeviceProp prop;
  auto rc = cudaGetDeviceProperties(&prop, 0);
  if (rc != cudaSuccess) return int(rc);
  d_reset<<<1, 1, 0, stream>>>();
  d_decode<<<prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / TPB),
             TPB, 0, stream>>>(input, output, output_size, exponent, offset);
  return int(cudaGetLastError());
}
