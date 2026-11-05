// Thin test ABI around the unmodified DRP wrapper. Serial calls, device 0 only.
#include <cuda_runtime.h>
#include <cstddef>
#include "SingleCompressorLossy.hh"

extern "C" void* codec_create(size_t bytes, float bound) {
  return new SLEEK::SingleCompressorLossy(bytes, bound);
}
extern "C" long long codec_max_size(void* ctx) {
  return static_cast<SLEEK::SingleCompressorLossy*>(ctx)->maxSize();
}
extern "C" int codec_encode(void* ctx, unsigned* state, unsigned* index,
                            const uint8_t* input, long long input_bytes,
                            uint8_t* output, long long output_stride,
                            cudaStream_t stream) {
  static_cast<SLEEK::SingleCompressorLossy*>(ctx)->updateGraph(
      stream, state, index, input, input_bytes, output, output_stride);
  return int(cudaGetLastError());
}
extern "C" void codec_destroy(void* ctx) {
  delete static_cast<SLEEK::SingleCompressorLossy*>(ctx);
}
