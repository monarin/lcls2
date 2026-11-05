// Include upstream decoder unchanged. Only trusted, outer-validated test blobs.
#define main upstream_sleek_cpu_main
#include "lossy_compressors_cpu/sleek_single_decompressor_lossy.cpp"
#undef main

extern "C" long long codec_decode_cpu(const byte* input, byte* output,
                                      float bound) {
  unsigned bits;
  memcpy(&bits, &bound, sizeof(bits));
  const int exponent = int((bits >> 23) & 255) + 24;
  const int offset = int((unsigned(exponent) << 23) - (1u << 23));
  long long size = 0;
  h_decode(input, output, size, exponent, offset);
  return size;
}
