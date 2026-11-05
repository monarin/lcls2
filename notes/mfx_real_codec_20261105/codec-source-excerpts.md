# Pinned upstream decoder source excerpts

SLEEK revision `00ff130d8d3182204b6b63fc22d5c85ad6b5f2e1`.
Exact excerpts, with original line numbers; full sources remain in scratch.
BSD-3-Clause, copyright 2026 Anju Mongandampulath Akathoott, Andrew Rodriguez, Martin Burtscher.
The unmodified source contains its full license; see the pinned source links below.

## lossy_compressors_gpu/sleek_single_decompressor_lossy.cu

SHA256 `4e620ed5254ea1bbb0448660e48821f43381e8d076ad134818e637c06998aea6`

[Source](https://github.com/burtscher/SLEEK/blob/00ff130d8d3182204b6b63fc22d5c85ad6b5f2e1/lossy_compressors_gpu/sleek_single_decompressor_lossy.cu)

```text
119: static __device__ inline type_u deQuantize(const type_u enc, const int thr_e, const int offs)
120: {
121:   const int e = 8;  // exponent bits
122:   const int m = 23;  // mantissa bits
123:   int dec = 0;  // default value is 0
124:   if (enc != 0) {
125:     const int abs = (enc + 1) >> 1;  // absolute value
126:     if (abs >= (1 << m)) {  // above threshold
127:       dec = abs + offs;  // decode losslessly
128:     } else if (abs > 0) {  // non-zero lossy case
129:       const int shift = __builtin_clz(abs) - (31 - m);  // compute shift amount
130:       dec = abs << shift;  // shift to normalized position
131:       dec &= (1 << m) - 1;  // remove implied 1
132:       dec |= (thr_e - shift) << m;  // insert biased exponent
133:     }
134:     dec |= enc << (e + m);  // insert sign bit
135:   }
136:   return dec;
137: }
```

```text
180:     const int logn = in[i];
181:     const int beg = i * chunksize;
182:     const int end = beg + chunksize;
183:     if (logn == 0) {
184:       for (int j = beg + lane; j < end; j += WS) {
185:         out_t[j] = 0;
186:       }
187:     } else if (logn == TB) {
188:       const int offs = bits[i] / TB - beg;
189:       for (int j = beg + lane; j < end; j += WS) {
190:         type_u val = in_t[offs + j];
191:         out_t[j] = deQuantize(val, thr_e, offs);
192:       }
193:     } else {
194:       const int incr = WS * logn;
```

```text
291: static __global__ __launch_bounds__(TPB, 4)
292: #else
293: static __global__ __launch_bounds__(TPB, 3)
294: #endif
295: void d_decode(const byte* const __restrict__ input, byte* const __restrict__ output, long long* const __restrict__ g_outsize, const int thr_e, const int offs)
296: {
297:   // allocate shared memory buffer
298:   __shared__ long long chunk [2 * (CS / sizeof(long long)) + 16];
299:   const int last = 2 * (CS / sizeof(long long)) + 1;
300:
301:   // input header
302:   long long* const head_in = (long long*)input;
303:   const long long outsize = head_in[0];
304:
305:   // initialize
```

```text
330:
331:     // create the 3 shared memory buffers
332:     byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
333:     byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
334:     byte* temp = (byte*)&chunk[2 * (CS / sizeof(long long))];
335:
336:     const long long offs = prevOffset + block_sum_reduction(sum, (long long*)out); //chunk[last + 1]);
337:     prevChunkID = chunkID;
338:     prevOffset = offs;
339:     __syncthreads();
340:
341:     // load chunk
342:     g2s(in, &data_in[offs], csize, out);
343:     __syncthreads();  // chunk produced, chunk[last] consumed
344:
345:     // decode
346:     const int osize = (int)min((long long)CS, outsize - base);
347:     if (csize < osize) {
348:       d_iSLEEK(csize, in, out,temp, thr_e, offs);
349:     } else {
350:       byte* tmp = in; in = out; out = tmp; // swap if no decode
```

## lossy_compressors_cpu/sleek_single_decompressor_lossy.cpp

SHA256 `18827459a14fffe1ffcb26dc507636ce115ec6799d413db40e7a788d54c03a1d`

[Source](https://github.com/burtscher/SLEEK/blob/00ff130d8d3182204b6b63fc22d5c85ad6b5f2e1/lossy_compressors_cpu/sleek_single_decompressor_lossy.cpp)

```text
152: static void h_decode(const byte* const __restrict__ input, byte* const __restrict__ output, long long& outsize, const int thr_e, const int offs)
153: {
154:   // input header
155:   long long* const head_in = (long long*)input;
156:   outsize = head_in[0];
157:
158:   // initialize
159:   const long long chunks = (outsize + CS - 1) / CS;  // round up
160:   unsigned short* const size_in = (unsigned short*)&head_in[1];
161:   byte* const data_in = (byte*)&size_in[chunks];
162:   long long* const start = new long long [chunks];
163:
164:   // convert chunk sizes into starting positions
165:   long long pfs = 0;
166:   for (long long chunkID = 0; chunkID < chunks; chunkID++) {
```

```text
179:     const long long base = chunkID * CS;
180:     const int osize = (int)std::min((long long)CS, outsize - base);
181:     int csize = size_in[chunkID];
182:     if (csize == osize) {
183:       // simply copy
184:       memcpy(&output[base], &data_in[start[chunkID]], osize);
185:     } else {
186:       // decompress
187:       memcpy(in, &data_in[start[chunkID]], csize);
188:       h_iSLEEK(csize, in, out, thr_e, offs);
189:       if (csize != osize) {fprintf(stderr, "ERROR: csize %d does not match osize %d in chunk %lld\n\n", csize, osize, chunkID);}
190:       memcpy(&output[base], out, csize);
191:     }
192:   }
193:
194:   // finish
195:   delete [] start;
```

## Source license

```text
BSD 3-Clause License

Copyright (c) 2026, Anju Mongandampulath Akathoott, Andrew Rodriguez, and Martin Burtscher

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
