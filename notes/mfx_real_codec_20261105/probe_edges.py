"""Additional codec diagnostic using large floats and IEEE special values.

Separate from the MFX distribution gate; does not alter compressor/decoder code.
"""
import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np

from trial import BOUND, bind, comparison, require, validate_blob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    gpu, cpu = bind(args.root)
    samples = np.array([0, 1, -3, 1e9, -1e9, np.inf, -np.inf, np.nan], dtype=np.float32)
    source = np.tile(samples, 16384)
    data = cp.asarray(source)
    ctx = gpu.codec_create(data.nbytes, BOUND)
    try:
        capacity = gpu.codec_max_size(ctx)
        encoded = cp.zeros(capacity + 8, cp.uint8)
        state, index = cp.ones(1, cp.uint32), cp.zeros(1, cp.uint32)
        require(gpu.codec_encode(ctx, state.data.ptr, index.data.ptr, data.data.ptr,
                                data.nbytes, encoded.data.ptr + 8, capacity, 0) == 0, 'encode')
        cp.cuda.Stream.null.synchronize()
        length = int(encoded[:8].view(cp.int64).get()[0])
        require(int(state.get()[0]) == 2 and 0 < length <= capacity, 'encoder completion')
        blob = encoded[8:8 + length].get()
        validate_blob(blob, data.nbytes)
        host = np.empty_like(source)
        require(cpu.codec_decode_cpu(blob.ctypes.data, host.ctypes.data, BOUND) == data.nbytes, 'cpu size')
        device, size = cp.empty_like(data), cp.zeros(1, cp.int64)
        require(gpu.codec_decode_gpu(encoded.data.ptr + 8, device.data.ptr, size.data.ptr,
                                    BOUND, 0) == 0, 'decode')
        cp.cuda.Stream.null.synchronize()
        require(int(size.get()[0]) == data.nbytes, 'gpu size')
        decoded = device.get()
        report = {'decoded_bytes': data.nbytes, 'compressed_bytes': length,
                  'cpu': comparison(source, host), 'gpu': comparison(source, decoded),
                  'decoder_bit_mismatches': int(np.count_nonzero(host.view(np.uint32) != decoded.view(np.uint32))),
                  'first_eight_bits': {k: [f'{x:08x}' for x in a[:8].view(np.uint32)]
                                       for k, a in [('input', source), ('cpu', host), ('gpu', decoded)]}}
        with args.output.open('x') as output:
            json.dump(report, output, indent=2)
            output.write('\n')
        print(json.dumps(report), flush=True)
    finally:
        gpu.codec_destroy(ctx)


if __name__ == '__main__':
    main()
