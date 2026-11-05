"""One retained MFX event: CPU lossless LZ4 size, with/without bitshuffle.

No XTC writing, calibration, performance claims or GPU codec equivalence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np


def digest(data):
    return hashlib.sha256(memoryview(np.ascontiguousarray(data)).cast('B')).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--packages', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.packages.resolve()))
    import bitshuffle
    import bitshuffle.ext
    import lz4
    import lz4.block

    repo = Path(__file__).resolve().parents[2]
    gate_path = repo / 'notes/mfx_real_codec_20261105/gate-result.json'
    gate = json.loads(gate_path.read_text())
    raw = np.load(args.raw, mmap_mode='r', allow_pickle=False)
    require(raw.shape == (32, 512, 1024) and raw.dtype == np.dtype('<u2'), 'wrong raw layout')
    rows = gate['records']
    require(len(rows) == 5 and sorted(p for r in rows for p in r['panels']) == list(range(32)),
            'invalid reference panel coverage')
    report = {'status': 'running', 'events': 1, 'timestamp': rows[0]['timestamp'],
              'source': str(args.raw.resolve()), 'source_raw_sha256': digest(raw),
              'reference_sha256': hashlib.sha256(gate_path.read_bytes()).hexdigest(),
              'python': sys.version, 'host': platform.node(), 'numpy': np.__version__,
              'bitshuffle': bitshuffle.__version__, 'lz4': lz4.__version__,
              'lz4_library': lz4.library_version_string(),
              'bitshuffle_extension_sha256': hashlib.sha256(Path(bitshuffle.ext.__file__).read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'block_bytes': 8192, 'block_elements': 4096,
              'size_accounting': 'Codec bytes plus four-byte length per 8KiB block; excludes XTC and external shape/index metadata.',
              'scope': 'CPU libraries, one original event, five original DRP panel groups; not nvCOMP or a throughput measurement.',
              'records': []}
    for row in rows:
        require(row['timestamp'] == report['timestamp'] and row['event'] == 0, 'mixed reference events')
        data = np.ascontiguousarray(raw[row['panels']])
        require(digest(data) == row['raw_sha256'], 'raw differs from validated MFX reference')
        source_bytes = data.tobytes()
        plain_bytes = 0
        for offset in range(0, len(source_bytes), 8192):
            block = source_bytes[offset:offset + 8192]
            compressed = lz4.block.compress(block, mode='default', store_size=False)
            require(lz4.block.decompress(compressed, uncompressed_size=len(block)) == block,
                    'plain LZ4 round-trip mismatch')
            plain_bytes += len(compressed) + 4  # Match bitshuffle's per-block length overhead.
        compressed = bitshuffle.compress_lz4(data, block_size=4096)
        decoded = bitshuffle.decompress_lz4(compressed, data.shape, data.dtype, block_size=4096)
        require(decoded.dtype == data.dtype and decoded.shape == data.shape and
                np.array_equal(decoded, data), 'bitshuffle+LZ4 round-trip mismatch')
        report['records'].append({'stream': row['stream'], 'panels': row['panels'],
                                  'raw_sha256': digest(data), 'raw_bytes': data.nbytes,
                                  'lz4_bytes': plain_bytes, 'bitshuffle_lz4_bytes': compressed.nbytes,
                                  'roundtrip_bitwise_equal': True})
    raw_bytes = sum(r['raw_bytes'] for r in report['records'])
    report['totals'] = {'raw_bytes': raw_bytes}
    for codec in ('lz4', 'bitshuffle_lz4'):
        compressed_bytes = sum(r[codec + '_bytes'] for r in report['records'])
        report['totals'][codec] = {'bytes': compressed_bytes, 'factor': raw_bytes / compressed_bytes}
    report['status'] = 'pass'
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(report['totals'], indent=2))


if __name__ == '__main__':
    main()
