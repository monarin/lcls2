"""Serial feasibility test, not a throughput benchmark or an XTC producer.

Runs unchanged DRP SLEEK encoder and upstream CPU/GPU decoders on trusted data.
Large outputs go to a new private directory; no database/network writes.
"""
import argparse
import ast
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np

PANELS = {5: [3, 7, 0, 1, 2], 6: [19, 15, 11, 31, 27, 23],
          7: [18, 14, 10, 6, 30, 26, 22], 8: [16, 12, 8, 4, 28, 24, 20],
          9: [17, 13, 9, 5, 29, 25, 21]}
BOUND = 3.0  # Same constructor argument as SleekReducer.cu; calibrated units.
FROZEN = Path('/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/r0001/stage1-20261103')


def digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def validate_blob(blob, decoded_bytes):
    """Check outer framing before passing our encoder output to upstream code.

    This is not a general validator for untrusted compressed input.
    """
    chunks = (decoded_bytes + 16383) // 16384
    require(len(blob) >= 8 + 2 * chunks, 'short codec header')
    require(int(np.frombuffer(blob[:8], dtype='<i8')[0]) == decoded_bytes,
            'wrong decoded size in codec header')
    sizes = np.frombuffer(blob[8:8 + 2 * chunks], dtype='<u2')
    limits = np.minimum(16384, decoded_bytes - np.arange(chunks) * 16384)
    require(bool(np.all((sizes > 0) & (sizes <= limits))), 'invalid chunk sizes')
    require(8 + 2 * chunks + int(sizes.sum()) == len(blob), 'wrong encoded length')


def comparison(reference, result):
    both_finite = np.isfinite(reference) & np.isfinite(result)
    same_class = (both_finite | (np.isnan(reference) & np.isnan(result)) |
                  (np.isposinf(reference) & np.isposinf(result)) |
                  (np.isneginf(reference) & np.isneginf(result)))
    # Float64 subtraction avoids overflow/rounding in the error measurement.
    error = np.abs(reference[both_finite].astype(np.float64) - result[both_finite].astype(np.float64))
    nonfinite_bad = int(np.count_nonzero(~same_class))
    bound_bad = int(np.count_nonzero(error > BOUND))
    return {'max_abs_error': float(error.max(initial=0)),
            'bound_violations': bound_bad, 'nonfinite_mismatches': nonfinite_bad,
            'pass': bound_bad == 0 and nonfinite_bad == 0}


class Batch:
    def __init__(self, raw, constants, cp):
        self.raw, self.constants = raw, constants
        self.presence = cp.ones(raw.shape[:2], cp.uint8)
        self.size = raw.shape[0]
        self.owners = []

    def input(self, name): return self.raw
    def present(self, name): return self.presence
    def segment_ids(self, name): return tuple(range(32))
    def calibconst(self, det, key): return self.constants[key]
    def keepalive(self, value): self.owners.append(value)


def bind(root):
    gpu = ct.CDLL(str(root / 'libcodec_gpu.so'))
    cpu = ct.CDLL(str(root / 'libcodec_cpu.so'))
    spec = [(gpu, 'codec_create', [ct.c_size_t, ct.c_float], ct.c_void_p),
            (gpu, 'codec_max_size', [ct.c_void_p], ct.c_longlong),
            (gpu, 'codec_destroy', [ct.c_void_p], None),
            (gpu, 'codec_encode', [ct.c_void_p] * 4 + [ct.c_longlong, ct.c_void_p,
                                                      ct.c_longlong, ct.c_void_p], ct.c_int),
            (gpu, 'codec_decode_gpu', [ct.c_void_p] * 3 + [ct.c_float, ct.c_void_p], ct.c_int),
            (cpu, 'codec_decode_cpu', [ct.c_void_p, ct.c_void_p, ct.c_float], ct.c_longlong)]
    for lib, name, args, result in spec:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    return gpu, cpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--events', type=int, default=1)
    args = parser.parse_args()
    require(1 <= args.events <= 1000, 'events must be in 1..1000')
    require(os.environ.get('PS_FAKESTEP_FLAG') == '0' and os.environ.get('PS_PARALLEL') == 'none',
            'use the pinned runtime wrapper')
    os.umask(0o077)
    args.output.mkdir(mode=0o700)  # Refuse overwrite, even of failed attempts.
    import cupy as cp
    from psana import DataSource
    repo = Path(__file__).resolve().parents[2]
    kernel_file = repo / 'psana/psana/gpu/examples/jungfrau_calibration.py'
    spec = importlib.util.spec_from_file_location('trial_calibration', kernel_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task = module.JungfrauCalibration(use_offset=True)
    manifest = json.loads((repo / 'notes/synthetic_mfx_stage1_20261103/manifest.json').read_text())
    constants, provenance = {}, {}
    for item in manifest['calibration']:
        doc = item['source_document']
        if item['detector'] != 'jungfrau' or doc['ctype'] not in ('pedestals', 'pixel_gain', 'pixel_offset'):
            continue
        array = np.memmap(FROZEN / item['file'], dtype=doc['data_dtype'], mode='r',
                          shape=ast.literal_eval(doc['data_shape']))
        require(digest(array) == item['sha256'], 'frozen constant digest mismatch')
        constants[doc['ctype']] = (array, doc)
        provenance[doc['ctype']] = {'id': doc['_id'], 'sha256': item['sha256'],
                                     'file': str(FROZEN / item['file'])}
    require(len(constants) == 3, 'missing constants')
    device_constants = {k: cp.asarray(v[0]) for k, v in constants.items()}
    gpu, cpu = bind(args.root)
    stream = cp.cuda.Stream.null
    report = {'status': 'running', 'events_requested': args.events, 'events_completed': 0,
              'source': {'exp': 'mfx101555026', 'run': 10, 'dir': '/sdf/data/lcls/ds/prj/public01/xtc'},
              'calibration': provenance, 'settings': {'use_offset': True, 'status': False, 'cmpars': None},
              'bound': BOUND, 'panel_order': PANELS, 'cupy': cp.__version__,
              'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(),
              'kernel_sha256': hashlib.sha256(kernel_file.read_bytes()).hexdigest(),
              'gpu': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), 'records': []}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    start = time.monotonic()
    ds = DataSource(**report['source'], max_events=args.events, live=False, skip_calib_load='all')
    run = next(ds.runs())
    det = run.Detector('jungfrau', status=False)
    # Offline injection of the byte-verified Stage-1 DB snapshot, before any cache use.
    det.raw._calibconst = constants
    require(list(det.raw._segment_numbers) == list(range(32)), 'unexpected dense panel order')
    for event_index, evt in enumerate(run.events()):
        raw = det.raw.raw(evt)
        require(raw.dtype == np.uint16 and raw.shape == (32, 512, 1024), 'unexpected raw layout')
        reference = det.raw.calib(evt, cversion=3, cmpars=None).reshape(raw.shape)
        batch = Batch(cp.asarray(raw[np.newaxis]), device_constants, cp)
        calibrated, _ = task.calibrate(batch, stream)
        gpu_reference = calibrated[0].get()
        require(np.array_equal(reference.view(np.uint32), gpu_reference.view(np.uint32)),
                'GPU calibration differs from CPU v3 bits')
        # Keep one raw/calibrated witness; larger trials reference source by timestamp/hash.
        if event_index == 0:
            np.save(args.output / 'raw-event0.npy', raw)
            np.save(args.output / 'calibrated-event0.npy', reference)
        for source_stream, panels in PANELS.items():
            observed = getattr(evt._dgrams[source_stream], 'jungfrau')
            require(list(observed) == panels, 'source event panel order differs from Configure plan')
            expected = np.ascontiguousarray(reference[panels])
            data = cp.ascontiguousarray(calibrated[0, panels])
            ctx = gpu.codec_create(data.nbytes, BOUND)
            try:
                capacity = gpu.codec_max_size(ctx)
                encoded = cp.zeros(capacity + 8, dtype=cp.uint8)
                state, index = cp.ones(1, cp.uint32), cp.zeros(1, cp.uint32)
                require(gpu.codec_encode(ctx, state.data.ptr, index.data.ptr, data.data.ptr,
                                        data.nbytes, encoded.data.ptr + 8, capacity, stream.ptr) == 0,
                        'encoder launch failed')
                stream.synchronize()
                length = int(encoded[:8].view(cp.int64).get()[0])
                require(int(state.get()[0]) == 2 and 0 < length <= capacity, 'encoder completion/size failure')
                blob = encoded[8:8 + length].get()
                validate_blob(blob, data.nbytes)
                payload_path = args.output / f's{source_stream:03d}.sleek'
                with payload_path.open('xb' if event_index == 0 else 'ab') as output:
                    offset = output.tell()
                    output.write(blob.tobytes())
                decoded_cpu = np.empty_like(expected)
                size = cpu.codec_decode_cpu(blob.ctypes.data, decoded_cpu.ctypes.data, BOUND)
                require(size == data.nbytes, 'CPU decoder size mismatch')
                decoded_gpu, decoded_size = cp.empty_like(data), cp.zeros(1, cp.int64)
                require(gpu.codec_decode_gpu(encoded.data.ptr + 8, decoded_gpu.data.ptr,
                                            decoded_size.data.ptr, BOUND, stream.ptr) == 0,
                        'GPU decoder launch failed')
                stream.synchronize()
                require(int(decoded_size.get()[0]) == data.nbytes, 'GPU decoder size mismatch')
                decoded_host = decoded_gpu.get()
                row = {'event': event_index, 'timestamp': int(evt.timestamp), 'stream': source_stream,
                       'panels': panels, 'raw_sha256': digest(raw[panels]),
                       'calibrated_sha256': digest(expected), 'calibration_bitwise_equal': True,
                       'compressed_sha256': digest(blob), 'compressed_bytes': length,
                       'payload_file': payload_path.name, 'payload_offset': offset,
                       'decoded_shape': list(expected.shape), 'decoded_dtype': '<f4',
                       'decoded_bytes': data.nbytes, 'cpu': comparison(expected, decoded_cpu),
                       'gpu': comparison(expected, decoded_host),
                       'decoder_bit_mismatches': int(np.count_nonzero(decoded_cpu.view(np.uint32) != decoded_host.view(np.uint32)))}
                report['records'].append(row)
                print(json.dumps(row), flush=True)
            finally:
                gpu.codec_destroy(ctx)
        report['events_completed'] += 1
        passed = all(r['cpu']['pass'] and r['gpu']['pass'] and r['decoder_bit_mismatches'] == 0
                     for r in report['records'])
        report['status'] = 'pass' if passed and report['events_completed'] == args.events else 'incomplete'
        report['elapsed_seconds'] = time.monotonic() - start
        if not passed:
            report['status'] = 'codec_gate_failed'
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        if not passed:
            break
    require(report['status'] == 'pass', 'codec gate failed or source event count incomplete; see report.json')


if __name__ == '__main__':
    main()
