"""Offline frame-by-frame gain-bit evidence; never discard startup frames."""
import argparse
import json
from pathlib import Path

import numpy as np
from pyrogue.utilities.fileio import FileReader
from psdaq.configdb.epixquad_layout import RAW_ASIC_LAYOUT, rogue_payload_to_daq_raw
from psdaq.debugtools.epixquad1kfps.capture_standalone import check_file_integrity
from psdaq.debugtools.epixquad1kfps.standalone_profile import file_sha256

parser = argparse.ArgumentParser()
parser.add_argument('attempt', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
path = args.attempt/'capture.dat'
check_file_integrity(path)
expected = np.load(args.attempt/'expected-map.npy', allow_pickle=False)
total_fp = np.zeros(expected.shape, dtype=np.uint32)
total_fn = np.zeros_like(total_fp)
later = np.zeros_like(total_fp)
frames = []
for record_index, (header, data) in enumerate(FileReader(str(path)).records(), 1):
    payload = np.asarray(data).view(np.uint8).tobytes()
    if header.channel != 1 or header.error or len(payload) not in (1095248, 1095288):
        continue
    raw, _ = rogue_payload_to_daq_raw(payload)
    low = (raw & 0x4000) != 0
    fp, fn = low & ~expected, ~low & expected
    mismatch = fp | fn
    total_fp += fp
    total_fn += fn
    if frames:
        later += mismatch
    per_asic = [0] * 16
    for segment in range(4):
        for tile in RAW_ASIC_LAYOUT:
            r0, r1 = tile['row_slice']
            c0, c1 = tile['col_slice']
            per_asic[4*segment+tile['slot']] = int(mismatch[segment,r0:r1,c0:c1].sum())
    frames.append({'valid_image_index_1based': len(frames)+1,
                   'file_record_index_1based': record_index,
                   'fp': int(fp.sum()), 'fn': int(fn.sum()),
                   'mismatches_by_asic': per_asic})
with np.load(args.attempt/'mismatches.npz', allow_pickle=False) as saved:
    np.testing.assert_array_equal(total_fp, saved['fp'])
    np.testing.assert_array_equal(total_fn, saved['fn'])

def histogram(counts):
    values, frequencies = np.unique(counts[counts > 0], return_counts=True)
    return [{'mismatch_frames': int(v), 'pixels': int(n)} for v,n in zip(values,frequencies)]

counts = total_fp + total_fn
result = {'source_data': str(path.resolve()), 'sha256': file_sha256(path),
          'frame_numbering': '1-based full error-free images; all frames retained',
          'verified_against_saved_counts': True, 'frames': frames,
          'occurrence_histogram_all_frames': histogram(counts),
          'occurrence_histogram_after_first': histogram(later),
          'unique_pixels_all_frames': int(np.count_nonzero(counts)),
          'unique_pixels_after_first': int(np.count_nonzero(later)),
          'pixels_only_in_first': int(np.count_nonzero((counts > 0) & (later == 0)))}
args.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='frames'},indent=2))
print('Per-frame FP/FN:', [(f['fp'],f['fn']) for f in frames])
