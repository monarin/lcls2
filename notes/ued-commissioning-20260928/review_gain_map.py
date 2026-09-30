"""Assess map captures after frame 1, retaining full and known-bad counts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pyrogue.utilities.fileio import FileReader
from psdaq.configdb.epixquad_layout import RAW_SHAPE, rogue_payload_to_daq_raw
from psdaq.debugtools.epixquad1kfps.capture_standalone import check_file_integrity
from psdaq.debugtools.epixquad1kfps.standalone_profile import file_sha256
from psdaq.debugtools.epixquad1kfps.write_gain_mode_standalone import _raw_to_direct

parser = argparse.ArgumentParser()
parser.add_argument('attempt', type=Path)
parser.add_argument('output', type=Path, help='New review directory')
args = parser.parse_args()
folder = Path(__file__).resolve().parent
record = json.loads((args.attempt/'test.json').read_text())
baseline = json.loads((folder/'test-attempt04.json').read_text())
if record['camera_identity']['carrier_ids'] != baseline['camera_identity']['carrier_ids']:
    raise ValueError('Known-bad list belongs to a different detector')
expected = np.load(args.attempt/'expected-map.npy', allow_pickle=False)
known = np.zeros(RAW_SHAPE, dtype=bool)
known_path = folder/'fl-persistent-mismatch-candidates.csv'
with known_path.open() as stream:
    for row in csv.DictReader(stream):
        known[tuple(int(row[k]) for k in ('segment', 'raw_row', 'raw_col'))] = True
historical = {kind: np.zeros(RAW_SHAPE, dtype=bool) for kind in ('FP', 'FN')}
historical_path = folder/'historical-expand2-run2-mismatches.csv'
with historical_path.open() as stream:
    for row in csv.DictReader(stream):
        historical[row['kind']][tuple(int(row[k]) for k in ('segment', 'raw_row', 'raw_col'))] = True
data_path = args.attempt/'capture.dat'
check_file_integrity(data_path)
assert file_sha256(data_path) == record['data']['sha256']
all_fp = np.zeros(RAW_SHAPE, dtype=np.uint32)
all_fn = np.zeros_like(all_fp)
fp = np.zeros_like(all_fp)
fn = np.zeros_like(all_fp)
frames = []
for header, data in FileReader(str(data_path)).records():
    payload = np.asarray(data).view(np.uint8).tobytes()
    if header.channel != 1 or header.error or len(payload) not in (1095248, 1095288):
        continue
    raw, _ = rogue_payload_to_daq_raw(payload)
    low = (raw & 0x4000) != 0
    fpos, fneg = low & ~expected, ~low & expected
    all_fp += fpos
    all_fn += fneg
    if frames:
        fp += fpos
        fn += fneg
    frames.append({'image_1based': len(frames)+1,
                   'fp': int(fpos.sum()), 'fn': int(fneg.sum()),
                   'fp_outside_known': int(fpos[~known].sum()),
                   'fn_outside_known': int(fneg[~known].sum())})
if len(frames) < 2:
    raise ValueError('Need at least two full images for post-startup assessment')
with np.load(args.attempt/'mismatches.npz', allow_pickle=False) as saved:
    np.testing.assert_array_equal(all_fp, saved['fp'])
    np.testing.assert_array_equal(all_fn, saved['fn'])

def counts_summary(pos, neg):
    result = {}
    for kind, counts in (('FP', pos), ('FN', neg)):
        result[kind] = {'unique_pixels': int(np.count_nonzero(counts)),
                        'occurrences': int(counts.sum()),
                        'known_unique': int(np.count_nonzero(counts[known])),
                        'outside_known_unique': int(np.count_nonzero(counts[~known])),
                        'outside_known_occurrences': int(counts[~known].sum()),
                        'historical_run2_overlap': int(np.count_nonzero((counts > 0) & historical[kind])),
                        'outside_historical_run2': int(np.count_nonzero((counts > 0) & ~historical[kind]))}
    return result

args.output.mkdir(parents=True, exist_ok=False)
np.savez_compressed(args.output/'after-first-counts.npz', fp=fp, fn=fn, known_bad_mask=known)
fields = ['segment', 'raw_row', 'raw_col', 'asic', 'asic_row', 'asic_col', 'kind',
          'occurrences', 'assessed_frames', 'fraction', 'within_known_bad', 'in_historical_run2']
with (args.output/'after-first-mismatches.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for coord_values in np.argwhere((fp > 0) | (fn > 0)):
        coord = tuple(int(v) for v in coord_values)
        kind = 'FP' if fp[coord] else 'FN'
        count = int((fp if kind == 'FP' else fn)[coord])
        writer.writerow(dict(zip(fields, [*coord, *_raw_to_direct(*coord), kind,
            count, len(frames)-1, count/(len(frames)-1), int(known[coord]), int(historical[kind][coord])])) )
review = {'source_attempt': str(args.attempt.resolve()), 'mode': record['mode']['name'],
          'gain_writer': record.get('gain_writer'),
          'requested_rate_hz': record.get('requested_rate_hz'),
          'trigger_readback': record.get('trigger_readback'),
          'parent_test_id': record.get('parent_test_id'),
          'phase_index': record.get('phase_index'),
          'source_status': record['status'], 'data_sha256': record['data']['sha256'],
          'reviewer_sha256': file_sha256(Path(__file__)),
          'policy': 'First full image retained but excluded from assessment by user instruction',
          'recorded_frames': len(frames), 'assessed_frames': len(frames)-1,
          'known_bad_pixels': int(known.sum()), 'selected_low_pixels': int(expected.sum()),
          'selected_known_bad': int((expected & known).sum()),
          'known_bad_csv_sha256': file_sha256(known_path),
          'historical_run2_csv_sha256': file_sha256(historical_path),
          'all_frames': counts_summary(all_fp, all_fn),
          'after_first': counts_summary(fp, fn), 'frames': frames,
          'output_sha256': {name: file_sha256(args.output/name) for name in
                            ('after-first-counts.npz', 'after-first-mismatches.csv')}}
(args.output/'review.json').write_text(json.dumps(review, indent=2)+'\n')
print(json.dumps({k:v for k,v in review.items() if k!='frames'}, indent=2))
