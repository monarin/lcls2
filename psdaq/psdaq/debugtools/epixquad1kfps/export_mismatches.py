"""Export recorded gain-bit mismatches as candidates for a future known-bad list.

Offline use: python -m psdaq.debugtools.epixquad1kfps.export_mismatches
ATTEMPT --outdir NEW_DIRECTORY. Original capture records are never rewritten.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from psdaq.configdb.epixquad_layout import RAW_SHAPE
from psdaq.debugtools.epixquad1kfps.standalone_profile import file_sha256
from psdaq.debugtools.epixquad1kfps.write_gain_mode_standalone import _raw_to_direct


def export_mismatches(attempt, outdir, record=None):
    """Keep every observed mismatch; promotion/filtering is a later decision."""
    attempt, outdir = Path(attempt).resolve(), Path(outdir)
    if record is None:
        record = json.loads((attempt/'test.json').read_text())
    frames = record['analysis']['valid_images']
    expected = np.load(attempt/'expected-map.npy', allow_pickle=False)
    with np.load(attempt/'mismatches.npz', allow_pickle=False) as data:
        fp, fn = data['fp'], data['fn']
    if expected.shape != RAW_SHAPE or expected.dtype != np.bool_:
        raise ValueError('Expected map must be a boolean DAQ raw map')
    if type(frames) is not int or frames < 0:
        raise ValueError('Invalid analyzed frame count')
    for counts in (fp, fn):
        if (counts.shape != RAW_SHAPE or not np.issubdtype(counts.dtype, np.integer)
                or np.any(counts < 0) or np.any(counts > frames)):
            raise ValueError('Mismatch counts must match raw shape and frame count')
    if np.any(fp[expected]) or np.any(fn[~expected]):
        raise ValueError('Mismatch polarity disagrees with expected map')

    candidates = (fp != 0) | (fn != 0)
    outdir.mkdir(parents=True, exist_ok=False)
    fields = ['segment', 'raw_row', 'raw_col', 'asic', 'asic_row', 'asic_col',
              'kind', 'expected_gainbit', 'observed_gainbit', 'occurrences',
              'valid_images', 'fraction']
    with (outdir/'mismatches.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for segment, row, col in np.argwhere(candidates):
            coord = int(segment), int(row), int(col)
            asic, asic_row, asic_col = _raw_to_direct(*coord)
            is_fp = bool(fp[coord])
            count = int(fp[coord] if is_fp else fn[coord])
            writer.writerow(dict(zip(fields, [*coord, asic, asic_row, asic_col,
                'FP' if is_fp else 'FN', int(expected[coord]), int(is_fp),
                count, frames, count / frames])))
    np.savez_compressed(outdir/'known-bad-candidates.npz', mask=candidates)
    metadata = {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'exporter_sha256': file_sha256(Path(__file__)),
        'classification': 'observed candidates; not an approved known-bad list',
        'policy': 'Any gain-bit mismatch in any valid image; no frame exclusions',
        'coordinate_system': 'zero-based DAQ raw (segment,row,col); ASIC programming coordinates',
        'raw_shape': list(RAW_SHAPE), 'gainbit_mask': '0x4000',
        'kind_definition': 'FP: expected 0, observed 1; FN: expected 1, observed 0',
        'candidate_pixels': int(np.count_nonzero(candidates)),
        'valid_images': frames, 'has_image_evidence': frames > 0,
        'source_attempt': str(attempt),
        'source': {key: record.get(key) for key in (
            'test_id', 'status', 'host', 'label', 'repetition', 'mode',
            'gain_writer', 'requested_rate_hz', 'trigger_readback',
            'parent_test_id', 'phase_index', 'map_written_this_phase', 'map_write_phase',
            'provenance_directory',
            'started_utc', 'finished_utc', 'requested_frames', 'camera_identity',
            'pcie_identity', 'data', 'analysis', 'error', 'analysis_error', 'cleanup_errors')},
        'source_files_sha256': {name: file_sha256(attempt/name) for name in (
            'mismatches.npz', 'expected-map.npy', 'profile.json', 'software.json')
            if (attempt/name).is_file()},
        'output_files_sha256': {name: file_sha256(outdir/name) for name in (
            'mismatches.csv', 'known-bad-candidates.npz')},
    }
    (outdir/'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True)+'\n')
    return {'directory': str(outdir.resolve()),
            'candidate_pixels': metadata['candidate_pixels'],
            'metadata_sha256': file_sha256(outdir/'metadata.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path)
    parser.add_argument('--outdir', required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(export_mismatches(args.attempt, args.outdir), indent=2))


if __name__ == '__main__':
    main()
