"""Read-only, bounded XTC2 gain-bit review; never waits for an active file."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
from psana import dgram


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('xtc', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    evidence = Path(__file__).resolve().parent.parent / 'ued-commissioning-20260928'
    mask_path = evidence / 'map-inputs/expand2.npy'
    known_path = evidence / 'fl-persistent-mismatch-candidates.csv'
    history_path = evidence / 'historical-expand2-run2-mismatches.csv'
    expected = np.load(mask_path, allow_pickle=False).astype(bool)
    assert expected.shape == (4, 352, 384)
    known = np.zeros_like(expected)
    historical = {k: np.zeros_like(expected) for k in ('FP', 'FN')}
    with known_path.open() as stream:
        for row in csv.DictReader(stream):
            known[tuple(int(row[k]) for k in ('segment', 'raw_row', 'raw_col'))] = True
    with history_path.open() as stream:
        for row in csv.DictReader(stream):
            historical[row['kind']][tuple(int(row[k]) for k in ('segment', 'raw_row', 'raw_col'))] = True
    identity = json.loads((evidence / 'test-attempt04.json').read_text())['camera_identity']['carrier_ids']
    assert known.sum() == 352
    args.output.mkdir(parents=True, exist_ok=False)
    counts = {scope: {k: np.zeros(expected.shape, np.uint32) for k in ('FP', 'FN')}
              for scope in ('all_good_frames', 'after_first')}
    frames, incomplete, damaged = [], [], []
    services = Counter()
    config = None
    config_summary = None
    digest = hashlib.sha256()
    offset = 0
    full_images = 0
    snapshot_size = args.xtc.stat().st_size
    with args.xtc.open('rb') as stream:
        while offset + 24 <= snapshot_size:
            header = stream.read(24)
            if len(header) != 24:
                break
            size = 12 + struct.unpack_from('<I', header, 20)[0]
            if size < 24 or size > 64 * 1024 * 1024:
                raise ValueError(f'Invalid datagram size {size} at {offset}')
            if offset + size > snapshot_size:
                break
            payload = header + stream.read(size - 24)
            if len(payload) != size:
                break
            service = (struct.unpack_from('<I', header, 8)[0] >> 24) & 15
            dg = dgram.Dgram(view=payload) if service == 2 else dgram.Dgram(view=payload, config=config)
            assert dg.service() == service
            digest.update(payload)
            offset += size
            services[service] += 1
            if service == 2:
                if config is not None:
                    raise ValueError('Multiple Configure datagrams: review each configuration separately')
                config = dg
                pixels = np.stack([dg.epixquad1kfps[s].config.asicPixelConfig for s in range(4)])
                trbits = np.stack([dg.epixquad1kfps[s].config.trbit for s in range(4)])
                np.testing.assert_array_equal(pixels, np.where(expected, 8, 12))
                np.testing.assert_array_equal(trbits, np.ones((4, 4), dtype=np.uint8))
                detids = [dg.software.epixquad1kfps[s].detid for s in range(4)]
                carriers = [[int(x) for x in detid.split('-')[1:3]] for detid in detids]
                assert carriers == identity, 'Known-bad list is for a different detector'
                config_summary = {'mode': 'MapFHL', 'map_matches_expand2': True,
                                  'trbits': trbits.tolist(), 'detids': detids,
                                  'carrier_ids_match_known_bad_detector': True,
                                  'configure_sha256': hashlib.sha256(payload).hexdigest()}
                continue
            if service != 12:
                continue
            index = services[12]
            segments = getattr(dg, 'epixquad1kfps', {})
            if set(segments) != set(range(4)) or any(
                    not hasattr(segments[s], 'raw') or
                    np.shape(segments[s].raw.raw) != (352, 384) for s in segments):
                incomplete.append(index)
                continue
            full_images += 1
            damage = [int(dg._xtc.damage)]
            for seg in segments.values():
                damage.extend((int(seg._xtc.damage), int(seg.raw._xtc.damage)))
            if any(damage):
                damaged.append({'event_1based': index, 'damage': damage})
                continue
            raw = np.stack([segments[s].raw.raw for s in range(4)])
            low = (raw & 0x4000) != 0
            mismatch = {'FP': low & ~expected, 'FN': ~low & expected}
            row = {'event_1based': index, 'full_image_1based': full_images,
                   'timestamp': int(dg.timestamp()), 'assessed': full_images > 1,
                   'top2_counts': np.bincount((raw >> 14).ravel(), minlength=4).tolist()}
            for kind, bad in mismatch.items():
                counts['all_good_frames'][kind] += bad
                if full_images > 1:
                    counts['after_first'][kind] += bad
                row[kind] = int(bad.sum())
                row[kind + '_outside_known'] = int((bad & ~known).sum())
            frames.append(row)
            if len(frames) == 51 or len(frames) % 1000 == 0:
                print(json.dumps({'progress_frames': len(frames), 'latest': row}), flush=True)
    if len(frames) < 2:
        raise ValueError('Need at least two undamaged full images')
    assessed = sum(row['assessed'] for row in frames)
    summary = {}
    for scope, kinds in counts.items():
        summary[scope] = {}
        for kind, array in kinds.items():
            summary[scope][kind] = {
                'unique_pixels': int(np.count_nonzero(array)), 'occurrences': int(array.sum()),
                'known_unique': int(np.count_nonzero(array[known])),
                'outside_known_unique': int(np.count_nonzero(array[~known])),
                'outside_known_occurrences': int(array[~known].sum()),
                'historical_run2_overlap': int(np.count_nonzero((array > 0) & historical[kind]))}
    np.savez_compressed(args.output / 'counts.npz', known=known, expected_low=expected,
                        **{scope + '_' + kind: array for scope, kinds in counts.items()
                           for kind, array in kinds.items()})
    with (args.output / 'after-first-mismatches.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['segment', 'raw_row', 'raw_col', 'kind', 'occurrences',
                         'assessed_frames', 'within_known_bad', 'in_historical_run2'])
        for kind, array in counts['after_first'].items():
            for coord in np.argwhere(array):
                coord = tuple(int(x) for x in coord)
                writer.writerow([*coord, kind, int(array[coord]), assessed,
                                 int(known[coord]), int(historical[kind][coord])])
    with (args.output / 'frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frames[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(frames)
    patterns = Counter((r['FP'], r['FN'], r['FP_outside_known'], r['FN_outside_known'])
                       for r in frames if r['assessed'])
    report = {'input': str(args.xtc), 'snapshot_bytes': snapshot_size,
              'complete_datagram_bytes': offset, 'unprocessed_tail_bytes': snapshot_size - offset,
              'complete_prefix_sha256': digest.hexdigest(),
              'file_bytes_after_review': args.xtc.stat().st_size,
              'services': dict(services), 'endrun_seen': bool(services[5]),
              'scope': 'Complete datagrams present at start; in-progress file is not a finalized run',
              'policy': 'Exclude first full image and damaged/incomplete events from assessment',
              'config': config_summary, 'good_full_frames': len(frames),
              'assessed_frames': assessed, 'incomplete_events': incomplete, 'damaged_events': damaged,
              'known_bad_pixels': int(known.sum()), 'selected_low_pixels': int(expected.sum()),
              'selected_known_bad': int((known & expected).sum()), 'summary': summary,
              'input_sha256': {str(p): sha(p) for p in (mask_path, known_path, history_path, Path(__file__))},
              'first_frame': frames[0],
              'assessed_patterns': [dict(zip(('FP', 'FN', 'FP_outside_known', 'FN_outside_known', 'frames'),
                                            (*pattern, count))) for pattern, count in patterns.items()],
              'python': sys.executable, 'dgram_module': dgram.__file__}
    times = [(r['timestamp'] >> 32) + (r['timestamp'] & 0xffffffff) * 1e-9 for r in frames]
    report['timestamp_span_seconds'] = times[-1] - times[0]
    report['mean_frame_rate_hz'] = (len(times) - 1) / (times[-1] - times[0])
    report['output_sha256'] = {name: sha(args.output / name) for name in
                              ('counts.npz', 'after-first-mismatches.csv', 'frames.csv')}
    (args.output / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'frames'}, indent=2))


if __name__ == '__main__':
    main()
