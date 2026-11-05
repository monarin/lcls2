"""Read back every stored blob, checking index coverage, framing and hashes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from trial import PANELS, require, validate_blob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.directory / 'report.json').read_text())
    require(report['status'] == 'pass', 'trial did not pass the MFX sample checks')
    count = report['events_requested']
    require(report['events_completed'] == count and len(report['records']) == 5 * count,
            'incomplete event/stream coverage')
    timestamps, positions, hashes = [], dict.fromkeys(PANELS, 0), {}
    for event in range(count):
        rows = report['records'][5 * event:5 * (event + 1)]
        require([r['stream'] for r in rows] == list(PANELS), 'stream ordering')
        require(all(r['event'] == event and r['timestamp'] == rows[0]['timestamp'] for r in rows),
                'event/timestamp mismatch')
        timestamps.append(rows[0]['timestamp'])
        for row in rows:
            stream = row['stream']
            require(row['calibration_bitwise_equal'] and row['cpu']['pass'] and row['gpu']['pass']
                    and row['decoder_bit_mismatches'] == 0, 'failed numerical check')
            require(row['panels'] == PANELS[stream] and row['decoded_dtype'] == '<f4'
                    and row['decoded_shape'] == [len(PANELS[stream]), 512, 1024]
                    and row['decoded_bytes'] == len(PANELS[stream]) * 512 * 1024 * 4, 'decoded layout')
            require(row['payload_file'] == f's{stream:03d}.sleek', 'unexpected filename')
            require(row['payload_offset'] == positions[stream], 'gap/overlap in file index')
            path = args.directory / row['payload_file']
            with path.open('rb') as source:
                source.seek(row['payload_offset'])
                blob = source.read(row['compressed_bytes'])
            require(len(blob) == row['compressed_bytes'], 'short payload read')
            require(hashlib.sha256(blob).hexdigest() == row['compressed_sha256'], 'payload hash mismatch')
            validate_blob(blob, row['decoded_bytes'])
            hashes.setdefault(stream, hashlib.sha256()).update(blob)
            positions[stream] += len(blob)
    require(all(a < b for a, b in zip(timestamps, timestamps[1:])), 'nonmonotonic timestamps')
    for stream, size in positions.items():
        require((args.directory / f's{stream:03d}.sleek').stat().st_size == size, 'trailing data')
    per_event = np.array([sum(r['compressed_bytes'] for r in report['records'][i:i + 5])
                          for i in range(0, 5 * count, 5)])
    result = {'status': 'pass', 'events': count, 'blobs': 5 * count,
              'compressed_bytes': int(per_event.sum()), 'mean_bytes_per_event': float(per_event.mean()),
              'min_bytes_per_event': int(per_event.min()), 'max_bytes_per_event': int(per_event.max()),
              'raw_bytes_per_event': 32 * 512 * 1024 * 2,
              'decoded_bytes_per_event': 32 * 512 * 1024 * 4,
              'first_timestamp': timestamps[0], 'last_timestamp': timestamps[-1],
              'max_error': max(r['gpu']['max_abs_error'] for r in report['records']),
              'files': {f's{s:03d}.sleek': {'bytes': positions[s], 'sha256': hashes[s].hexdigest()}
                        for s in PANELS},
              'report_sha256': hashlib.sha256((args.directory / 'report.json').read_bytes()).hexdigest()}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
