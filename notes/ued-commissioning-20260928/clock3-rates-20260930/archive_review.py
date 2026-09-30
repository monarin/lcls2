"""Review and archive the authorized clock-3 captures without hardware access."""
from collections import Counter
import gzip
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

folder = Path(__file__).resolve().parent
commissioning = folder.parent
results = []
for attempt in (11, 12):
    parent = commissioning / ('attempt%02d-mapfhl' % attempt)
    top = json.loads((parent / 'test.json').read_text())
    sources = [parent / phase['path'] for phase in top.get('phases', [])] or [parent]
    shutil.copyfile(parent / 'software.json', folder / ('software-attempt%d.json' % attempt))
    (folder / ('test-attempt%d.json.gz' % attempt)).write_bytes(
        gzip.compress((parent / 'test.json').read_bytes(), mtime=0))
    for source in sources:
        label = 'attempt%d%s' % (attempt, '-' + source.name if source != parent else '')
        review = source / 'review-after-first'
        if not review.exists():
            with (folder / (label + '-gain-review.log')).open('w') as log:
                subprocess.run([sys.executable, str(commissioning / 'review_gain_map.py'),
                                str(source), str(review)], stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        with (folder / (label + '-payload-review.log')).open('w') as log:
            subprocess.run([sys.executable, str(commissioning / 'review_payloads.py'),
                            str(source / 'capture.dat'), str(folder / (label + '-payloads.json'))],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        for name in ('review.json', 'after-first-mismatches.csv'):
            shutil.copyfile(review / name, folder / (label + '-' + name))
        (folder / (label + '-test.json.gz')).write_bytes(
            gzip.compress((source / 'test.json').read_bytes(), mtime=0))
        record = json.loads((source / 'test.json').read_text())
        gain = json.loads((review / 'review.json').read_text())
        sequences = []
        flags = Counter()
        with (source / 'capture.dat').open('rb') as stream:
            while True:
                header = stream.read(8)
                if not header:
                    break
                length, flag, error, channel = struct.unpack('<IHBB', header)
                payload = stream.read(length - 4)
                flags[(flag, error)] += 1
                if channel == 1 and not error and len(payload) in (1095248, 1095288):
                    # Acq/Rdout RTL: 32-byte camera header, sequence at byte 8.
                    # Existing layout decoder establishes image start at 40/80.
                    image_start = 40 + len(payload) - 1095248
                    sequences.append(struct.unpack_from('<I', payload, image_start - 24)[0])
        jumps = [{'image_1based': i + 2, 'before': a, 'after': b}
                 for i, (a, b) in enumerate(zip(sequences, sequences[1:]))
                 if (b - a) % (2**32) != 1]
        results.append({
            'label': label, 'source': str(source), 'status': record['status'],
            'data': record['data'], 'requested_rate_hz': record['requested_rate_hz'],
            'readout_clock': record['readout_clock'], 'trigger_readback': record['trigger_readback'],
            'stream_counts': record['stream_counts'], 'cleanup_errors': record['cleanup_errors'],
            'map_written_this_phase': record['map_written_this_phase'],
            'diagnostics_before': record['diagnostics_before'],
            'diagnostics_after': record['diagnostics_after'],
            'gain_after_first': gain['after_first'], 'assessed_frames': gain['assessed_frames'],
            'first_frame': gain['frames'][0],
            'sequence_first': sequences[0], 'sequence_last': sequences[-1],
            'sequence_discontinuities': jumps,
            'record_flags_errors': [{'flags': flag, 'error': error, 'count': count}
                                    for (flag, error), count in sorted(flags.items())],
        })
    shutil.copyfile(parent / 'console.log', folder / ('console-attempt%d.log' % attempt))
(folder / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
for result in results:
    print(result['label'], result['status'], result['requested_rate_hz'],
          result['stream_counts'], result['gain_after_first'],
          'sequence jumps:', result['sequence_discontinuities'])
