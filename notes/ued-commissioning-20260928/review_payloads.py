"""Offline record histogram and ADC statistics for a commissioning capture."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from pyrogue.utilities.fileio import FileReader
from psdaq.configdb.epixquad_layout import rogue_payload_to_daq_raw
from psdaq.debugtools.epixquad1kfps.capture_standalone import check_file_integrity
from psdaq.debugtools.epixquad1kfps.standalone_profile import file_sha256

parser = argparse.ArgumentParser()
parser.add_argument('capture', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
check_file_integrity(args.capture)
histogram = Counter()
statistics = []
short_examples = []
for header, data in FileReader(str(args.capture)).records():
    payload = np.asarray(data).view(np.uint8).tobytes()
    histogram[(len(payload), int(header.error), int(header.channel))] += 1
    if len(payload) == 48 and len(short_examples) < 3:
        short_examples.append(payload.hex())
    if header.error or header.channel != 1 or len(payload) not in (1095248, 1095288):
        continue
    raw, _ = rogue_payload_to_daq_raw(payload)
    adc = raw & 0x3fff
    statistics.append({
        'mean_by_segment': adc.mean(axis=(1, 2)).tolist(),
        'min': int(adc.min()), 'max': int(adc.max()),
        'zero_fraction': float(np.mean(adc == 0)),
    })
result = {
    'data_file': str(args.capture.resolve()),
    'sha256': file_sha256(args.capture),
    'bytes': args.capture.stat().st_size,
    'records': [dict(payload_bytes=size, error=error, channel=channel, count=count)
                for (size, error, channel), count in sorted(histogram.items())],
    'image_statistics': statistics,
    'short_examples_hex': short_examples,
}
args.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({key: value for key, value in result.items()
                  if key not in ('image_statistics', 'short_examples_hex')}, indent=2))
print('First image:', statistics[:1])
print('Last image:', statistics[-1:])
