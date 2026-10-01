"""Archive attempt17 evidence and inspect native framing/timing, without hardware."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import shutil
import struct

import numpy as np
from psdaq.configdb.epixquad_layout import event_batcher_subframes
from psdaq.debugtools.epixquad1kfps.capture_standalone import ImageCounts, check_file_integrity
from psdaq.debugtools.epixquad1kfps.standalone_profile import file_sha256

out = Path(__file__).resolve().parent
folder = out.parent
attempt = folder / 'attempt17-mapfhl'
review_dir = attempt / 'review-external'
review = json.loads((review_dir / 'review.json').read_text())
capture = attempt / 'capture.dat'
check_file_integrity(capture)
assert file_sha256(capture) == review['data_sha256']
assert file_sha256(attempt / 'test.json') == review['source_test_json_sha256']
assert file_sha256(attempt / 'expected-map.npy') == review['expected_map_sha256']
assert review['expected_map_sha256'] == file_sha256(folder / 'map-inputs/expand2.npy')

records = []
profile = json.loads((folder.parents[1] /
    'psdaq/psdaq/debugtools/epixquad1kfps/profiles/drp-ued-cmp003-external.json').read_text())
counts = ImageCounts(profile['image_payload_bytes'], 512 * 1024**2)
with capture.open('rb') as stream:
    while header := stream.read(8):
        length, flags, error, channel = struct.unpack('<IHBB', header)
        payload = stream.read(length - 4)
        assert error == 0 and channel == 1
        assert counts.room(len(payload))
        counts.observed(payload, error)
        subframes = event_batcher_subframes(payload)
        assert not any(f['last_user'] & 1 for f in subframes.values())
        # Pds::TimingHeader / PulseId (EbDgram.hh), then TimeStamp low/high.
        pulse, ns, sec = struct.unpack_from('<QII', payload, subframes[0]['offset'])
        assert ns < 10**9 and pulse >> 56 == 12  # L1Accept, no timing error bit
        record = dict(record_1based=len(records)+1, payload_bytes=len(payload),
                      timing_ns=sec*10**9+ns, pulse_id=pulse & 0x00ffffffffffffff,
                      camera_sequence=None, subframes=subframes)
        if 2 in subframes:
            record['camera_sequence'] = struct.unpack_from(
                '<I', payload, subframes[2]['offset']+8)[0]
        records.append(record)
images = [r for r in records if r['camera_sequence'] is not None]
assert len(images) == review['recorded_frames'] == 490
assert counts.snapshot()['valid_images'] == 490
assert counts.snapshot()['invalid_records'] == 2
assert counts.snapshot()['fault'] is None
sequence_deltas = Counter((b['camera_sequence']-a['camera_sequence']) & 0xffffffff
                         for a, b in zip(images, images[1:]))
time_deltas = [b['timing_ns']-a['timing_ns'] for a, b in zip(images, images[1:])]
assert sequence_deltas == {3: 489} and min(time_deltas) > 0

reference = folder.parent / 'ued-full-daq-20260930/run3-gainbits/after-first-mismatches.csv'
with reference.open() as stream:
    daq_fp = {tuple(int(row[k]) for k in ('segment', 'raw_row', 'raw_col'))
              for row in csv.DictReader(stream)
              if row['kind'] == 'FP' and int(row['within_known_bad']) == 0}
assert len(daq_fp) == 43
with np.load(review_dir/'after-first-counts.npz', allow_pickle=False) as data:
    reproduced = sorted(coord for coord in daq_fp if data['fp'][coord])
    assert not reproduced

result = dict(
    native_file=str(capture), native_sha256=review['data_sha256'],
    original_status=review['source_status'], original_stop='Capture byte limit reached',
    original_record_retained=True, full_images=len(images), assessed_images=len(images)-1,
    corrected_counter_offline_replay={key: counts.snapshot()[key] for key in
        ('valid_images', 'invalid_records', 'records', 'bytes', 'fault')},
    first_full_subframes=images[0]['subframes'],
    nonimage_records=[r for r in records if r['camera_sequence'] is None],
    first_timing_ns=images[0]['timing_ns'], last_timing_ns=images[-1]['timing_ns'],
    observed_image_rate_hz=(len(images)-1)*1e9/(images[-1]['timing_ns']-images[0]['timing_ns']),
    timing_delta_ns_min=min(time_deltas), timing_delta_ns_max=max(time_deltas),
    camera_sequence_first=images[0]['camera_sequence'],
    camera_sequence_last=images[-1]['camera_sequence'], camera_sequence_deltas=dict(sequence_deltas),
    sequence_interpretation='Every third camera acquisition, consistent with 1080 Hz run / 360 Hz readout',
    daq_run3_fp_csv=str(reference), daq_run3_fp_csv_sha256=file_sha256(reference),
    daq_run3_outside_known_fp_pixels=len(daq_fp), daq_run3_fp_reproduced=reproduced,
    after_first=review['after_first'], first_frame=review['frames'][0],
)
(out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
with (out/'timing.csv').open('w', newline='') as stream:
    keys = ['record_1based', 'payload_bytes', 'timing_ns', 'pulse_id', 'camera_sequence']
    writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore', lineterminator='\n')
    writer.writeheader()
    writer.writerows(records)
for name in ('review.json', 'after-first-counts.npz', 'after-first-mismatches.csv', 'all-counts.npz'):
    shutil.copyfile(review_dir/name, out/name)
for name in ('test.json', 'software.json', 'camera-prepared.yml', 'camera-armed.yml',
             'pcie-prepared.yml', 'pcie-armed.yml', 'console.log', 'ready.json',
             'profile.json', 'resolved-profile.json', 'source-profile.json'):
    (out/(name+'.gz')).write_bytes(gzip.compress((attempt/name).read_bytes(), mtime=0))
for name in ('execution-attempt17.json', 'restore-attempt17.json', 'restore-attempt17.log'):
    shutil.copyfile(folder/name, out/name)
(out/'archive-sha256.json').write_text(json.dumps({p.name: file_sha256(p)
    for p in sorted(out.iterdir()) if p.is_file() and p.name != 'archive-sha256.json'}, indent=2)+'\n')
print(json.dumps(result, indent=2))
