"""Offline archival comparison of SYNC False/True/False captures 14, 15, 16."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct

import yaml

HERE = Path(__file__).resolve().parent
STAND = HERE.parent.parent / 'ued-commissioning-20260928'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flat(obj, prefix=''):
    result = {}
    for key, value in obj.items():
        path = prefix + str(key)
        if isinstance(value, dict):
            result.update(flat(value, path + '.'))
        else:
            result[path] = value
    return result


def fp_coordinates(path):
    with path.open() as stream:
        return {tuple(int(row[k]) for k in ('segment', 'raw_row', 'raw_col'))
                for row in csv.DictReader(stream)
                if row['kind'] == 'FP' and not int(row['within_known_bad'])}


def main():
    daq_csv = HERE.parent / 'run3-gainbits/after-first-mismatches.csv'
    daq_fp = fp_coordinates(daq_csv)
    assert len(daq_fp) == 43
    results, configs = [], {}
    for attempt, requested in ((14, False), (15, True), (16, False)):
        source = STAND / ('attempt%d-mapfhl' % attempt)
        record = json.loads((source / 'test.json').read_text())
        review_dir = source / 'review-after-first'
        review = json.loads((review_dir / 'review.json').read_text())
        assert record['status'] == 'completed'
        assert record['asic_sync_inj_en'] == dict(requested=requested, before_map=requested, after_map=requested)
        assert digest(source / 'capture.dat') == record['data']['sha256'] == review['data_sha256']
        label = 'attempt%d-' % attempt
        for snapshot in record['config_snapshots']:
            assert digest(source / snapshot['path']) == snapshot['sha256']
            shutil.copyfile(source / snapshot['path'], HERE / (label + snapshot['path']))
        for name in ('software.json', 'console.log'):
            shutil.copyfile(source / name, HERE / (label + name))
        for name in ('review.json', 'after-first-mismatches.csv', 'after-first-counts.npz'):
            shutil.copyfile(review_dir / name, HERE / (label + name))
        for name in ('execution-attempt%d.json', 'restore-attempt%d.json', 'restore-attempt%d.log'):
            name = name % attempt
            shutil.copyfile(STAND / name, HERE / name)
        (HERE / (label + 'test.json.gz')).write_bytes(gzip.compress((source / 'test.json').read_bytes(), mtime=0))
        execution = json.loads((HERE / ('execution-attempt%d.json' % attempt)).read_text())
        restore = json.loads((HERE / ('restore-attempt%d.json' % attempt)).read_text())
        assert execution['capture_exit'] == execution['restore_exit'] == 0
        assert restore['status'] == 'restored' and restore['readback']['RdoutCore.RdoutEn'] is False
        fp = fp_coordinates(review_dir / 'after-first-mismatches.csv')
        sequences, histogram = [], Counter()
        with (source / 'capture.dat').open('rb') as stream:
            while True:
                header = stream.read(8)
                if not header:
                    break
                assert len(header) == 8
                length, flags, error, channel = struct.unpack('<IHBB', header)
                assert length >= 4
                payload = stream.read(length - 4)
                assert len(payload) == length - 4
                histogram[(len(payload), flags, error, channel)] += 1
                if channel == 1 and not error and len(payload) in (1095248, 1095288):
                    image_start = 40 + len(payload) - 1095248
                    sequences.append(struct.unpack_from('<I', payload, image_start - 24)[0])
        assert len(sequences) == review['recorded_frames']
        configs[attempt] = {}
        for device in ('camera', 'pcie'):
            configs[attempt][device] = flat(next(iter(yaml.safe_load(
                (source / (device + '-armed.yml')).read_text()).values())))
        assert configs[attempt]['camera']['AcqCore.AsicSyncInjEn'] is requested
        results.append(dict(attempt=attempt, sync_inj_en=record['asic_sync_inj_en'],
            started_utc=record['started_utc'], finished_utc=record['finished_utc'],
            data=record['data'], selected_map=record['inputs']['selected_map'],
            recorded_frames=review['recorded_frames'], assessed_frames=review['assessed_frames'],
            first_frame=review['frames'][0], after_first=review['after_first'],
            daq43_overlap=sorted(fp & daq_fp), daq43_missing=sorted(daq_fp - fp),
            extra_fp_not_in_daq43=sorted(fp - daq_fp),
            readout_clock=record['readout_clock'], trigger_readback=record['trigger_readback'],
            stream_counts=record['stream_counts'], cleanup_errors=record['cleanup_errors'],
            sequence_first=sequences[0], sequence_last=sequences[-1],
            sequence_discontinuities=[dict(image_1based=i+2, before=a, after=b)
                for i, (a, b) in enumerate(zip(sequences, sequences[1:])) if (b-a) % 2**32 != 1],
            payload_records=[dict(payload_bytes=size, flags=flags, error=error, channel=channel, count=count)
                for (size, flags, error, channel), count in sorted(histogram.items())],
            diagnostics_before=record['diagnostics_before'], diagnostics_after=record['diagnostics_after'],
            restore=restore))
    comparisons = {}
    assert len({r['selected_map']['sha256'] for r in results}) == 1
    for result in results:
        assert result['readout_clock'] == dict(full_period_ticks=3, half_period_ticks=1, adc_pipeline_delay=68)
        assert result['trigger_readback']['source'] == 3
        assert result['trigger_readback']['auto_period_ticks'] == 434028
    for other in (15, 16):
        comparisons['14-vs-%d' % other] = {}
        for device in ('camera', 'pcie'):
            a, b = configs[14][device], configs[other][device]
            assert a.keys() == b.keys()
            comparisons['14-vs-%d' % other][device] = [dict(path=key, attempt14=a[key], other=b[key])
                for key in sorted(a) if a[key] != b[key]]
    output = dict(results=results, config_differences=comparisons,
        daq_reference=str(daq_csv), daq_reference_sha256=digest(daq_csv),
        archive_script_sha256=digest(Path(__file__)),
        limitations=['First image retained but excluded from gain assessment',
                     'SaveConfig does not read pixel SRAM; disabled-device values can be cached',
                     'Independent configurations and map writes per attempt; external DAQ timing is not used'])
    (HERE / 'results.json').write_text(json.dumps(output, indent=2) + '\n')
    for result in results:
        print(result['attempt'], result['sync_inj_en'], result['recorded_frames'],
              result['after_first'], 'DAQ43 overlap', len(result['daq43_overlap']))
    print('config differences', json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
