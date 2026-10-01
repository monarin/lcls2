"""Offline replay tests built from three existing run-3 events; no live access."""
import importlib.util
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('hook_compare', Path(__file__).with_name('compare.py'))
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
SOURCE = Path(os.environ.get('UED_DIAG_TEST_XTC',
    '/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/xtc/uedc00107-r0003-s001-c000.xtc2'))


def input_packet(raw, pulse, timestamp):
    # Inverse camera wire ordering; timings are synthetic identities for replay.
    stripes = np.empty((176,8,384), dtype='<u2')
    for segment, up, down in ((2,0,2),(3,1,3),(0,4,6),(1,5,7)):
        stripes[:,up] = raw[segment,:176][::-1,::-1]
        stripes[:,down] = raw[segment,176:,::-1]
    camera = bytearray(1095232)
    camera[32:32+stripes.nbytes] = stripes.tobytes()
    packet = bytearray(b'\x21'+bytes(7))
    timing = struct.pack('<QQ', pulse | (12 << 56), timestamp)+bytes(16)
    for dest, data in ((0,timing),(2,camera),(3,bytes(48))):
        packet += data
        packet += struct.pack('<IBBBB', len(data),dest,2 if dest == 2 else 0,0,2)
    return packet


@pytest.fixture
def capture(tmp_path):
    if not SOURCE.exists():
        pytest.skip('Commissioning replay requires retained run-3 XTC2')
    config_bytes, events = None, []
    with SOURCE.open('rb') as stream:
        while len(events) < 3:
            prefix = stream.read(24)
            size, service, ts = review.header(prefix)
            data = prefix+stream.read(size-24)
            if service == 2:
                config_bytes = data
            if service == 12:
                events.append(data)
    config = review.dgram.Dgram(view=config_bytes)
    folder = tmp_path/'diagnostic'
    folder.mkdir()
    (folder/'configure.xtc2').write_bytes(config_bytes)
    samples = []
    for i, event in enumerate(events):
        raw, _ = review.image(event, config)
        ts = review.header(event)[2]
        packet = input_packet(raw, i+1, ts)
        (folder/f'input-{i}.bin').write_bytes(packet)
        (folder/f'prewrite-{i}.xtc2').write_bytes(event)
        samples.append(dict(slot=i,pulse_id=i+1,timestamp=ts,
                            input=f'input-{i}.bin',input_bytes=len(packet),
                            prewrite=f'prewrite-{i}.xtc2',prewrite_bytes=len(event),
                            submitted_to_file_writer=True))
    (folder/'manifest.json').write_text(json.dumps(dict(schema_version=1,status='complete',
                                                       requested_frames=3,samples=samples)))
    endrun = bytearray(events[-1][:24])
    struct.pack_into('<I',endrun,8,5 << 24)
    struct.pack_into('<I',endrun,20,12)
    disk = tmp_path/'replay.xtc2'
    disk.write_bytes(config_bytes+b''.join(events)+endrun)
    return folder, disk, config, events, bytes(endrun)


def test_equal_recorded_data_and_existing_fps(capture, tmp_path):
    folder, disk, _, _, _ = capture
    result = review.compare(folder,[disk],tmp_path/'review')
    assert result['complete_comparison'] and result['all_prewrite_disk_bytes_equal']
    assert result['assessed'] == 2
    assert result['word_differences'] == dict(input_vs_prewrite=0,prewrite_vs_disk=0)
    for stage in ('input','prewrite','disk'):
        assert result['after_first'][stage+'_FP']['outside_known_unique'] == 43
        assert result['after_first'][stage+'_FN']['outside_known_unique'] == 0


def test_input_vs_conversion_difference(capture, tmp_path):
    folder, disk, config, events, _ = capture
    raw, _ = review.image(events[1],config)
    raw[0,10,10] ^= 0x4000
    (folder/'input-1.bin').write_bytes(input_packet(raw,2,review.header(events[1])[2]))
    result = review.compare(folder,[disk],tmp_path/'review')
    assert result['word_differences'] == dict(input_vs_prewrite=1,prewrite_vs_disk=0)
    assert result['all_prewrite_disk_bytes_equal']


def test_disk_pixel_corruption_detected(capture, tmp_path):
    folder, disk, config, events, endrun = capture
    dg = review.dgram.Dgram(view=events[1],config=config)
    offset = events[1].find(dg.epixquad1kfps[0].raw.raw.tobytes())
    assert offset >= 0
    changed = bytearray(events[1])
    changed[offset+(10*384+10)*2+1] ^= 0x40
    events[1] = bytes(changed)
    disk.write_bytes((folder/'configure.xtc2').read_bytes()+b''.join(events)+endrun)
    result = review.compare(folder,[disk],tmp_path/'review')
    assert result['word_differences'] == dict(input_vs_prewrite=0,prewrite_vs_disk=1)
    assert not result['all_prewrite_disk_bytes_equal']


def test_missing_disk_event_is_not_clean(capture, tmp_path):
    folder, disk, _, events, endrun = capture
    disk.write_bytes((folder/'configure.xtc2').read_bytes()+b''.join(events[:2])+endrun)
    result = review.compare(folder,[disk],tmp_path/'review')
    assert not result['complete_comparison']
    assert result['frames'][2]['errors'] == ['No matching disk event']


def test_truncated_file_rejected(capture, tmp_path):
    folder, disk, _, _, _ = capture
    disk.write_bytes(disk.read_bytes()[:-1])
    with pytest.raises(ValueError,match='Short'):
        review.compare(folder,[disk],tmp_path/'review')


def test_wrong_incoming_identity_rejected(capture, tmp_path):
    folder, disk, _, _, _ = capture
    p = folder/'input-1.bin'
    data = bytearray(p.read_bytes())
    data[8] ^= 1
    p.write_bytes(data)
    result = review.compare(folder,[disk],tmp_path/'review')
    assert not result['complete_comparison']
    assert result['frames'][1]['errors'] == ['Incoming timing does not match DRP identity']


def test_not_submitted_is_not_clean(capture, tmp_path):
    folder, disk, _, _, _ = capture
    p = folder/'manifest.json'
    data = json.loads(p.read_text())
    data['samples'][1]['submitted_to_file_writer'] = False
    p.write_text(json.dumps(data))
    result = review.compare(folder,[disk],tmp_path/'review')
    assert not result['complete_comparison']
    assert result['frames'][1]['errors'] == ['Event was not selected for disk writing']


def test_without_endrun_is_incomplete(capture, tmp_path):
    folder, disk, _, _, _ = capture
    disk.write_bytes(disk.read_bytes()[:-24])
    result = review.compare(folder,[disk],tmp_path/'review')
    assert not result['complete_comparison'] and not result['endrun_seen']


def test_duplicate_timestamp_rejected(capture, tmp_path):
    folder, disk, _, events, endrun = capture
    disk.write_bytes((folder/'configure.xtc2').read_bytes()+b''.join(events)+events[1]+endrun)
    with pytest.raises(ValueError,match='duplicate'):
        review.compare(folder,[disk],tmp_path/'review')


def test_incomplete_hook_rejected(capture, tmp_path):
    folder, disk, _, _, _ = capture
    p = folder/'manifest.json'
    data = json.loads(p.read_text())
    data['status'] = 'incomplete'
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='incomplete'):
        review.compare(folder,[disk],tmp_path/'review')
