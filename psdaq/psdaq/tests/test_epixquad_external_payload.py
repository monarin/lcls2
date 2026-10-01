"""External batcher framing, independent of hardware and Rogue."""
import struct

import numpy as np
import pytest

from psdaq.configdb.epixquad_layout import (
    DAQ_RAW_FRAME_WORDS, RAW_SHAPE, ROGUE_EXTERNAL_FRAME_BYTES,
    event_batcher_subframes, rogue_image_skip_words, rogue_payload_to_daq_raw,
)
from psdaq.debugtools.epixquad1kfps.capture_standalone import ImageCounts


def batcher(parts):
    result = bytearray(b'\x21' + bytes(7))
    for dest, data in parts:
        result += data
        result += bytes((-len(data)) % 8)
        result += struct.pack('<IBBBB', len(data), dest, 2 if dest == 2 else 0, 0, 2)
    return result


def camera():
    result = bytearray(1095232)
    words = np.arange(DAQ_RAW_FRAME_WORDS, dtype='<u2')
    result[32:32+words.nbytes] = words.tobytes()
    return result, words


@pytest.mark.parametrize('order,skip', [((0, 2, 3), 40), ((2, 0, 3), 20)])
def test_external_data_location_and_panel_orientation(order, skip):
    image, words = camera()
    parts = {0: bytes(32), 2: image, 3: bytes(48)}
    data = batcher([(dest, parts[dest]) for dest in order])
    assert len(data) == ROGUE_EXTERNAL_FRAME_BYTES
    raw, offset = rogue_payload_to_daq_raw(data)
    assert offset == skip
    expected = np.empty(RAW_SHAPE, dtype=np.uint16)
    stripes = words.reshape(176, 8, 384)
    # Vectorized mapping independent of the decoder's per-row cursor.
    for segment, up, down in ((2, 0, 2), (3, 1, 3), (0, 4, 6), (1, 5, 7)):
        expected[segment, :176] = stripes[::-1, up, ::-1]
        expected[segment, 176:] = stripes[:, down, ::-1]
    np.testing.assert_array_equal(raw, expected)
    with pytest.raises(ValueError, match='tails'):
        rogue_image_skip_words(len(data))
    counts = ImageCounts([len(data)], 10_000_000)
    counts.observed(data, 0)
    assert counts.snapshot()['valid_images'] == 1
    counts.observed(data, 1)
    assert counts.snapshot()['invalid_records'] == 1


def test_short_timing_records_are_not_images():
    data = batcher([(0, bytes(32)), (3, bytes(48))])
    assert len(data) == 104
    assert set(event_batcher_subframes(data)) == {0, 3}
    counts = ImageCounts([ROGUE_EXTERNAL_FRAME_BYTES], 10_000_000)
    counts.observed(data, 0)
    assert counts.snapshot()['valid_images'] == 0


@pytest.mark.parametrize('corruption', ['version', 'zero', 'overrun', 'width',
                                       'duplicate', 'missing_camera', 'short_camera', 'eofe'])
def test_external_corruption_rejected(corruption):
    image, _ = camera()
    data = batcher([(0, bytes(32)), (2, image), (3, bytes(48))])
    tail = len(data) - 8
    camera_tail = tail - 56
    if corruption == 'version':
        data[0] = 0x22
    elif corruption == 'zero':
        struct.pack_into('<I', data, tail, 0)
    elif corruption == 'overrun':
        struct.pack_into('<I', data, tail, len(data))
    elif corruption == 'width':
        data[tail+7] = 255
    elif corruption == 'duplicate':
        data[tail+4] = 2
    elif corruption == 'missing_camera':
        data[camera_tail+4] = 1
    elif corruption == 'short_camera':
        struct.pack_into('<I', data, camera_tail, len(image)-1)
    elif corruption == 'eofe':
        data[camera_tail+6] = 1
    with pytest.raises(ValueError):
        rogue_payload_to_daq_raw(data)
    counts = ImageCounts([len(data)], 10_000_000)
    counts.observed(data, 0)
    assert counts.snapshot()['valid_images'] == 0


def test_truncated_batcher_rejected():
    data = batcher([(0, bytes(32)), (3, bytes(48))])
    with pytest.raises(ValueError):
        event_batcher_subframes(data[:-1])
