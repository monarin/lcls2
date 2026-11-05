"""Audit every JF header/shape using bounded block reads; hash all source blocks."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
import struct

import numpy as np

from common import ARTIFACTS, HERE, check_stat, wire


def inspect_frame(fd, row, expected_header, schemas, source_id, read_at=os.pread):
    base, size = int(row['bd_offset']), int(row['bd_size'])
    cache = {}

    def read(offset, length):
        wire.require(offset >= 0 and offset + length <= size, 'Read outside datagram')
        # Two small prefix windows normally cover both Jungfrau ShapesData trees.
        key = offset // 256 * 256
        if key not in cache:
            cache[key] = read_at(fd, min(256, size - key), base + key)
        block = cache[key]
        if offset + length <= key + len(block):
            return block[offset-key:offset-key+length]
        data = read_at(fd, length, base + offset)
        wire.require(len(data) == length, 'Short metadata read')
        return data

    header = wire.HEADER.unpack(read(0, 24))
    # SMD reconstructs L1Accept with Src(0); bigdata retains its node Src.
    # Compare timing/control, then validate the bigdata-specific fields.
    wire.require(header[:2] == expected_header[:2] and header[2] == source_id
                 and header[4] == 0 and header[-1] == size - 12, 'SMD/bigdata event header mismatch')
    nodes = []

    def walk(pos, end):
        src, damage, contains, extent = wire.XTC.unpack(read(pos, 12))
        wire.require(extent >= 12 and pos + extent == end, 'Malformed XTC extent')
        kind = contains & 4095
        node = {'offset': pos, 'src': src, 'damage': damage, 'kind': kind, 'size': extent, 'children': []}
        nodes.append(node)
        if kind in (0, 1):
            child = pos + 12
            while child < end:
                n = wire.XTC.unpack(read(child, 12))[-1]
                wire.require(n >= 12 and child + n <= end, 'Malformed child extent')
                node['children'].append(walk(child, child + n))
                child += n
        return node

    walk(12, size)
    present = []
    for node in nodes:
        if node['kind'] != 1:
            continue
        wire.require(node['src'] in schemas, 'Unexpected event NamesId')
        schema = schemas[node['src']]
        wire.require(schema['detname'] == 'jungfrau1M' and schema['alg'] == 'raw', 'Mixed JF stream changed')
        shape_nodes = [x for x in node['children'] if x['kind'] == 2]
        data_nodes = [x for x in node['children'] if x['kind'] == 3]
        wire.require(len(shape_nodes) == len(data_nodes) == 1, 'Missing/duplicate Shapes or Data')
        sh, data = shape_nodes[0], data_nodes[0]
        wire.require(sh['size'] == 12 + 20 * schema['arrays'], 'Wrong Shapes length')
        dims = struct.unpack('<' + 'I' * 5 * schema['arrays'], read(sh['offset'] + 12, sh['size'] - 12))
        nbytes = array_index = 0
        for name, dtype, rank in schema['fields']:
            shape = dims[array_index*5:array_index*5+rank] if rank else ()
            if rank:
                array_index += 1
            if name == 'raw':
                wire.require(dtype == 1 and shape == (1, 512, 1024), 'Wrong JF pixel shape/type')
            nbytes += math.prod(shape) * wire.SIZES[dtype]
        wire.require(nbytes == data['size'] - 12, 'Wrong Data length')
        present.append(schema['segment'])
    wire.require(len(set(present)) == len(present), 'Duplicate JF segment')
    damage = 0
    for node in nodes:
        damage |= node['damage']
    return sorted(present), damage


def main():
    manifest = json.loads((HERE / 'inventory.json').read_text())
    entry = manifest['streams'][3]
    check_stat(entry['bigdata'])
    index = np.load(ARTIFACTS / 'indexes/s003.npy', allow_pickle=False)
    events = index[index['service'] == 12]
    smd = (ARTIFACTS / 'smd/xpp101605526-r0311-s003-c000.smd.xtc2').read_bytes()
    wire.require(wire.sha(smd) == entry['smd']['sha256'], 'SMD snapshot changed')
    schemas = {int(k): v for k, v in entry['schemas'].items()}
    config = (ARTIFACTS / 'config/s003.bd.bin').read_bytes()
    wire.require(wire.sha(config) == entry['bd_config_sha256'], 'Configure snapshot changed')
    source_id = wire.header(config)[2]
    fd = os.open(entry['bigdata']['path'], os.O_RDONLY)
    block_bytes = 64 * 1024 * 1024
    file_size = entry['bigdata']['size']

    def chunk(start):
        damage_counts, presence_counts, issues = Counter(), Counter(), []
        blocks, hashes = {}, {}

        def blocked_read(fd, length, offset):
            answer = bytearray()
            while length:
                key = offset // block_bytes * block_bytes
                if key not in blocks:
                    block = os.pread(fd, min(block_bytes, file_size - key), key)
                    wire.require(len(block) == min(block_bytes, file_size - key), 'Short source block')
                    hashes[key] = wire.sha(block)
                    if len(blocks) == 2:
                        del blocks[next(iter(blocks))]
                    blocks[key] = block
                n = min(length, len(blocks[key]) - (offset-key))
                wire.require(n > 0, 'Read past source EOF')
                answer.extend(blocks[key][offset-key:offset-key+n])
                offset += n
                length -= n
            return bytes(answer)

        stop = min(start + 5000, len(events))
        for i in range(start, stop):
            row = events[i]
            h = wire.HEADER.unpack_from(smd, int(row['smd_offset']))
            present, damage = inspect_frame(fd, row, h, schemas, source_id, blocked_read)
            damage_counts[damage] += 1
            presence_counts[tuple(present)] += 1
            if damage or present != [0, 1]:
                issues.append({'event_index': i, 'timestamp': int(row['timestamp']),
                               'damage': damage, 'segments': present})
        print(f'Audited JF events {start}..{stop-1}', flush=True)
        if stop == len(events):
            blocked_read(fd, 1, file_size - 1)
        return damage_counts, presence_counts, issues, hashes

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(chunk, range(0, len(events), 5000)))
    finally:
        os.close(fd)
    damage, presence, issues, block_hashes = Counter(), Counter(), [], {}
    for d, p, bad, hashes in results:
        damage.update(d)
        presence.update(p)
        issues.extend(bad)
        for offset, digest in hashes.items():
            wire.require(offset not in block_hashes or block_hashes[offset] == digest, 'Source block changed during audit')
            block_hashes[offset] = digest
    wire.require(sorted(block_hashes) == list(range(0, file_size, block_bytes)), 'Source block coverage incomplete')
    check_stat(entry['bigdata'])
    wire.save_json(ARTIFACTS / 'jungfrau_issues.json', issues)
    wire.save_json(ARTIFACTS / 'jungfrau_block_hashes.json',
                   {'block_bytes': block_bytes, 'file_size': file_size, 'hashes': dict(sorted(block_hashes.items()))})
    report = {'status': 'audit_complete', 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'events': len(events), 'damage_histogram': dict(damage),
              'segment_presence': {str(k): v for k, v in presence.items()},
              'issue_events': len(issues), 'first_issues': issues[:10],
              'issues_sha256': wire.file_sha(ARTIFACTS / 'jungfrau_issues.json'),
              'schema_and_bounds_checks': 'pass for every source JF L1Accept',
              'pixel_data_audited': 'All source bytes read and block-hashed for efficient I/O; pixel values interpreted only in representative samples.json.',
              'source_block_hashes_sha256': wire.file_sha(ARTIFACTS / 'jungfrau_block_hashes.json'),
              'source_bytes_read_and_hashed': file_size,
              'script_sha256': wire.file_sha(HERE / 'audit_jungfrau.py')}
    wire.save_json(HERE / 'jungfrau_audit.json', report)
    print(f'Completed {len(events)} events; {len(issues)} damaged/missing-frame events', flush=True)


if __name__ == '__main__':
    main()
