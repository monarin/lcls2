"""Freeze SMD indexes, configurations and transition evidence; no source writes."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import struct

import numpy as np

from common import ARTIFACTS, HERE, SELECTED, SOURCE, SOURCE_EVENTS, check_stat, private_root, record_stat, wire

DTYPE = np.dtype([(k, '<u8') for k in ('timestamp', 'service', 'smd_offset', 'smd_size', 'bd_offset', 'bd_size')])


def freeze(path, data):
    if path.exists():
        wire.require(path.read_bytes() == data, 'Partial snapshot changed: ' + str(path))
    else:
        path.write_bytes(data)


def inspect_stream(stream):
    stem = f'xpp101605526-r0311-s{stream:03d}-c000'
    bd = SOURCE / (stem + '.xtc2')
    smd = SOURCE / 'smalldata' / (stem + '.smd.xtc2')
    before = [record_stat(bd), record_stat(smd)]
    blob = smd.read_bytes()
    freeze(ARTIFACTS / 'smd' / smd.name, blob)
    frames = list(wire.records(blob))
    smd_config = frames[0][1]
    sc = wire.schemas(smd_config)
    rows, transitions = [], []
    cursor = 0
    with bd.open('rb') as f:
        config = wire.read_record(f, 0)
        freeze(ARTIFACTS / 'config' / f's{stream:03d}.bd.bin', config)
        for smd_offset, raw in frames:
            h = wire.header(raw)
            svc = (h[1] >> 24) & 15
            if svc == 12:
                (_, off), (_, size) = wire.smd_offsets(raw, sc)
                wire.require(off == cursor and off + size <= before[0]['size'], 'Invalid SMD range')
            elif svc == 10:
                # SMD generator copies SlowUpdate. Infer its bigdata length;
                # the next event offset and final EOF independently bound it.
                # Representative byte checks follow below. Full retained-byte
                # comparison remains a Stage 2 writer/validator requirement.
                size = len(raw)
            else:
                actual = wire.read_record(f, cursor)
                bh = wire.header(actual)
                wire.require(bh[:5] == h[:5], 'SMD/bigdata transition header mismatch')
                size = len(actual)
                filename = f's{stream:03d}-record{len(rows):06d}-svc{svc}.bin'
                freeze(ARTIFACTS / 'transitions' / filename, actual)
                transitions.append({'service': svc, 'timestamp': h[0], 'size': size,
                                    'sha256': wire.sha(actual), 'file': 'transitions/' + filename})
            rows.append((h[0], svc, smd_offset, len(raw), cursor, size))
            cursor += size
    index = np.array(rows, dtype=DTYPE)
    wire.require(cursor == before[0]['size'], 'Unindexed bigdata tail')
    wire.require(np.all(index['timestamp'][1:] > index['timestamp'][:-1]), 'Unordered timeline')
    wire.require(int(np.count_nonzero(index['service'] == 12)) == SOURCE_EVENTS, 'Wrong event count')
    slow = index[index['service'] == 10]
    with bd.open('rb') as f:
        for row in (slow[0], slow[-1]):
            a, n = int(row['smd_offset']), int(row['smd_size'])
            wire.require(wire.read_record(f, int(row['bd_offset'])) == blob[a:a+n], 'Sampled SlowUpdate mismatch')
    for r in before:
        check_stat(r)
    np.save(ARTIFACTS / 'indexes' / f's{stream:03d}.npy', index, allow_pickle=False)
    result = {'stream': stream, 'selected': stream in SELECTED, 'bigdata': before[0],
              'smd': dict(before[1], sha256=wire.sha(blob)), 'records': len(rows),
              'service_counts': dict(Counter(map(int, index['service']))),
              'bd_config_bytes': len(config), 'smd_config_bytes': len(smd_config),
              'bd_config_sha256': wire.sha(config), 'schemas': wire.schemas(config),
              'smd_schemas': sc, 'transitions': transitions,
              'index_sha256': wire.file_sha(ARTIFACTS / 'indexes' / f's{stream:03d}.npy')}
    print(f'Stream {stream}: {len(rows)} records, {result["service_counts"]}', flush=True)
    return result, index


def state_check(index):
    state = 'start'
    expected = {2: ('start', 'configured'), 4: ('configured', 'run'),
                6: ('run', 'step'), 8: ('step', 'enabled'),
                9: ('enabled', 'step_done'), 7: ('step_done', 'run_done'),
                5: ('run_done', 'end')}
    for row in index:
        svc = int(row['service'])
        if svc in (10, 12):
            wire.require(state == 'enabled', 'Payload outside enabled interval')
        else:
            before, after = expected[svc]
            wire.require(state == before, f'Unexpected transition {svc} in {state}')
            state = after
    wire.require(state == 'end', 'Missing EndRun')


def main():
    private_root()
    for name in ('smd', 'config', 'transitions', 'indexes'):
        (ARTIFACTS / name).mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        inspected = list(pool.map(inspect_stream, range(7)))
    first = inspected[0][1]
    state_check(first)
    for _, index in inspected:
        wire.require(np.array_equal(index[['timestamp', 'service']], first[['timestamp', 'service']]),
                     'Stream timelines disagree')
    report = {'status': 'pass', 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'source': {'experiment': 'xpp101605526', 'run': 311, 'events': SOURCE_EVENTS},
              'selected_streams': list(SELECTED), 'snapshot': str(ARTIFACTS),
              'streams': [r for r, _ in inspected], 'all_stream_timelines_equal': True,
              'one_step_one_enable_state_machine': 'pass',
              'code_sha256': {n: wire.file_sha(HERE / n) for n in ('inventory.py', 'common.py', 'runtime.sh')},
              'scope': 'All SMD offsets and non-SlowUpdate transition payloads checked; first/last SlowUpdate bytes checked per stream, other lengths inferred and bounded by next offsets/EOF. JF L1 headers audited separately. No full bigdata hash.'}
    wire.save_json(HERE / 'inventory.json', report)
    print('PASS all seven aligned source timelines and one-step state machine', flush=True)


if __name__ == '__main__':
    main()
