"""Compute exact copied-record schedules and sizes, without generating XTC."""
from collections import Counter
import json
import struct

import numpy as np

from common import ARTIFACTS, HERE, SELECTED, SOURCE_EVENTS, wire
from inventory import state_check


def schedule(index, event_count):
    svc = index['service']
    opening = [int(np.flatnonzero(svc == n)[0]) for n in (2, 4, 6, 8)]
    closing = [int(np.flatnonzero(svc == n)[0]) for n in (9, 7, 5)]
    body = np.arange(opening[-1] + 1, closing[0])
    wire.require(np.all(np.isin(svc[body], [10, 12])), 'Unexpected body transition')
    nsource = int(np.count_nonzero(svc[body] == 12))
    wire.require(nsource == SOURCE_EVENTS and event_count > 0, 'Wrong source/target event count')
    period = wire.to_ns(int(index['timestamp'][closing[0]])) - wire.to_ns(int(index['timestamp'][opening[-1]]))
    records, timestamps, cycles = list(opening), [int(index['timestamp'][i]) for i in opening], [-1]*4
    remaining, cycle = event_count, 0
    while remaining:
        take = min(remaining, nsource)
        last = int(np.flatnonzero(svc[body] == 12)[take-1])
        # Full cycles retain trailing SlowUpdates; a prefix ends at its last event.
        selected = body if take == nsource else body[:last+1]
        records.extend(map(int, selected))
        timestamps.extend(wire.from_ns(wire.to_ns(int(index['timestamp'][i])) + cycle*period) for i in selected)
        cycles.extend([cycle]*len(selected))
        remaining -= take
        cycle += 1
    last_ns = wire.to_ns(timestamps[-1])
    records.extend(closing)
    timestamps.extend(wire.from_ns(last_ns + j*1000) for j in (1, 2, 3))
    cycles.extend([-1]*3)
    records, timestamps, cycles = np.asarray(records, dtype='<u4'), np.asarray(timestamps, dtype='<u8'), np.asarray(cycles, dtype='<i4')
    wire.require(np.all(timestamps[1:] > timestamps[:-1]), 'Output timeline is not strictly ordered')
    timeline = np.empty(len(records), dtype=[('timestamp', '<u8'), ('service', '<u8')])
    timeline['timestamp'], timeline['service'] = timestamps, svc[records]
    state_check(timeline)
    wire.require(np.count_nonzero(timeline['service'] == 12) == event_count, 'Wrong output count')
    return records, timestamps, cycles, period


def main():
    inventory = json.loads((HERE / 'inventory.json').read_text())
    indexes = {s: np.load(ARTIFACTS / 'indexes' / f's{s:03d}.npy', allow_pickle=False) for s in SELECTED}
    for s, index in indexes.items():
        wire.require(wire.file_sha(ARTIFACTS/'indexes'/f's{s:03d}.npy') == inventory['streams'][s]['index_sha256'], 'Index changed')
    identity = []
    for s in SELECTED:
        entry = inventory['streams'][s]
        schemas = {int(k): v for k, v in entry['schemas'].items()}
        for transition in entry['transitions']:
            raw = (ARTIFACTS / transition['file']).read_bytes()
            wire.require(wire.sha(raw) == transition['sha256'], 'Transition template changed')
            if transition['service'] == 4:
                key, schema = wire.schema_for(schemas, 'runinfo')
                fields = wire.spans(raw, key, schema)
                p, n, _, _ = fields['expt']
                q, m, dtype, rank = fields['runnum']
                wire.require(wire.cstring(raw[p:p+n]) == 'xpp101605526' and len(b'prjdatpsana\0') <= n, 'Runinfo experiment does not fit')
                wire.require((m, dtype, rank) == (4, 2, 0) and struct.unpack_from('<I', raw, q)[0] == 311, 'Runinfo number mismatch')
                identity.append({'stream': s, 'experiment_field_bytes': n, 'run_number_bytes': m})
            if transition['service'] in (6, 7, 8, 9, 5):
                wire.require(len(raw) == 24, 'Nonempty transition needs extra rewrite/size specification')
    plans = []
    for run, count in ((3, 2400), (4, 400000)):
        rec, ts, cyc, period = schedule(indexes[SELECTED[0]], count)
        mapping = ARTIFACTS / f'run{run}_mapping.npz'
        np.savez(mapping, source_record=rec, timestamp=ts, cycle=cyc)
        services = indexes[SELECTED[0]]['service'][rec]
        output_event_rows = np.flatnonzero(services == 12)
        boundaries = []
        for n in (0, 2399, 2400, 151151, 151152, 302303, 302304, count-1):
            if 0 <= n < count:
                i = output_event_rows[n]
                boundaries.append({'output_event_index': n, 'source_event_index': n % SOURCE_EVENTS,
                                   'source_cycle': int(cyc[i]), 'timestamp': int(ts[i]),
                                   'source_record': int(rec[i])})
        files = []
        for s, index in indexes.items():
            for kind in ('bd', 'smd'):
                size = int(np.sum(index[kind + '_size'][rec]))
                name = f'prjdatpsana-r{run:04d}-s{s:03d}-c000' + ('.xtc2' if kind == 'bd' else '.smd.xtc2')
                files.append({'stream': s, 'kind': kind, 'relative_path': name if kind == 'bd' else 'smalldata/' + name,
                              'bytes': size})
        gaps = [wire.to_ns(int(b))-wire.to_ns(int(a)) for a, b in zip(ts, ts[1:])]
        plans.append({'run': run, 'events': count, 'records_per_stream': len(rec), 'step_count': 1,
                      'service_counts': dict(Counter(map(int, services))), 'files': files,
                      'total_bytes': sum(f['bytes'] for f in files),
                      'mapping_sha256': wire.file_sha(mapping), 'mapping_path': str(mapping),
                      'source_cycle_offset_ns': period, 'min_adjacent_gap_ns': min(gaps),
                      'boundary_events': boundaries, 'closing_timestamps': list(map(int, ts[-3:]))})
    report = {'status': 'pass', 'selected_source_streams': list(SELECTED),
              'excluded_source_streams': {'0': 'alvium_1 image', '1': 'epix100 image', '4': 'alvium_tt camera data'},
              'source_stream_ids_preserved': True, 'chunk_count_per_stream': 1,
              'runinfo': identity, 'native_step_payload': 'empty 24-byte BeginStep/EndStep; no scan Names added',
              'enable_chunkinfo_payload': 'absent; no payload string to rename',
              'size_contract': 'Whole selected streams, byte-length-preserving identity/time/offset edits only; no additional bin payload or Names. Stage 2 must match exactly or revise estimate.',
              'plans': plans, 'script_sha256': wire.file_sha(HERE / 'plan_layout.py')}
    wire.save_json(HERE / 'layout.json', report)
    print(json.dumps([{'run':p['run'],'events':p['events'],'bytes':p['total_bytes'],'records':p['records_per_stream']} for p in plans]), flush=True)


if __name__ == '__main__':
    main()
