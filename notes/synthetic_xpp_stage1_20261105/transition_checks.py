"""Compare frozen SMD transitions with their independently read bigdata copies."""
from datetime import datetime, timezone
import json
from pathlib import Path

from common import ARTIFACTS, HERE, wire


def main():
    inventory = json.loads((HERE / 'inventory.json').read_text())
    checked = []
    for entry in inventory['streams']:
        path = ARTIFACTS / 'smd' / Path(entry['smd']['path']).name
        smd = path.read_bytes()
        wire.require(wire.sha(smd) == entry['smd']['sha256'], 'Frozen SMD changed')
        records = {((wire.header(raw)[1] >> 24) & 15): raw
                   for _, raw in wire.records(smd)
                   if ((wire.header(raw)[1] >> 24) & 15) in (4, 5, 6, 7, 8, 9)}
        wire.require(set(records) == {4, 5, 6, 7, 8, 9}, 'Missing transition')
        for template in entry['transitions']:
            service = template['service']
            if service not in records:
                continue
            raw = (ARTIFACTS / template['file']).read_bytes()
            wire.require(wire.sha(raw) == template['sha256'], 'Frozen template changed')
            if service == 5:
                wire.require(len(records[service]) == 24, 'SMD EndRun is not header-only')
                wire.require(wire.header(records[service])[:5] == wire.header(raw)[:5],
                             'EndRun header mismatch')
            else:
                wire.require(records[service] == raw, 'SMD/bigdata transition bytes differ')
            checked.append({'stream': entry['stream'], 'service': service,
                            'smd_bytes': len(records[service]), 'bigdata_bytes': len(raw),
                            'check': 'header-only EndRun' if service == 5 else 'byte-identical'})
    wire.require(len(checked) == 42, 'Wrong transition check count')
    wire.save_json(HERE / 'transition_checks.json', {
        'status': 'pass', 'checked_utc': datetime.now(timezone.utc).isoformat(),
        'checks': checked, 'script_sha256': wire.file_sha(HERE / 'transition_checks.py')})
    print('PASS 35 SMD/bigdata transition byte comparisons and seven header-only SMD EndRuns')


if __name__ == '__main__':
    main()
