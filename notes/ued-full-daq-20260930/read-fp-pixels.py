"""Targeted post-run ASIC readback through the existing DAQ Rogue servers.

Default is preflight only. --execute pauses/restores triggers and changes SACI
address/configuration controls, but never writes pixel data or loads a map.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np
from psdaq.control.DaqControl import DaqControl
from psdaq.debugtools.epixquad1kfps.write_gain_mode_standalone import (
    _raw_to_direct, _direct_to_raw, BANK_OFFSETS,
)


def targets(root):
    source = root/'run4-gainbits/after-first-mismatches.csv'
    mask = np.load(root.parent/'ued-commissioning-20260928/map-inputs/expand2.npy')
    rows = []
    for item in csv.DictReader(source.read_text().splitlines()):
        if item['kind'] != 'FP' or item['within_known_bad'] != '0':
            continue
        coord = tuple(int(item[k]) for k in ('segment', 'raw_row', 'raw_col'))
        asic, row, col = _raw_to_direct(*coord)
        assert _direct_to_raw(asic, row, col) == coord and mask[coord] == 0
        rows.append(dict(zip(('segment', 'raw_row', 'raw_col'), coord),
                         asic=asic, asic_row=row, asic_col=col,
                         saci_col=BANK_OFFSETS[col//48] | (col % 48), expected_code=12))
    assert len(rows) == 44 and len({(r['asic'], r['asic_row'], r['asic_col']) for r in rows}) == 44
    return rows, hashlib.sha256(source.read_bytes()).hexdigest()


def put(variable, value):
    variable.set(value)
    if variable.get(read=True) != value:
        raise RuntimeError('Control readback mismatch: ' + variable.path)


def read_asic(saci, rows, state):
    enabled = saci.enable.get()
    state['enable_before'] = enabled
    is_en = None
    try:
        saci.enable.set(True)
        is_en = bool(saci.IsEn.get(read=True))
        state['IsEn_before'] = is_en
        state['trbit_before'] = int(saci.trbit.get(read=True))
        # Address controls are WO: do not label cached addresses hardware readback.
        state['address_controls'] = 'WO; finish with CmdPrepForRead, no claim of address restoration'
        put(saci.IsEn, True)
        saci.CmdPrepForRead.set(0)
        saci.PrepareMultiConfig.set(0)
        for repeat in (1, 2):
            for row in rows:
                saci.RowCounter.set(row['asic_row'])
                saci.ColCounter.set(row['saci_col'])
                row['readback_%d' % repeat] = int(saci.WritePixelData.get(read=True))
                row['trbit'] = state['trbit_before']
    finally:
        errors = []
        for name, operation in (
            ('CmdPrepForRead', lambda: saci.CmdPrepForRead.set(0)),
            ('IsEn', lambda: put(saci.IsEn, is_en) if is_en is not None else None),
            ('trbit', lambda: state.update(trbit_after=int(saci.trbit.get(read=True)))),
            ('enable', lambda: saci.enable.set(enabled)),
        ):
            try:
                operation()
            except Exception as exc:
                errors.append(name + ': ' + repr(exc))
        if state.get('trbit_after') != state.get('trbit_before'):
            errors.append('trbit changed')
        state['cleanup_errors'] = errors
        if errors:
            raise RuntimeError('ASIC cleanup failed: ' + '; '.join(errors))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    rows, digest = targets(root)
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                  host=socket.gethostname(), execute=args.execute,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_csv_sha256=digest, targets=rows, asics={}, status='preflight')
    def save():
        (args.output/'readback.json').write_text(json.dumps(record, indent=2)+'\n')
    save()
    from pyrogue.interfaces import VirtualClient
    control = DaqControl(host='drp-ued-cmp002', platform=0, timeout=3000)
    start = time.monotonic()
    try:
        with ExitStack() as stack:
            camera_client = stack.enter_context(VirtualClient(addr='localhost', port=9107))
            pcie_client = stack.enter_context(VirtualClient(addr='localhost', port=9103))
            for client in (camera_client, pcie_client):
                if client.root is None:
                    raise RuntimeError('Existing DAQ Rogue server unavailable')
                client.setTimeout(10000, False)
            camera, pcie = camera_client.root, pcie_client.root
            record['daq_state'] = control.getState()
            if record['daq_state'] not in ('reset', 'unallocated', 'allocated', 'connected'):
                raise RuntimeError('Require DAQ below Configured and no active transitions')
            regs = camera.SystemRegs
            record['carrier_ids'] = [[int(regs.CarrierIdLow[i].get(read=True)),
                                      int(regs.CarrierIdHigh[i].get(read=True))] for i in range(4)]
            expected = json.loads((root.parent/'ued-commissioning-20260928/test-attempt04.json').read_text())['camera_identity']['carrier_ids']
            if record['carrier_ids'] != expected:
                raise RuntimeError('Detector identity does not match run-4 target list')
            manager = pcie.DevPcie.Hsio.TimingRx.TriggerEventManager
            controls = [('DAQ_MasterEnable', manager.TriggerEventBuffer[0].MasterEnable),
                        ('Run_MasterEnable', manager.TriggerEventBuffer[4].MasterEnable),
                        ('TrigEn', regs.TrigEn), ('AutoTrigEn', regs.AutoTrigEn),
                        ('RdoutEn', camera.RdoutCore.RdoutEn)]
            previous = {name: variable.get(read=True) for name, variable in controls}
            record['controls_before'] = previous
            save()
            if not args.execute:
                print(json.dumps(record, indent=2))
                return
            if control.getState() != record['daq_state']:
                raise RuntimeError('DAQ state changed during preflight')
            changed = []
            restore_errors = []
            asic_cleanup_ok = True
            try:
                for name, variable in controls:
                    changed.append((name, variable))
                    put(variable, False)
                time.sleep(0.1)
                for index in sorted({r['asic'] for r in rows}):
                    if control.getState() != record['daq_state']:
                        raise RuntimeError('DAQ state changed during readback')
                    state = record['asics'][str(index)] = {}
                    try:
                        read_asic(camera.Epix10kaSaci[index],
                                  [r for r in rows if r['asic'] == index], state)
                    finally:
                        if state.get('cleanup_errors'):
                            asic_cleanup_ok = False
                        save()
            finally:
                # Keep triggering disabled if ASIC cleanup cannot be confirmed.
                if asic_cleanup_ok:
                    for name, variable in reversed(changed):
                        try:
                            put(variable, previous[name])
                        except Exception as exc:
                            restore_errors.append(name + ': ' + repr(exc))
                else:
                    restore_errors.append('ASIC cleanup failed; trigger/readout gates left disabled')
                record['restore_errors'] = restore_errors
                record['controls_after'] = {name: variable.get(read=True) for name, variable in controls}
                save()
            if restore_errors:
                raise RuntimeError('Control restoration failed')
            for row in rows:
                row['repeat_agrees'] = row['readback_1'] == row['readback_2']
                row['matches_expected'] = row['repeat_agrees'] and row['readback_1'] == 12 and row['trbit'] == 1
            with (args.output/'pixels.csv').open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
            record['status'] = 'complete'
            record['readback_counts'] = dict(Counter(r['readback_1'] for r in rows))
            record['matches_expected'] = sum(r['matches_expected'] for r in rows)
            record['daq_state_after'] = control.getState()
    except Exception as exc:
        record['status'] = 'failed'
        record['error'] = repr(exc)
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - start
        save()
    print(json.dumps({k:v for k,v in record.items() if k != 'targets'}, indent=2))


if __name__ == '__main__':
    main()
