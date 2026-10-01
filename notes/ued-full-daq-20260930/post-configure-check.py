"""Explicit staged control for the authorized post-Configure 44-pixel test."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from psdaq.control.DaqControl import DaqControl


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('status', 'configure', 'run', 'run-unmodified', 'unallocate', 'rediscover'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--repair-record', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    control = DaqControl(host='drp-ued-cmp002', platform=0, timeout=5000)
    record = dict(action=args.action, started_utc=datetime.now(timezone.utc).isoformat(), history=[])
    def save():
        args.output.write_text(json.dumps(record, indent=2)+'\n')
    def status(strict=True):
        result = dict(zip(('transition','state','alias','recording','platform','bypass',
                           'experiment','run','last_run'), control.getStatus()))
        record['history'].append(dict(time_utc=datetime.now(timezone.utc).isoformat(), **result))
        save()
        if result['state'] == 'error':
            # The front REP socket can remain busy throughout Configure.
            # Reset a timed-out REQ socket before the next status poll.
            control.front_req_init()
            if not strict:
                return result
        if result['state'] == 'error' or result['alias'] != 'TEST' or result['experiment'] != 'uedc00107':
            raise RuntimeError('Unexpected DAQ status/alias/experiment')
        return result
    def transition(target, timeout=80):
        error = control.setState(target)
        if error:
            raise RuntimeError(error)
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            current = status(strict=False)
            if current['state'] == target:
                return current
            time.sleep(0.5)
        raise RuntimeError('Timeout waiting for '+target)
    try:
        before = status()
        if args.action == 'rediscover':
            if before['state'] != 'unallocated':
                raise RuntimeError('Rediscovery requires unallocated DAQ')
            transition('reset')
            transition('unallocated')
        elif args.action == 'unallocate':
            if before['state'] not in ('configured','connected','allocated'):
                raise RuntimeError('Expected a stopped DAQ above unallocated')
            transition('unallocated')
        elif args.action == 'configure':
            if before['state'] not in ('unallocated','allocated','connected'):
                raise RuntimeError('Expected DAQ below Configured')
            transition('configured')
        elif args.action in ('run', 'run-unmodified'):
            if before['state'] != 'configured':
                raise RuntimeError('Require Configured state')
            if args.action == 'run':
                if args.repair_record is None:
                    raise RuntimeError('Require repair evidence')
                repair = json.loads(args.repair_record.read_text())
                if not (repair['status'] == 'complete' and repair['matches_expected'] == 44
                        and repair['daq_state'] == repair['daq_state_after'] == 'configured'
                        and repair['overwrite_fh'] and not repair['restore_errors']):
                    raise RuntimeError('Repair did not complete in Configured state')
                record['repair_record'] = str(args.repair_record)
            else:
                record['post_configure_pixel_repair'] = False
            if not before['recording']:
                error = control.setRecord(True)
                if error:
                    raise RuntimeError(error)
            if not status()['recording']:
                raise RuntimeError('Recording is not enabled')
            try:
                running = transition('running', timeout=30)
                record['run_number'] = running['run']
                save()
                time.sleep(3)
            finally:
                # EndRun normally, remaining Configured so there is no map rewrite.
                transition('configured', timeout=30)
        record['status'] = 'complete'
    except Exception as exc:
        record['status'] = 'failed'
        record['error'] = repr(exc)
        raise
    finally:
        save()
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
