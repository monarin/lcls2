"""Validate distributed step/event delivery and optional calibrated samples."""
import argparse
import json
import os
from pathlib import Path
import traceback

from common import (CALIB_OPTIONS, DEST, SAMPLES, TIMES, fingerprint, require,
                    runtime, save, setup, validate_events)
setup()
from mpi4py import MPI
from psana import DataSource


def check(args):
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    require(size == args.world_size and size > 3, 'MPI launcher size mismatch')
    require(os.environ.get('PS_PARALLEL') == 'mpi', 'MPI mode must be explicit')
    reference = json.loads(args.reference.read_text())
    require(reference['status'] == 'pass' and reference['kind'] == 'pilot', 'Missing calibrated serial acceptance')
    expected_steps = {s['step']: s['timestamp'] for s in reference['steps']}
    result = {'world_size': size, 'eb_nodes': int(os.environ['PS_EB_NODES']),
              'smd_events': int(os.environ['PS_SMD_N_EVENTS']), 'batch_size': args.batch,
              'calibrated': args.calibrated, 'passes': {}}
    for mode in ('steps', 'events'):
        kwargs = dict(exp='prjdatpsana', run=1, dir=str(DEST), live=False,
                      batch_size=args.batch, use_calib_cache=False)
        if not (args.calibrated and mode == 'steps'):
            kwargs['skip_calib_load'] = 'all'
        ds = DataSource(**kwargs)
        local = {'rank': rank, 'role': ds.comms.node_type(), 'events': [], 'steps': []}
        for run in ds.runs():
            require(type(run).__name__ == 'RunParallel', 'MPI fell back to serial')
            require(run.expt == 'prjdatpsana' and run.runnum == 1, 'Wrong run identity')
            value, doc = run.Detector('step_value'), run.Detector('step_docstring')
            jf = run.Detector('jungfrau')
            groups = ((None, run.events()),) if mode == 'events' else ((s, s.events()) for s in run.steps())
            for step, events in groups:
                if step is not None:
                    n = int(value(step))
                    require(n in expected_steps and int(step.evt.timestamp) == expected_steps[n], 'Wrong distributed step timestamp')
                    require(json.loads(doc(step))['step'] == n, 'Wrong step document')
                    local['steps'].append(n)
                for evt in events:
                    timestamp, actual = int(evt.timestamp), int(value(evt))
                    require(json.loads(doc(evt))['step'] == actual, 'Wrong event step document')
                    if step is not None:
                        require(actual == n, 'Step iteration carries wrong scan state')
                    index = TIMES.index(timestamp)
                    require(fingerprint(jf.raw.raw(evt)) == reference['events'][index]['fields']['jungfrau/raw/raw'], 'MPI raw mismatch')
                    record = {'timestamp': timestamp, 'step': actual}
                    if args.calibrated and mode == 'steps' and index in SAMPLES:
                        require(fingerprint(jf.raw.calib(evt, **CALIB_OPTIONS)) == reference['events'][index]['calibrated']['calib'], 'MPI calibrated mismatch')
                        record['calibrated_sample'] = index
                    local['events'].append(record)
        gathered = comm.gather(local, root=0)
        if rank == 0:
            flat = [e for r in gathered for e in r['events']]
            counts = validate_events(flat)
            require(sum(r['role'] == 'eb' for r in gathered) == result['eb_nodes'], 'Wrong EB topology')
            require(sum(r['role'] == 'smd0' for r in gathered) == 1, 'Wrong SMD0 topology')
            for r in gathered:
                require(r['role'] == 'bd' or not r['events'], 'Helper rank received user events')
                require(r['steps'] == sorted(set(r['steps'])), 'Repeated or unordered local steps')
                require([e['timestamp'] for e in r['events']] == sorted(e['timestamp'] for e in r['events']), 'Unordered local events')
            if args.calibrated and mode == 'steps':
                require(sorted(e['calibrated_sample'] for e in flat if 'calibrated_sample' in e) == SAMPLES, 'Missing calibrated samples')
            result['passes'][mode] = {'step_counts': counts, 'ranks': gathered}
        comm.Barrier()
    if rank == 0:
        result.update(status='pass', runtime=runtime(), mpi_library=MPI.Get_library_version())
        save(args.report, result)
        print('PASS',args.report,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--calibrated', action='store_true')
    args = parser.parse_args()
    try:
        check(args)
    except Exception:
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
