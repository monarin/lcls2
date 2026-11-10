"""Six matched 10k runs; same accepted runtime and owner-scoped cache gate."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'sim10_stream_length_20261109'
sys.path.insert(0, str(BASE))
from support import fixture, mount, require, sha, ts_hash, write, ROOT, EVENT_BYTES


def file_stats():
    result = []
    for row in json.loads((BASE / 'native/input-stat-after.json').read_text())['files']:
        p = Path(row['path'])
        s = p.stat()
        current = dict(path=str(p), bytes=s.st_size, uid=s.st_uid,
                       inode=s.st_ino, mtime_ns=s.st_mtime_ns)
        require(current == row, 'Accepted input metadata changed: ' + str(p))
        result.append(current)
    return result


def validate(result, reference, streams):
    require(result['status'] == 'pass' and result['events'] == 10000, 'Case status/count')
    require(result['bytes'] == 10000 * EVENT_BYTES, 'Case bytes')
    require(result['framework'] == dict(batch_size=100, streams=streams,
            budget_gib=6, bulk_target_bytes=524288, smd_events=1000), 'Controls')
    bd = result['rows'][2]
    require(bd['timestamps'] == reference[:10000], 'Event identity/order')
    require(bd['io']['completed_requests'] == 50000 and
            bd['io']['completed_bytes'] == 33582000000, 'Read accounting')
    require(len(bd['callbacks']) == 100 and all(c['size'] == 100 for c in bd['callbacks']), 'Callbacks')
    require(len({c['stream'] for c in bd['callbacks']}) == streams, 'Stream count')
    require(all(x['resident_pages'] == 0 for x in result['measurement']['initial_cache']), 'Cold gate')
    require(bd['cpu_reads'] == bd['io']['pressure_drains'] == 0, 'CPU reads/pressure')
    require(all(r['cleanup_complete'] for r in result['rows']), 'Cleanup')
    require(all(d['closed'] and d['pending'] == d['active'] == d['held'] == 0
                for d in bd['drain']), 'Retirement')


def main(out):
    out.mkdir(exist_ok=False, parents=True)
    original = file_stats()
    _, streams, stamps = fixture()
    allocated = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    gpu = subprocess.check_output(['nvidia-smi', '-i', allocated[0],
        '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    require(gpu.startswith('GPU-') and '\n' not in gpu, 'Physical GPU')
    os.environ['JF_GPU_UUID'] = gpu
    info = dict(exp='mfx101555026', run=10, directory=str(ROOT),
                files=[s['files']['bd']['path'] for s in streams], timestamps=stamps,
                event_bytes=EVENT_BYTES, generation_sha256=sha(ROOT/'generation.json'),
                timestamp_sha256=ts_hash(stamps))
    write(out/'inputs.json', info)
    files = list(HERE.glob('*.py')) + list(HERE.glob('*.sbatch'))
    files += [HERE.parents[1]/'psana/psana/gpu/scripts/bulk_phase_timing.py']
    write(out/'provenance.json', dict(job=os.environ['SLURM_JOB_ID'], started=time.time(),
        gpu=gpu, input_metadata=original, mount=mount(),
        prior_validation=str(BASE/'native/input-validation.json'),
        prior_validation_sha256=sha(BASE/'native/input-validation.json'),
        source_sha256={str(p):sha(p) for p in files},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()))
    cases = []
    for index, (mode, nstreams) in enumerate([
            ('clean',1), ('clean',2), ('timing',2), ('timing',1),
            ('nsys',1), ('nsys',2)]):
        name = '%02d-%s-s%d' % (index, mode, nstreams)
        program = BASE/'runner.py' if mode == 'clean' else HERE/'trace_runner.py'
        args = [str(program), '--inputs', str(out/'inputs.json'), '--output',
                str(out/(name+'.json')), '--events','10000','--streams',str(nstreams)]
        if mode == 'nsys':
            args = [str(HERE/'trace_rank.py'), str(out/name), *args]
        cmd = ['srun','--mpi=cray_shasta','--nodes=1','--ntasks=3',
               '--ntasks-per-node=3','--cpus-per-task=16','--gpus-per-node=1',
               '--gpu-bind=none','--cpu-bind=none','--kill-on-bad-exit=1',
               'numactl','--interleave=all',sys.executable,*args]
        print('Starting', name, flush=True)
        write(out/(name+'.command.json'), cmd)
        with (out/(name+'.log')).open('x') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        result = json.loads((out/(name+'.json')).read_text())
        validate(result, stamps, nstreams)
        require(result['job'] == os.environ['SLURM_JOB_ID'], 'Job identity')
        require(result['rows'][2]['device']['mask'] == gpu, 'GPU identity')
        if cases:
            require(result['host'] == cases[0]['host'], 'Host changed')
            require(result['rows'][2]['affinity'] == cases[0]['affinity'], 'Affinity changed')
        if mode != 'clean':
            phases = json.loads((out/(name+'.json.phases.json')).read_text())
            require(phases['counters']['reader']['total_requests'] == 50000, 'Timing read count')
            require(phases['counters']['reader']['total_bytes'] == result['bytes'], 'Timing bytes')
            require(all(p['self_ns'] >= 0 for p in phases['phases'].values()), 'Exclusive phases')
        if mode == 'nsys':
            require((out/(name+'.nsys-rep')).stat().st_size > 0, 'Missing viewable capture')
        cases.append(dict(name=name, mode=mode, streams=nstreams, seconds=result['seconds'],
            gib_s=result['gib_s'], host=result['host'], affinity=result['rows'][2]['affinity']))
        write(out/'progress.json',cases)
        print(json.dumps(cases[-1]),flush=True)
    require(file_stats() == original, 'Input changed during trace')
    write(out/'complete.json',dict(status='pass',cases=cases,finished=time.time(),input_unchanged=True))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
