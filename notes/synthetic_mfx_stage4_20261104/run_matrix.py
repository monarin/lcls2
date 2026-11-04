"""Bounded, sequential local CPU MPI acceptance; fail closed on any timeout."""
import os
from pathlib import Path
import subprocess

from common import HERE, require, save

MPIRUN = '/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/xpp_drp_gpu_311/bin/mpirun'


def main():
    folder = HERE / 'mpi'
    folder.mkdir(exist_ok=False)
    cases = [(4, 1, smd, batch, False) for smd, batch in [(19, 7), (20, 20), (21, 19), (61, 21)]]
    cases += [(7, 2, smd, batch, False) for smd, batch in [(19, 7), (20, 20), (21, 19), (61, 21)]]
    cases.append((4, 1, 20, 20, True))
    results = []
    for ranks, eb, smd, batch, calibrated in cases:
        label = f'n{ranks}-eb{eb}-smd{smd}-batch{batch}-calib{int(calibrated)}'
        report = folder / (label+'.json')
        command = [MPIRUN, '--oversubscribe', '--bind-to', 'none', '--mca', 'btl', 'self,vader', '-n', str(ranks),
                   'bash', str(HERE/'runtime.sh'), str(HERE/'mpi_check.py'),
                   '--reference', str(HERE/'pilot.json'), '--report', str(report),
                   '--batch', str(batch), '--world-size', str(ranks)]
        if calibrated:
            command.append('--calibrated')
        env = dict(os.environ, PS_PARALLEL='mpi', PS_EB_NODES=str(eb), PS_SMD_N_EVENTS=str(smd))
        print('Running',label,flush=True)
        with (folder/(label+'.log')).open('x') as log:
            proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        require(proc.returncode == 0 and report.exists(), 'MPI case failed: '+label)
        results.append({'case': label, 'command': command, 'report': str(report), 'returncode': proc.returncode})
    save(folder/'summary.json', {'status': 'pass', 'cases': results})


if __name__ == '__main__':
    main()
