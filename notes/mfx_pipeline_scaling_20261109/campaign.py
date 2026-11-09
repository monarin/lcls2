"""Bounded same-node comparison; each case is a fresh MPI job."""
import json
import os
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
output=Path(sys.argv[1]);output.mkdir(mode=0o770,exist_ok=False)
# Set before MPI/CUDA imports, so peer discovery sees exactly one physical GPU.
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id,name','--format=csv,noheader'],text=True)
(output/'gpu-inventory.csv').write_text(inventory)
os.environ['CUDA_VISIBLE_DEVICES']=inventory.splitlines()[0].split(',')[0].strip()
cases=[]
def execute(name,bds,streams,cache,task='full',events=10000,diagnostic=False):
    cmd=['mpiexec','-n',str(bds+2),'--bind-to','none',sys.executable,str(HERE/'runner.py'),
        '--output',str(output/(name+'.json')),'--bds',str(bds),'--streams',str(streams),'--cache',cache,'--task',task,'--events',str(events)]
    if diagnostic:cmd+=['--diagnostic']
    with (output/(name+'.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
    r=json.loads((output/(name+'.json')).read_text())
    if r['status']!='pass':raise ValueError('Failed case')
    cases.append(dict(name=name,command=cmd,jf_gib_s=r['jf_gib_s'],events_per_second=r['events_per_second']))
    (output/'progress.json').write_text(json.dumps(cases,indent=2)+'\n')
    print(json.dumps(cases[-1]),flush=True)
# Largest bank footprint first at bounded event count, followed by full controls.
execute('pilot-b4-s1',4,1,'cold',events=400)
execute('pilot-b2-s2',2,2,'cold',events=400)
# Same-node standalone read control uses the previously reviewed NIC/cache harness.
control=dict(name='standalone',pattern='events',chunk_mib=0,threads=4,task_mib=4,
             bounce_mib=16,depth=35,direct=False,cpus=4,network_profile=True)
with (output/'standalone.log').open('x') as log:
    subprocess.run([sys.executable,str(HERE.parent/'mfx_kvikio_read_20261108/read_case.py'),
                    '--config',json.dumps(control),'--output',str(output/'standalone.json')],
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
for repeat in (1,2):
    configs=[(1,2,'full'),(2,2,'full'),(4,1,'full')]
    if repeat==1:configs += [(1,1,'full'),(2,1,'full'),(1,2,'parse')]
    # Reverse order on second repeat to expose ordering/history sensitivity.
    if repeat==2:configs.reverse()
    for bds,streams,task in configs:
        for cache in ('cold','warm'):
            execute('%s-b%d-s%d-%s-r%d'%(task,bds,streams,cache,repeat),bds,streams,cache,task)
for bds,streams,task in [(1,2,'full'),(2,2,'full'),(1,2,'parse')]:
    execute('diag-%s-b%d-s%d'%(task,bds,streams),bds,streams,'cold',task,diagnostic=True)
(output/'complete.json').write_text(json.dumps(dict(status='pass',cases=cases),indent=2)+'\n')
