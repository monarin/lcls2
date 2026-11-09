"""One-node cold FFB framework matrix, all cases use one BD and one GPU."""
import json,os,shutil,signal,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from support import HERE,NOTES,raw,ffb,require,sha,load_fixture,write,source_hashes

def main(output):
    output.mkdir(parents=True,exist_ok=False);stage=ffb.FFB/('jf-history-sim10-'+os.environ['SLURM_JOB_ID']);stage.mkdir(mode=0o700)
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    try:
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id,name','--format=csv,noheader'],text=True)
        os.environ['JF_GPU_UUID']=inventory.splitlines()[0].split(',')[0].strip()
        rawfiles=raw.paths(raw.SOURCE);expected=json.loads((raw.HERE/'expected-sha256.json').read_text())
        with ThreadPoolExecutor(max_workers=4) as pool:hashes=list(pool.map(sha,rawfiles))
        require(hashes==[expected[p.name] for p in rawfiles],'Raw SHA mismatch')
        rawplan,rawstamps=raw.requests(raw.SOURCE)
        print('Uncompressed source SHA/SMD verified',flush=True)
        files,sizes,plan,identity=load_fixture();require(shutil.disk_usage(stage).free>sum(sizes)+(2<<30),'FFB capacity')
        small=[];(stage/'smalldata').mkdir()
        receipt=json.loads((NOTES/'mfx_expand10_10k_20261106/files.json').read_text())
        refs={r['path']:r for r in receipt['files']}
        for p,row in zip(files,identity['files']):
            target=stage/p.name;shutil.copyfile(p,target);require(sha(target)==row['sha256'],'Sim10 SHA');small.append(target)
            smd=p.parent/'smalldata'/p.name.replace('.xtc2','.smd.xtc2');dest=stage/'smalldata'/smd.name
            shutil.copyfile(smd,dest);require(sha(dest)==refs[str(smd)]['sha256'],'SMD SHA')
        manifest=json.loads((files[0].parent/'generation.json').read_text())
        inputs=dict(raw=dict(exp='mfx101210926',run=387,directory=str(raw.SOURCE),files=list(map(str,rawfiles)),timestamps=rawstamps,event_bytes=33557176),
            sim10=dict(exp='mfx101555026',run=10,directory=str(stage),files=list(map(str,small)),timestamps=[r['timestamp'] for r in manifest['schedule'] if r['service']==12],event_bytes=3358200))
        frozen=json.loads((NOTES/'jf_framework_cold_20261109/sim10_identity_check.json').read_text())
        require(sha(files[0].parent/'generation.json')==frozen['accepted_generation_sha256'] and raw.ts_hash(inputs['sim10']['timestamps'])==frozen['timestamp_sha256'],'Sim10 frozen identity')
        write(output/'inputs.json',inputs)
        mounts={k:json.loads(subprocess.check_output(['findmnt','-J','-T',v['directory']],text=True)) for k,v in inputs.items()}
        require(all(v['filesystems'][0]['source']=='drpsrcf/ffb' for v in mounts.values()),'Require FFB for both')
        write(output/'provenance.json',dict(raw_sha256=expected,sim10_identity=identity,mounts=mounts,gpu_inventory=inventory,source_sha256=source_hashes()))
        cases=[]
        def execute(name,mode,batch=20,events=10000,preflight=False):
            cmd=['numactl','--interleave=all','mpiexec','-n','3','--bind-to','none',sys.executable,str(HERE/'runner.py'),'--inputs',str(output/'inputs.json'),'--output',str(output/(name+'.json')),'--mode',mode,'--events',str(events),'--batch-size',str(batch)]
            if preflight:cmd.append('--preflight')
            print('Starting '+name,flush=True)
            with (output/(name+'.log')).open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            r=json.loads((output/(name+'.json')).read_text());require(r['status']=='pass','Case failure')
            cases.append(dict(name=name,mode=mode,batch=batch,events=events,command=cmd,gib_s=r['gib_s'],hz=r['hz']))
            write(output/'progress.json',cases);print(json.dumps(cases[-1]),flush=True)
        execute('preflight-raw-dense','raw-dense',events=200,preflight=True)
        execute('preflight-sim10-field','sim10-field',batch=100,events=200)
        for i,(mode,batch) in enumerate([('raw-dense',20),('sim10-field',20),('sim10-field',100),('sim10-field',100),('sim10-field',20),('raw-dense',20)]):execute('%02d-%s-b%d'%(i,mode,batch),mode,batch=batch)
        write(output/'complete.json',dict(status='pass',cases=cases))
    finally:
        shutil.rmtree(stage);write(output/'cleanup.json',dict(stage=str(stage),removed=not stage.exists()))

if __name__=='__main__':main(Path(sys.argv[1]))
