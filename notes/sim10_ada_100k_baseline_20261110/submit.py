"""Freeze the reviewed harness and pinned PR, validate compatibility, then submit once."""
import argparse,json,subprocess,sys
from pathlib import Path
from prepare_runtime import PR_SHA,prepare

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--reviewed-sha',required=True);p.add_argument('--submit',action='store_true')
    a=p.parse_args();repo=Path(__file__).resolve().parents[2]
    git=lambda *args:subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
    if git('rev-parse','HEAD')!=a.reviewed_sha or git('status','--porcelain'):
        raise RuntimeError('Submit only the clean reviewed commit')
    a.attempt=a.attempt.resolve();a.attempt.mkdir(parents=True,exist_ok=False)
    source=a.attempt/'source';pr=a.attempt/'pr-source'
    for target,commit in ((source,a.reviewed_sha),(pr,PR_SHA)):
        subprocess.run(['git','-C',str(repo),'worktree','add','--detach',str(target),commit],check=True)
    prepare(pr,a.attempt/'native-compatibility.json')
    harness=source/'notes'/Path(__file__).parent.name
    command=['sbatch','--parsable','--nodelist=sdfada016','--exclude=sdfada011',
             '--output='+str(a.attempt/'slurm-%j.out'),'--error='+str(a.attempt/'slurm-%j.err'),
             str(harness/'run.sbatch'),str(harness),str(a.attempt)]
    record=dict(harness_sha=a.reviewed_sha,pr_sha=PR_SHA,command=command,submitted=False)
    (a.attempt/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
    if a.submit:
        job=subprocess.check_output(command,text=True).strip();record.update(submitted=True,job_id=job)
        (a.attempt/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
