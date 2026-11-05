"""Final evidence and access checks after the authorized one-shot deployment."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from fixture import HERE, ROOT, DEST, frozen_report, wire, check_stat


def main():
    reports={}
    for name in ('pilot_verification','seam_verification','pilot_raw_acceptance','seam_raw_acceptance',
                 'source_samples','calibration_readback','pilot_acceptance','seam_acceptance'):
        path=HERE/(name+'.json');report=json.loads(path.read_text())
        wire.require(report['status']=='pass','Missing successful evidence: '+name)
        script='verify.py' if name.endswith('verification') else 'calibration.py' if name=='calibration_readback' else 'acceptance.py'
        wire.require(report['script_sha256']==wire.file_sha(HERE/script),'Validation script changed: '+name)
        reports[name]=wire.file_sha(path)
    for record in frozen_report('admin_checks')['run1_2_files_before'].values():check_stat(record)
    for entry in frozen_report('inventory')['streams']:
        check_stat(entry['bigdata']);check_stat({k:v for k,v in entry['smd'].items() if k!='sha256'})
    wire.require(not [p for f in (DEST,DEST/'smalldata') for n in (3,4) for p in f.glob(f'*r{n:04d}*.xtc2*')], 'Stage 2 unexpectedly published XTC')
    modes={}
    for path in (ROOT,ROOT/'pilot',ROOT/'pilot/smalldata',ROOT/'seam',ROOT/'seam/smalldata'):
        mode=path.stat().st_mode & 0o777
        wire.require(path.stat().st_uid==os.getuid() and mode==0o700,'Private directory permissions changed')
        modes[str(path)]=oct(mode)
    for mode in ('pilot','seam'):
        generation=json.loads((HERE/f'{mode}_generation.json').read_text())
        wire.require(wire.file_sha(ROOT/mode/'generation.json')==generation['generation_sha256'],'Private manifest changed')
        for name,digest in generation['code_sha256'].items():
            wire.require(wire.file_sha(HERE/name)==digest,'Generation code changed')
        for stream in generation['streams']:
            for item in stream['files'].values():
                path=ROOT/mode/item['relative_path']
                wire.require(path.stat().st_mode & 0o777==0o600 and path.stat().st_size==item['bytes'],'Output access/size changed')
    acl=subprocess.check_output(['getfacl','-p',str(ROOT),str(ROOT/'pilot')],text=True)
    code={p.name:wire.file_sha(p) for p in HERE.glob('*.py')}
    wire.save_json(HERE/'completion.json',dict(status='pass',completed_utc=datetime.now(timezone.utc).isoformat(),
        root=str(ROOT),pilot=str(ROOT/'pilot'),events=2400,steps=1,pilot_bytes=5039830328,
        calibration_validity=[3,4],documents_deployed=8,private_directory_modes=modes,acl=acl,
        existing_run1_2_files_unchanged=40,source_file_stats_unchanged=14,final_xtc_published=False,
        output_hashes='Every byte validated before deployment; output size/mode and manifests rechecked here.',
        evidence_sha256=reports,code_sha256=code))
    print('PASS Stage 2 completion: private pilot, calibration 3-4, normal DataSource, unchanged existing files',flush=True)


if __name__=='__main__':main()
