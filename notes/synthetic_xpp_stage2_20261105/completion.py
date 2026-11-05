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
    previous=frozen_report('admin_checks')['run1_2_files_before']
    for record in previous.values():check_stat(record)
    source_entries=frozen_report('inventory')['streams']
    for entry in source_entries:
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
    pilot=json.loads((HERE/'pilot_generation.json').read_text())
    accepted=json.loads((HERE/'pilot_acceptance.json').read_text())
    deployed=json.loads((HERE/'calibration_readback.json').read_text())
    code={p.name:wire.file_sha(p) for p in HERE.glob('*.py')}
    wire.save_json(HERE/'completion.json',dict(status='pass',completed_utc=datetime.now(timezone.utc).isoformat(),
        root=str(ROOT),pilot=str(ROOT/'pilot'),events=accepted['events'],steps=len(accepted['step_counts']),pilot_bytes=pilot['total_bytes'],
        calibration_validity=[3,4],documents_deployed=deployed['document_count'],private_directory_modes=modes,acl=acl,
        existing_run1_2_files_unchanged=len(previous),source_file_stats_unchanged=2*len(source_entries),final_xtc_published=False,
        output_hashes='Every byte validated before deployment; output size/mode and manifests rechecked here.',
        evidence_sha256=reports,code_sha256=code))
    print('PASS Stage 2 completion: private pilot, calibration 3-4, normal DataSource, unchanged existing files',flush=True)


if __name__=='__main__':main()
