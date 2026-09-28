"""Build an offline-reviewed UED baseline from the saved ConfigDB response.

Construct register models only: no roots, hardware transports, or device access.
"""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import yaml
from psdaq.debugtools.epixquad1kfps import standalone_hardware as hw
hw.load_modules()
import ePixQuad
import ePixAsics

folder=Path(__file__).parent
raw=gzip.decompress((folder/'beam-config.json.gz').read_bytes())
config=json.loads(raw)
source=config['expert']['EpixQuad']
output=folder.parents[1]/'psdaq/psdaq/debugtools/epixquad1kfps/baselines/ued_beam_20260928.yml'
models={'SystemRegs':ePixQuad.SystemRegs(), 'AcqCore':ePixQuad.AcqCore(),
        'RdoutCore':ePixQuad.RdoutCore(memBase=None,offset=0),
        'VguardDac':ePixQuad.VguardDac()}
for i in range(16):
    models['Epix10kaSaci%d'%i]=ePixAsics.Epix10kaAsic()
blocks={}
excluded=[]
for name,values in source.items():
    if name not in models:
        excluded.append({'path':name,'reason':'No scope or ADC-test commands in this image-only baseline'})
        continue
    mapped='Epix10kaSaci[%s]'%name[len('Epix10kaSaci'):] if name.startswith('Epix10kaSaci') else name
    dest={'enable':True}
    if name.startswith('Epix10kaSaci'):
        dest['IsEn']=True
    for key,value in values.items():
        path=name+'.'+key
        node=models[name].nodes.get(key)
        if name.startswith('Epix10kaSaci') and key=='IsEn':
            continue
        if (name.startswith('Epix10kaSaci') and key=='PixelDummy') or (
            name=='Epix10kaSaci3' and (key.startswith('CompEn') or key in ('Preamp','MonostPulser','PulserDac'))):
            excluded.append({'path':path,'reason':'Same explicit skip as epixquad1kfps_config.apply_dict'})
            continue
        if node is None or (node.mode != 'RW' and not (name.startswith('Epix10kaSaci') and key in ('RowStartAddr','ColStartAddr') and node.mode == 'WO')):
            excluded.append({'path':path,'reason':'Absent or not RW in the pinned register model'})
            continue
        if (name=='AcqCore' and key in ('AsicRoClkT','AsicRoClkHalfT')) or (name=='RdoutCore' and key=='AdcPipelineDelay'):
            value |= 0xaaaa0000
        dest[key]=copy.deepcopy(value)
    blocks[mapped]=dest
blocks['SystemRegs'].update(TrigEn=False,AutoTrigEn=False)
blocks['RdoutCore']['RdoutEn']=False
blocks['AcqCore']['AsicSyncInjEn']=False
header=('# UED BEAM configuration snapshot, 2026-09-28. Not hardware validated.\n'
        '# Source JSON SHA256: '+hashlib.sha256(raw).hexdigest()+'\n'
        '# No ASIC bitmap, ADC training, PROM, power-cycle or scope operations.\n'
        '# SACI IsEn remains true for configuration; the gain writer disables it.\n')
output.write_text(header+yaml.safe_dump({'Top':blocks},sort_keys=False))
report={'source_sha256':hashlib.sha256(raw).hexdigest(),'yaml_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'excluded':excluded,'test_overrides':{'SystemRegs.TrigEn':False,'SystemRegs.AutoTrigEn':False,
        'RdoutCore.RdoutEn':False,'AcqCore.AsicSyncInjEn':False,'ASIC IsEn':True},
        'write_keys':['AcqCore.AsicRoClkHalfT','RdoutCore.AdcPipelineDelay'],
        'note':'ASIC gain map/trbit programmed after baseline; load_camera_baseline enables IsEn before LoadConfig and disables it afterward'}
(folder/'baseline-review.json').write_text(json.dumps(report,indent=2)+'\n')
print(output)
print(json.dumps(report,indent=2))
