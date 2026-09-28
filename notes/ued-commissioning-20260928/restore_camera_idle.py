"""Restore the preflight camera idle trigger state after this commissioning run."""
import json
from pathlib import Path
import socket
import sys
from psdaq.debugtools.epixquad1kfps import standalone_hardware as hw

folder=Path(__file__).resolve().parent
before=json.loads((folder/'preflight.json').read_text())
if socket.gethostname().split('.')[0]!='drp-ued-cmp003':
    raise RuntimeError('Wrong host')
state=before['camera']
assert state['RdoutCore.RdoutEn'] is False
assert state['SystemRegs.TrigSrcSel']==3
assert state['SystemRegs.AutoTrigEn'] is True
assert state['SystemRegs.TrigEn'] is True
hw.load_modules()
import ePixQuad
import pyrogue as pr
import rogue.hardware.axi
import rogue.protocols.srp
root=pr.Root(pollEn=False,initRead=False,initWrite=False,timeout=2.0)
dma=rogue.hardware.axi.AxiStreamDma('/dev/datadev_0',1,True)
srp=rogue.protocols.srp.SrpV3()
pr.streamConnectBiDir(dma,srp)
root.addInterface(dma)
root.addInterface(srp)
for name,cls,offset in [('AxiVersion',ePixQuad.EpixVersion,0),
                       ('SystemRegs',ePixQuad.SystemRegs,0x00100000),
                       ('RdoutCore',ePixQuad.RdoutCore,0x01100000)]:
    root.add(cls(name=name,memBase=srp,offset=offset))
result={'status':'failed','scope':'camera idle triggering only; PCIe remains in standalone test configuration'}
try:
    root.start()
    identity=hw.camera_identity(root)
    if identity['carrier_ids']!=before['identity']['carrier_ids'] or identity['git_hash']!=before['identity']['git_hash']:
        raise RuntimeError('Camera identity changed; no restoration writes made')
    regs=root.SystemRegs
    regs.TrigEn.set(False)
    regs.AutoTrigEn.set(False)
    root.RdoutCore.RdoutEn.set(False)
    regs.TrigSrcSel.set(state['SystemRegs.TrigSrcSel'])
    regs.AutoTrigPer.set(state['SystemRegs.AutoTrigPer'])
    regs.AutoTrigEn.set(state['SystemRegs.AutoTrigEn'])
    regs.TrigEn.set(state['SystemRegs.TrigEn'])
    actual={'SystemRegs.'+name:getattr(regs,name).get(read=True) for name in
            ('TrigSrcSel','AutoTrigPer','AutoTrigEn','TrigEn')}
    actual['RdoutCore.RdoutEn']=root.RdoutCore.RdoutEn.get(read=True)
    if any(value!=state[key] for key,value in actual.items()):
        raise RuntimeError('Idle trigger readback did not match preflight')
    result.update(status='restored',readback=actual,identity=hw.camera_identity(root))
except Exception as exc:
    result['error']=repr(exc)
    raise
finally:
    if root.running: root.stop()
    (folder/'restore.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
