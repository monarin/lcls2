"""Exercise the actual baseline loader against memory emulation, never hardware."""
from pathlib import Path
from psdaq.debugtools.epixquad1kfps import standalone_hardware as hw
hw.load_modules()
import ePixQuad
import ePixAsics
import pyrogue as pr
from pyrogue.interfaces.simulation import MemEmulate

baseline=Path(__file__).resolve().parents[2]/'psdaq/psdaq/debugtools/epixquad1kfps/baselines/ued_beam_20260928.yml'
root=pr.Root(name='Top',pollEn=False,initRead=False,initWrite=False)
mem=MemEmulate()
for name,cls,offset in [('SystemRegs',ePixQuad.SystemRegs,0x00100000),
                       ('AcqCore',ePixQuad.AcqCore,0x01000000),
                       ('RdoutCore',ePixQuad.RdoutCore,0x01100000),
                       ('VguardDac',ePixQuad.VguardDac,0x00500000)]:
    root.add(cls(name=name,memBase=mem,offset=offset))
for i in range(16):
    root.add(ePixAsics.Epix10kaAsic(name='Epix10kaSaci[%d]'%i,
             memBase=mem,offset=0x04000000+i*0x00400000,enabled=False))
try:
    root.start()
    hw.load_camera_baseline(root,str(baseline),list(range(16)))
    assert root.AcqCore.AsicRoClkHalfT.get(read=True)==0xaaaa0003
    assert root.RdoutCore.AdcPipelineDelay.get(read=True)==0xaaaa0044
    assert not root.SystemRegs.TrigEn.get(read=True)
    assert not root.SystemRegs.AutoTrigEn.get(read=True)
    assert not root.RdoutCore.RdoutEn.get(read=True)
    assert not root.AcqCore.DummyAcqEn.get(read=True)
    assert not root.AcqCore.AsicSyncInjEn.get(read=True)
    for i in range(16):
        asic=root.Epix10kaSaci[i]
        assert not asic.enable.get()
        asic.enable.set(True)
        assert not asic.IsEn.get(read=True)
        assert asic.RowStopAddr.get(read=True)==177
        assert asic.ColStopAddr.get(read=True)==47
    print('PASS: baseline loaded through real PyRogue with emulated memory; all 16 ASICs configured and disabled; triggers/readout off; keyed clock and pipeline values correct')
finally:
    if root.running: root.stop()
