"""Read current UED timing/camera settings without device initialization writes."""
import json
from datetime import datetime, timezone
import time
from psdaq.debugtools.epixquad1kfps import standalone_hardware as hw
hw.load_modules()
import pyrogue as pr
import rogue.hardware.axi
import rogue.protocols.srp
import surf.axi
import ePixQuad
import lcls2_pgp_pcie_apps

result = {'utc': datetime.now(timezone.utc).isoformat(), 'register_writes': False,
          'runtime': hw.runtime_snapshot(), 'device': '/dev/datadev_0', 'lane': 0}

def read(root, paths):
    values = {}
    for path in paths:
        try:
            node = root
            for part in path.split('.'):
                node = getattr(node, part)
            values[path] = node.get(read=True)
        except Exception as exc:
            values[path] = {'error': str(exc)}
    return values

root = pr.Root(pollEn=False, initRead=False, initWrite=False, timeout=2.0)
mem = rogue.hardware.axi.AxiMemMap('/dev/datadev_0')
root.add(lcls2_pgp_pcie_apps.PcieFpga(name='DevPcie', memBase=mem,
         pgp4=True, enLclsI=False, enLclsII=True, boardType='XilinxKcu1500'))
timing='DevPcie.Hsio.TimingRx.'
paths=[timing+'TimingFrameRx.'+name for name in
       ('ClkSel','ModeSel','ModeSelEn','RxLinkUp','RxDown','sofCount','CrcErrCount','RxDecErrCount')]
paths += [timing+'TimingPhyMonitor.UseMiniTpg',timing+'XpmMiniWrapper.XpmMini.HwEnable']
paths += ['DevPcie.Application.AppLane[0].'+name for name in
          ('VcDataTap','XpmPauseThresh','EventBuilder.Bypass','EventBuilder.Timeout','EventBuilder.Blowoff')]
for i in (0,4):
    paths += [timing+'TriggerEventManager.TriggerEventBuffer[%d].'%i+name for name in
              ('MasterEnable','TriggerSource','TriggerDelay','Partition','TriggerCount')]
    paths += [timing+'TriggerEventManager.EvrV2CoreTriggers.EvrV2ChannelReg[%d].'%i+name for name in
              ('EnableReg','RateType','RateSel','DestType')]
    paths += [timing+'TriggerEventManager.EvrV2CoreTriggers.EvrV2TriggerReg[%d].'%i+name for name in
              ('EnableTrig','Source','Delay','Width','Polarity')]
try:
    root.start()
    result['pcie']=read(root, paths)
finally:
    if root.running: root.stop()

root=pr.Root(pollEn=False, initRead=False, initWrite=False, timeout=2.0)
dma=rogue.hardware.axi.AxiStreamDma('/dev/datadev_0',1,True)
srp=rogue.protocols.srp.SrpV3()
pr.streamConnectBiDir(dma,srp)
root.addInterface(dma)
root.addInterface(srp)
for name, cls, offset in [('AxiVersion',ePixQuad.EpixVersion,0),
                          ('SystemRegs',ePixQuad.SystemRegs,0x00100000),
                          ('AcqCore',ePixQuad.AcqCore,0x01000000),
                          ('RdoutCore',ePixQuad.RdoutCore,0x01100000),
                          ('RdoutStreamMonitoring',surf.axi.AxiStreamMonAxiL,0x01300000)]:
    root.add(cls(name=name,memBase=srp,offset=offset))
paths=['SystemRegs.'+name for name in ('TrigEn','TrigSrcSel','AutoTrigEn','AutoTrigPer',
       'AutoTrigFreqHz','DcDcEnable','AsicAnaEn','AsicDigEn','DdrVttEn','AsicMask','AdcTestDone','AdcTestFailed')]
paths += ['AcqCore.'+name for name in ('AcqToAsicR0Delay','AsicR0Width','AsicR0ToAsicAcq',
          'AsicAcqWidth','AsicAcqLToPPmatL','AsicPpmatToReadout','AsicRoClkHalfT','AsicRoClkT',
          'AsicSyncInjEn','AsicSyncInjDly','DummyAcqEn')]
paths += ['RdoutCore.'+name for name in ('RdoutEn','SeqCount','AdcPipelineDelay','BuffersRdy',
          'OverSampleEn','OverSampleSize','sAxisDropFrameCount','mAxisDropFrameCount')]
stream_paths=['RdoutStreamMonitoring.Ch[0].'+name for name in ('FrameCnt','FrameRate','FrameSize','FrameSizeMin','FrameSizeMax')]
try:
    root.start()
    result['identity']=hw.camera_identity(root)
    result['camera']=read(root,paths)
    result['stream_first']=read(root,stream_paths)
    time.sleep(1)
    result['stream_second']=read(root,stream_paths)
finally:
    if root.running: root.stop()
print(json.dumps(result,indent=2),flush=True)
