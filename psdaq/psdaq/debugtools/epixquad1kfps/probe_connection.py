#!/usr/bin/env python3
"""Read PCIe identity, PGP4 link status and optional camera VC1 registers.

No DevRoot/Top startup, YAML load, polling, resets, training, or register writes.
Without --lane, only PCIe identity is read. A lane probe requires a PGP4 image
and a ready link before opening its SRPv3 register channel. The DMA driver must
allow the register channel to be opened; do not stop another user to force it.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from psdaq.debugtools.epixquad1kfps import standalone_hardware as hardware


def read_lane(device, lane):
    import pyrogue as pr
    import rogue.hardware.axi
    import surf.protocols.pgp as pgp
    root = pr.Root(pollEn=False, initRead=False, initWrite=False, timeout=2.0)
    mem = rogue.hardware.axi.AxiMemMap(device)
    link = pgp.Pgp4AxiL(name='Link', offset=0x00800000 + lane*0x10000,
                       memBase=mem, numVc=4, writeEn=False)
    root.add(link)
    try:
        root.start()
        return {'rx_ready': bool(link.RxStatus.LinkReady.get(read=True)),
                'remote_ready': bool(link.RxStatus.RemRxLinkReady.get(read=True)),
                'tx_ready': bool(link.TxStatus.LinkReady.get(read=True)),
                'rx_clock_hz': int(link.RxStatus.RxClockFreqRaw.get(read=True)),
                'tx_clock_hz': int(link.TxStatus.TxClockFreqRaw.get(read=True)),
                'link_down_count': int(link.RxStatus.LinkDownCnt.get(read=True)),
                'link_error_count': int(link.RxStatus.LinkErrorCnt.get(read=True))}
    finally:
        if root.running:
            root.stop()


def read_camera(device, lane):
    import ePixQuad
    import pyrogue as pr
    import rogue.hardware.axi
    import rogue.protocols.srp
    root = pr.Root(pollEn=False, initRead=False, initWrite=False, timeout=2.0)
    dma = rogue.hardware.axi.AxiStreamDma(device, lane*256 + 1, True)
    srp = rogue.protocols.srp.SrpV3()
    pr.streamConnectBiDir(dma, srp)
    root.addInterface(dma)
    root.addInterface(srp)
    root.add(ePixQuad.EpixVersion(name='AxiVersion', offset=0, memBase=srp))
    root.add(ePixQuad.SystemRegs(name='SystemRegs', offset=0x00100000, memBase=srp))
    try:
        root.start()
        result = hardware.camera_identity(root)
        regs = root.SystemRegs
        result['asic_mask'] = int(regs.AsicMask.get(read=True))
        result['trigger_enabled'] = bool(regs.TrigEn.get(read=True))
        result['auto_trigger_enabled'] = bool(regs.AutoTrigEn.get(read=True))
        result['trigger_source'] = int(regs.TrigSrcSel.get(read=True))
        return result
    finally:
        if root.running:
            root.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev', default='/dev/datadev_0')
    parser.add_argument('--lane', type=int, choices=range(4), help='Optional PGP4 camera lane to read')
    args = parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(),
              'device': args.dev, 'lane': args.lane, 'register_writes': False}
    try:
        hardware.load_modules()
        result['runtime'] = hardware.runtime_snapshot()
        result['pcie'] = hardware.probe_pcie(args.dev)
        if args.lane is not None:
            if 'Pgp4' not in result['pcie']['image_name']:
                raise ValueError('Lane register layout only verified for PGP4 firmware')
            result['link'] = read_lane(args.dev, args.lane)
            if not all(result['link'][key] for key in ('rx_ready', 'remote_ready', 'tx_ready')):
                raise RuntimeError('PGP link is not ready; camera register channel was not opened')
            result['camera'] = read_camera(args.dev, args.lane)
        result['status'] = 'ok'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = repr(exc)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
