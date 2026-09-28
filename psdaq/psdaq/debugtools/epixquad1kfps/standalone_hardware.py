"""Shared hardware startup for profiled writer, GUI and headless capture.

No hardware imports or device access occur at module import time.
"""
from contextlib import contextmanager, ExitStack
import sys

from psdaq.debugtools.epixquad1kfps.standalone_profile import (
    check_identity, validate_hardware,
)


def load_modules():
    from psdaq.utils import enable_lcls2_pgp_pcie_apps, enable_epix_quad1kfps  # noqa: F401
    import ePixQuad
    import lcls2_pgp_pcie_apps
    return ePixQuad, lcls2_pgp_pcie_apps


def probe_pcie(device):
    """Read BAR0 version registers before starting DevRoot (which writes timing)."""
    import axipcie
    import pyrogue as pr
    import rogue.hardware.axi
    mem = rogue.hardware.axi.AxiMemMap(device)
    root = pr.Root(pollEn=False, initRead=False)
    version = axipcie.PcieAxiVersion(offset=0x20000, memBase=mem)
    root.add(version)
    try:
        root.start()
        build = version.BuildStamp.get(read=True)
        return {'board_type': version.PCIE_HW_TYPE_G.getDisp(read=True),
                'image_name': version.ImageName.get(read=True),
                'fpga_version': int(version.FpgaVersion.get(read=True)),
                'build_stamp': build, 'git_hash': int(version.GitHash.get(read=True))}
    finally:
        root.stop()


def configure_lane(pbase, profile):
    app = getattr(pbase.DevPcie.Application, 'AppLane[%d]' % profile['lane'])
    app.VcDataTap.set(profile['data_vc'])
    app.XpmPauseThresh.set(0x20)
    app.EventBuilder.Timeout.set(profile['event_builder_timeout'])
    app.EventBuilder.Bypass.set(0)
    app.EventBuilder.Blowoff.set(True)


@contextmanager
def session(profile, camera_vc_mask=0x2, writer=False, on_identity=None,
            poll_en=False, init_read=False):
    validate_hardware(profile)
    if not 0 < camera_vc_mask <= 0xf or not camera_vc_mask & 0x2:
        raise ValueError('Camera VC mask must include register VC1')
    ePixQuad, pcie = load_modules()
    identity = probe_pcie(profile['device'])
    if on_identity:
        on_identity(identity)
    check_identity(profile, identity)
    # Register cleanup before start: __enter__ failures must also release roots.
    with ExitStack() as stack:
        pbase = pcie.DevRoot(
            dev=profile['device'], enLclsI=False, enLclsII=True,
            yamlFileLclsI=None, yamlFileLclsII=profile['pcie_yaml'],
            startupMode=True, standAloneMode=profile['timing'] == 'xpmmini',
            pgp4=profile['pgp4'], dataVc=profile['data_vc'],
            pcieBoardType=profile['board_type'], pollEn=poll_en, initRead=init_read)
        stack.callback(pbase.stop)
        pbase.start()
        stack.callback(pbase.StopRun)
        configure_lane(pbase, profile)
        cbase = ePixQuad.Top(
            dev=profile['device'], hwType='datadev', lane=profile['lane'],
            promWrEn=profile['prom_bypass'], enVcMask=camera_vc_mask,
            enWriter=writer, enPrbs=False, pollEn=poll_en, initRead=init_read)
        stack.callback(cbase.stop)
        cbase.start()
        yield pbase, cbase


def camera_identity(camera):
    version = camera.AxiVersion
    regs = camera.SystemRegs
    result = {'fpga_version': int(version.FpgaVersion.get(read=True)),
            'build_stamp': version.BuildStamp.get(read=True),
            'git_hash': int(version.GitHash.get(read=True)),
            'carrier_ids': [[int(regs.CarrierIdLow[i].get(read=True)),
                             int(regs.CarrierIdHigh[i].get(read=True))] for i in range(4)],
            'adc_done': bool(regs.AdcTestDone.get(read=True)),
            'adc_failed': bool(regs.AdcTestFailed.get(read=True))}
    result['adc_done'] = result['adc_done'] and bool(regs.AdcTestDone.get(read=True))
    return result


def load_camera_baseline(camera, path, asics):
    """Enable SACI writes before bulk YAML loading, then disable every ASIC.

    Call with camera triggers/readout disabled. Like DAQ config_expert, IsEn
    must reach hardware before other ASIC settings, not in the same bulk write.
    """
    entered = []
    try:
        for index in asics:
            saci = camera.Epix10kaSaci[index]
            saci.enable.set(True)
            entered.append(saci)
            saci.IsEn.set(True)
        camera.LoadConfig(path)
    finally:
        primary = sys.exc_info()[1]
        errors = []
        for saci in entered:
            for variable in (saci.IsEn, saci.enable):
                try:
                    variable.set(False)
                except Exception as exc:
                    errors.append(repr(exc))
        if errors:
            message = 'ASIC cleanup after baseline load failed: '+ '; '.join(errors)
            print(message, file=sys.stderr)
            if primary is None:
                raise RuntimeError(message)


def runtime_snapshot():
    import rogue
    names = ('rogue', 'pyrogue', 'ePixQuad', 'lcls2_pgp_pcie_apps', 'surf',
             'axipcie', 'lcls2_pgp_fw_lib', 'l2si_core', 'LclsTimingCore')
    return {'rogue_version': rogue.Version.current(),
            'module_paths': {name: getattr(sys.modules.get(name), '__file__', None)
                             for name in names}}
