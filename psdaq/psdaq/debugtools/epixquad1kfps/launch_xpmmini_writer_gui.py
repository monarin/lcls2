#!/usr/bin/env python3

"""Launch the standalone ePixQuad writer GUI using an explicit host profile.

Use capture_standalone.py for headless capture with automatic test records.
No roots are constructed for --dry-run. Hardware startup requires exclusive
access to the card and may configure timing and the detector.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from psdaq.debugtools.epixquad1kfps.standalone_profile import load_profile
from psdaq.debugtools.epixquad1kfps.standalone_hardware import session


def _parse_bool(text):
    value = str(text).lower()
    if value in ("1", "true", "t", "yes", "y"):
        return True
    if value in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError('Expected true or false')


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Launch profiled timing plus ePixQuad StreamWriter PyDM GUI."
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dev", default=None, help="datadev device path")
    parser.add_argument("--lane", type=int, default=None, help="PGP lane connected to the camera")
    parser.add_argument("--data-vc", type=int, default=None, help="VC selected by AppLane.VcDataTap")
    parser.add_argument(
        "--camera-vc-mask",
        type=lambda value: int(value, 0),
        default=0xF,
        help="VC mask for ePixQuad.Top; default 0xf opens data/register/scope/monitoring",
    )
    parser.add_argument(
        "--prom-bypass",
        type=_parse_bool,
        default=None,
        help="pass promWrEn to ePixQuad.Top; profile controls startup PROM validation",
    )
    parser.add_argument(
        "--lcls2-yaml",
        default=None,
        help="lcls2-pgp-pcie-apps LCLS-II defaults YAML loaded at DevRoot startup",
    )
    parser.add_argument(
        "--no-lcls2-yaml",
        action="store_true",
        help="do not load lcls2-pgp-pcie-apps LCLS-II defaults YAML",
    )
    parser.add_argument(
        "--start-run",
        action="store_true",
        help="execute DevRoot.StartRun() after both roots are open",
    )
    parser.add_argument(
        "--pollEn",
        type=_parse_bool,
        default=False,
        help="enable PyRogue polling",
    )
    parser.add_argument(
        "--initRead",
        type=_parse_bool,
        default=False,
        help="read all variables at root startup",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    overrides = {field: getattr(args, arg) for arg, field in
                 [('dev', 'device'), ('lane', 'lane'), ('data_vc', 'data_vc'),
                  ('prom_bypass', 'prom_bypass'), ('lcls2_yaml', 'pcie_yaml')]
                 if getattr(args, arg) is not None}
    if args.no_lcls2_yaml:
        overrides['pcie_yaml'] = None
    profile = load_profile(args.profile, overrides)
    print(json.dumps(profile, indent=2))
    if args.dry_run:
        return 0
    ui_path = str(Path(__file__).with_name('xpmmini_writer_pydm.py'))
    with session(profile, camera_vc_mask=args.camera_vc_mask, writer=True,
                 poll_en=args.pollEn, init_read=args.initRead) as (pbase, cbase):
        import pyrogue.pydm
        cbase.StreamWriter.hidden = False
        if args.start_run:
            pbase.StartRun()
        print('Use Top.StreamWriter to open/close the data file and DevRoot to start/stop.')
        pyrogue.pydm.runPyDM(
            serverList=f"{pbase.zmqServer.address},{cbase.zmqServer.address}",
            ui=ui_path, title="ePixQuad Standalone Writer", sizeX=1000, sizeY=1000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
