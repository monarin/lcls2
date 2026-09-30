"""Select a gain writer without copying the DAQ programming implementation.

Importing this module, preparing maps, and describing writers never import
hardware packages. Only resolve_writer('daq') imports the DAQ module.
"""
import importlib
from pathlib import Path
import sys

import numpy as np

from psdaq.debugtools.epixquad1kfps import write_gain_mode_standalone as gain
from psdaq.debugtools.epixquad1kfps.standalone_profile import file_sha256


def pixel_config_map(mode, selected):
    """Build DAQ register codes, not the boolean expected output gain-bit map."""
    result = np.full(gain.RAW_SHAPE, mode.background_value, dtype=np.uint8)
    for asic, row, col in selected:
        coord = gain._direct_to_raw(asic, row, col)
        if coord is None:
            raise ValueError('DAQ raw-map writer cannot represent selected ghost rows')
        result[coord] = mode.selected_value
    return result


def describe_writer(name):
    if name == 'standalone':
        path = Path(gain.__file__).resolve()
        function = '_write_gain_mode'
        sequence = 'all ASIC backgrounds, then selected pixels in input order'
    elif name == 'daq':
        path = Path(__file__).resolve().parents[2] / 'configdb/epixquad1kfps_config.py'
        function = '_apply_raw_pixel_map_program'
        sequence = 'per-ASIC most-common background followed by exceptions'
    else:
        raise ValueError('Unknown gain writer: '+name)
    return {'name': name, 'source_path': str(path), 'source_sha256': file_sha256(path),
            'function': function, 'sequence': sequence,
            'scope': 'gain-map programming only; standalone baseline and timing'}


def resolve_writer(description):
    """Import the actual DAQ helper and refuse a different installed checkout."""
    if description['name'] == 'standalone':
        module = gain
    else:
        module = importlib.import_module('psdaq.configdb.epixquad1kfps_config')
    path = Path(module.__file__).resolve()
    if str(path) != description['source_path'] or file_sha256(path) != description['source_sha256']:
        raise RuntimeError('Loaded gain writer differs from the recorded source')
    return getattr(module, description['function'])


def program(camera, args, selected, description, writer):
    if description['name'] == 'standalone':
        return writer(camera, args.mode, args.asics, selected, [],
                      args.injection_delay, verbose=args.verbose)
    if args.asics != list(range(16)):
        raise ValueError('DAQ raw-map helper requires all 16 ASICs')
    config = pixel_config_map(args.mode, selected)
    entered = []
    try:
        for index in args.asics:
            saci = camera.Epix10kaSaci[index]
            entered.append(saci)
            saci.enable.set(True)
            saci.IsEn.set(True)
        for index in args.asics:
            camera.Epix10kaSaci[index].trbit.set(int(args.mode.trbit))
        writer(camera, config, args.asics)
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
            message = 'ASIC cleanup after DAQ map write failed: '+ '; '.join(errors)
            print(message, file=sys.stderr)
            if primary is None:
                raise RuntimeError(message)
