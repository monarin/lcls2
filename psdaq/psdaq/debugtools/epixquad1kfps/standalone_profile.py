"""Host profiles and provenance. Importing this module never opens hardware."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
from psdaq.debugtools.epixquad1kfps import ued_external_timing


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_profile(path, overrides=None, bundle=None):
    path = Path(path).expanduser().resolve()
    profile = json.loads(path.read_text())
    required = {'schema_version', 'name', 'status', 'hosts', 'device', 'lane',
                'data_vc', 'pgp4', 'timing', 'board_type', 'image_name_contains',
                'minimum_pcie_version', 'prom_bypass', 'pcie_yaml', 'camera_yaml',
                'camera_trigger_source', 'camera_auto_trigger', 'rate_hz',
                'event_builder_timeout', 'image_payload_bytes'}
    if not isinstance(profile, dict) or not required <= profile.keys():
        raise ValueError('Profile is missing required fields')
    if profile['schema_version'] != 1:
        raise ValueError('Unsupported profile schema_version')
    if profile['status'] not in ('ready_for_test', 'unvalidated'):
        raise ValueError('Unknown profile status')
    if not isinstance(profile['hosts'], list) or not profile['hosts'] or any(
            not isinstance(host, str) or not host for host in profile['hosts']):
        raise ValueError('Profile hosts must be a nonempty list of host names')
    resolved = copy.deepcopy(profile)
    overrides = overrides or {}
    if set(overrides) - required:
        raise ValueError('Unknown profile override')
    resolved.update(overrides)
    bundle = bundle or os.environ.get('SUBMODULEDIR')
    for field in ('pcie_yaml', 'camera_yaml'):
        value = resolved[field]
        if value is not None:
            target = Path(value).expanduser()
            if not target.is_absolute():
                if value.startswith(('./', '../')):
                    target = path.parent / target
                elif not bundle:
                    raise ValueError('Set SUBMODULEDIR to resolve profile YAML paths')
                else:
                    target = Path(bundle) / target
            resolved[field] = str(target.resolve())
    resolved['profile_path'] = str(path)
    resolved['profile_sha256'] = file_sha256(path)
    resolved['overrides'] = overrides
    ued_external_timing.validate(resolved)
    return resolved


def validate_hardware(profile, capture=False, hostname=None):
    """Fail before importing hardware modules or opening a device."""
    if profile['status'] != 'ready_for_test':
        raise ValueError('Profile is unvalidated; hardware access is disabled')
    host = (hostname or socket.gethostname()).split('.')[0]
    if host not in profile['hosts']:
        raise ValueError('Host %s is not allowed by profile %s' % (host, profile['name']))
    for field in ('device', 'board_type', 'image_name_contains'):
        if not isinstance(profile[field], str) or not profile[field]:
            raise ValueError('Missing profile field: '+field)
    for field in ('pgp4', 'prom_bypass'):
        if type(profile[field]) is not bool:
            raise ValueError('Expected boolean profile field: '+field)
    for field, low, high in (('lane', 0, 3), ('data_vc', 0, 3),
                             ('minimum_pcie_version', 1, 0xffffffff),
                             ('event_builder_timeout', 1, 0xffffffff)):
        if type(profile[field]) is not int or not low <= profile[field] <= high:
            raise ValueError('Invalid profile field: '+field)
    if profile['timing'] not in ('xpmmini', 'external'):
        raise ValueError('Select xpmmini or external timing explicitly')
    for field in ('pcie_yaml', 'camera_yaml'):
        if profile[field] is not None and not Path(profile[field]).is_file():
            raise ValueError('Missing '+field+': '+profile[field])
    if capture:
        if not profile['camera_yaml']:
            raise ValueError('Capture requires an explicit camera YAML baseline')
        if type(profile['camera_auto_trigger']) is not bool:
            raise ValueError('Specify camera_auto_trigger')
        if type(profile['camera_trigger_source']) is not int or not 0 <= profile['camera_trigger_source'] <= 3:
            raise ValueError('Specify camera_trigger_source')
        rate = profile['rate_hz']
        if type(rate) not in (float, int) or not math.isfinite(rate) or rate <= 0:
            raise ValueError('Specify a positive finite rate_hz')
        sizes = profile['image_payload_bytes']
        if not isinstance(sizes, list) or not sizes or any(type(n) is not int or n < 1081384 or n % 2 for n in sizes):
            raise ValueError('Specify expected full-image payload byte counts')
        if profile['data_vc'] != 0:
            raise ValueError('This recorder supports camera image VC0 only')


def check_identity(profile, identity):
    if identity['board_type'] != profile['board_type']:
        raise ValueError('PCIe board does not match profile: '+str(identity))
    if profile['image_name_contains'] not in identity['image_name']:
        raise ValueError('PCIe image does not match profile: '+str(identity))
    if identity['fpga_version'] < profile['minimum_pcie_version']:
        raise ValueError('PCIe firmware is older than the profile minimum')


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True,
                                   stderr=subprocess.PIPE, timeout=30).strip()


def software_snapshot(checkout, bundle):
    """Save actual pins and tracked diffs, not just a previously copied manifest."""
    repos = []

    def visit(path, expected=None):
        if not (path / '.git').exists():
            raise ValueError('Uninitialized submodule: '+str(path))
        sha = git(path, 'rev-parse', 'HEAD')
        repos.append({'path': str(path.relative_to(bundle)), 'commit': sha,
                      'recorded_commit': expected,
                      'untracked': git(path, 'ls-files', '--others', '--exclude-standard'),
                      'diff': git(path, 'diff', '--binary', '--ignore-submodules=all', 'HEAD')})
        for line in git(path, 'ls-tree', '-r', 'HEAD').splitlines():
            fields, name = line.split('\t', 1)
            mode, _, child = fields.split()
            if mode == '160000':
                visit(path / name, child)

    bundle = Path(bundle).resolve()
    visit(bundle)
    tool_dir = Path(__file__).parent
    return {'lcls2_commit': git(checkout, 'rev-parse', 'HEAD'),
            'lcls2_diff': git(checkout, 'diff', '--binary', '--ignore-submodules=all', 'HEAD'),
            'scripts_sha256': {p.name: file_sha256(p) for p in tool_dir.glob('*.py')},
            'bundle_path': str(bundle), 'repositories': repos,
            'patch_sh_sha256': file_sha256(bundle / 'patch.sh')}
