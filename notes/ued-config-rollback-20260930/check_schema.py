"""Exercise schema-only functions from actual old/new source, without hardware."""
import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from psdaq.configdb.epixquad_layout import RAW_SHAPE, RAW_ASIC_LAYOUT

folder = Path(__file__).resolve().parent
checkout = folder.parents[1]
relative = 'psdaq/psdaq/configdb/epixquad1kfps_config.py'
current = checkout / relative
deployed = Path('/sdf/group/lcls/ds/ana/sw/conda2/rel/ued/lcls2_091026/install/lib/python3.9/site-packages/psdaq/configdb/epixquad1kfps_config.py')
legacy_revision = subprocess.check_output(
    ['git', '-C', str(checkout), 'rev-parse', '8e6f55eb2^'], text=True).strip()
legacy = subprocess.check_output(
    ['git', '-C', str(checkout), 'show', legacy_revision + ':' + relative], text=True)
cfg = json.loads(gzip.decompress((folder / 'beam-current.json.gz').read_bytes()))


def functions(source, names, namespace):
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<schema-only>', 'exec'), namespace)
    return namespace


old = functions(legacy, ['_hydrate_map_mode_partial_config'], {'ocfg': cfg})
map_cfg = copy.deepcopy(cfg)
map_cfg['user']['gain_mode'] = 5
try:
    old['_hydrate_map_mode_partial_config'](map_cfg)
except KeyError as exc:
    old_error = str(exc)
else:
    raise AssertionError('Legacy map-mode path unexpectedly accepted missing pixel_map')

# Exercise the actual legacy fixed-gain expression which accesses ocfg directly.
tree = ast.parse(legacy)
user_to_expert = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'user_to_expert')
old_assignment = next(n for n in ast.walk(user_to_expert)
                      if isinstance(n, ast.Assign) and "ocfg['user']['pixel_map']" in ast.unparse(n))
try:
    exec(compile(ast.Module(body=[old_assignment], type_ignores=[]), '<legacy-fixed-gain>', 'exec'),
         {'np': np, 'ocfg': cfg, 'mapv': 12})
except KeyError as exc:
    old_fixed_error = str(exc)
else:
    raise AssertionError('Legacy fixed-gain path unexpectedly accepted missing pixel_map')

names = ['_raw_pixel_map_shape', '_normalize_raw_pixel_map', '_get_cfg_pixel_map_raw',
         '_get_cfg_trbit_by_asic', '_trbits_asic_to_panel_order',
         '_cbits_config_from_raw_map', '_analysis_config_from_cfg']
outputs = []
sources = []
for path in (current, deployed):
    source = path.read_text()
    ns = functions(source, names, {
        'np': np, 'RAW_SHAPE': RAW_SHAPE,
        'PANEL_ASIC_LAYOUT': tuple(sorted(RAW_ASIC_LAYOUT,
                                         key=lambda x: (x['row_slice'][0], x['col_slice'][0]))),
    })
    result = ns['_analysis_config_from_cfg'](cfg)
    assert [x.shape for x in result] == [(4, 352, 384), (4, 4), (4, 352, 384)]
    outputs.append(result)
    sources.append({'path': str(path), 'sha256': hashlib.sha256(source.encode()).hexdigest()})
for actual, expected in zip(outputs[0], outputs[1]):
    np.testing.assert_array_equal(actual, expected)
result = {'scope': 'Schema functions only; no hardware imports, initialization, or database writes',
          'legacy_revision': legacy_revision, 'legacy_map_error': old_error,
          'legacy_fixed_gain_error': old_fixed_error,
          'current_and_deployed_schema_outputs_equal': True, 'sources': sources,
          'shapes': [list(x.shape) for x in outputs[0]]}
(folder / 'offline-check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
