"""Load checkout Python with existing SDF native extensions, then run pytest."""
import importlib.util
import os
from pathlib import Path
import sys

repo = Path(__file__).resolve().parents[2]
source = repo / 'psana/psana'
installed = Path('/sdf/home/m/monarin/lcls2_worktree/psana2-gpu/install/lib/python3.9/site-packages')
sys.path.insert(0, str(installed))
spec = importlib.util.spec_from_file_location('psana', source/'__init__.py',
    submodule_search_locations=[str(source),str(installed/'psana')])
module = importlib.util.module_from_spec(spec)
sys.modules['psana'] = module
spec.loader.exec_module(module)
import psana.gpu.gpu_input_group as groups
assert Path(groups.__file__).resolve() == source/'gpu/gpu_input_group.py'
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
