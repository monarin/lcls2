"""Parse the diagnostic launch with installed daqmgr; never submit jobs."""
import json
import os
from pathlib import Path
import shlex
import shutil
import tempfile
from unittest.mock import patch

from psdaq.slurm.main import Runner


def main():
    expected = {'xpmpva', 'groupca', 'daqstat', 'control', 'control_gui',
                'teb0', 'timing_0', 'epixquad1kfps_0'}
    # Runner creates a log directory during construction. Keep it temporary.
    # These explicit-host configs do not query Slurm during parsing.
    with tempfile.TemporaryDirectory(prefix='ued-launch-check-') as output:
        runner = Runner(os.environ['UED_GAINMAP_CNF'], output=output)
        assert set(runner.config) == expected
        control = shlex.split(runner.config['control']['cmd'])
        assert control[control.index('-C') + 1] == 'TEST'
        for name, row in runner.config.items():
            if name == 'epixquad1kfps_0':
                tokens = shlex.split(row['cmd'])
                assert tokens[0] == 'drp'
                assert shutil.which('drp') == str(Path(os.environ['TESTRELDIR'])/'bin/drp')
                diagnostic = ('epixDiagDir=' + os.environ['UED_GAINMAP_DIAG_DIR']
                              + ',epixDiagFrames=51')
                assert tokens.count(diagnostic) == 1
                assert tokens[tokens.index(diagnostic) - 1] == '-k'
                assert row['host'] == 'drp-ued-cmp003'
            else:
                assert 'epixDiag' not in row['cmd']
        # Simulate the worker environment: custom shell variables are absent
        # unless explicitly supplied in the process row's env field.
        worker_env = {k: v for k, v in os.environ.items()
                      if not k.startswith('UED_GAINMAP_')}
        for assignment in shlex.split(runner.config['daqstat']['env']):
            key, value = assignment.split('=', 1)
            worker_env[key] = value
        with patch.dict(os.environ, worker_env, clear=True):
            worker_runner = Runner(runner.configfilename, output=output)
        assert worker_runner.config == runner.config
        print(json.dumps({'status': 'passed', 'config': runner.configfilename,
                          'processes': sorted(runner.config),
                          'platform': runner.platform, 'xpm': runner.xpm_id,
                          'alias': 'TEST', 'diagnostic_frames': 51,
                          'daqstat_filtered_environment': 'passed',
                          'scope': 'Runner construction only; no DAQ start or submission'},
                         indent=2))


if __name__ == '__main__':
    main()
