"""Explicit readout-clock selection and read-only rate-test diagnostics."""


def configure_clock(camera, period, write=False):
    """Call while quiescent, after baseline loading; verify before triggering."""
    if write:
        camera.AcqCore.AsicRoClkT.set(0xAAAA0000 | period)
    actual = {
        'full_period_ticks': int(camera.AcqCore.AsicRoClkT.get(read=True)),
        'half_period_ticks': int(camera.AcqCore.AsicRoClkHalfT.get(read=True)),
        'adc_pipeline_delay': int(camera.RdoutCore.AdcPipelineDelay.get(read=True)),
    }
    if actual['full_period_ticks'] != period or actual['half_period_ticks'] != period // 2:
        raise RuntimeError('Readout clock did not match request: %r' % actual)
    return actual


def snapshot(pcie, camera, lane):
    """Unavailable counters remain explicit errors, never fabricated zeros."""
    result = {}

    def read(root, path):
        try:
            node = root
            for part in path.split('.'):
                node = getattr(node, part)
            return int(node.get(read=True))
        except Exception as exc:
            return {'error': str(exc)}

    camera_paths = ['AcqCore.AcqCount', 'RdoutCore.SeqCount', 'RdoutCore.BuffersRdy',
                    'SystemRegs.AdcTestDone', 'SystemRegs.AdcTestFailed',
                    'SystemRegs.TrigPeriodMin', 'SystemRegs.TrigPeriodMax']
    camera_paths += ['RdoutCore.LineBufErr[%d]' % i for i in range(4)]
    result['camera'] = {path: read(camera, path) for path in camera_paths}
    prefix = 'DevPcie.Application.AppLane[%d].EventBuilder.' % lane
    paths = ['Timeout', 'Bypass', 'Blowoff', 'TransactionCnt']
    paths += ['%s[%d]' % (name, i) for name in ('DataCnt', 'NullCnt', 'TimeoutDropCnt')
              for i in range(3)]
    result['event_builder'] = {path: read(pcie, prefix + path) for path in paths}
    return result
