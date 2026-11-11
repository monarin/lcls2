"""Read only the owner's cached PiranhaHoldDiagnostics JSON on a trusted server.

    python inspect_hold.py --host 127.0.0.1 --port BASE --path ROOT.PiranhaHoldDiagnostics

Use the pinned pilot Python. SimpleClient uses pickle, so connect only to the
trusted existing root. No get(), ReadAll, VirtualClient, root/device constructor,
UART operation or subscription is used. The subprocess deadline also bounds
library import, DNS/connection setup and a stalled pinned SimpleClient.value().
"""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

from evidence import DIAGNOSTIC_NAME, MAX_DIAGNOSTIC_BYTES


RESULT_PREFIX = 'PIRANHA_CACHED_JSON='
MAX_TIMEOUT = 60.0


def _validate(host, port, path, timeout):
    if not isinstance(host, str) or not host or len(host) > 253 or any(char.isspace() for char in host):
        raise ValueError('A bounded server host is required')
    if type(port) is not int or not 1 <= port <= 65533:
        raise ValueError('Port must be the advertised base port (1..65533)')
    if not math.isfinite(timeout) or not 0 < timeout <= MAX_TIMEOUT:
        raise ValueError('Timeout must be finite and in (0, 60] seconds')
    if (not isinstance(path, str) or len(path) > 512 or not re.fullmatch(
            r'[A-Za-z_][A-Za-z0-9_]*\.' + re.escape(DIAGNOSTIC_NAME), path)):
        raise ValueError('Only the exact ROOT.' + DIAGNOSTIC_NAME + ' cached node is allowed')


def _cached_value(host, port, path, timeout):
    """Pinned SimpleClient adapter; socket timeouts supplement the outer deadline."""
    import pyrogue.interfaces
    import zmq

    client = pyrogue.interfaces.SimpleClient(addr=host, port=port)
    try:
        client._req.setsockopt(zmq.LINGER, 0)
        client._req.setsockopt(zmq.SNDTIMEO, max(1, int(timeout * 1000)))
        client._req.setsockopt(zmq.RCVTIMEO, max(1, int(timeout * 1000)))
        client._req.setsockopt(zmq.MAXMSGSIZE, MAX_DIAGNOSTIC_BYTES + 4096)
        value = client.value(path)
        if not isinstance(value, str) or len(value) > MAX_DIAGNOSTIC_BYTES:
            raise ValueError('Cached diagnostic JSON is missing or oversized')
        result = json.loads(value)
        if not isinstance(result, dict):
            raise ValueError('Cached diagnostic JSON must be an object')
        return result
    finally:
        client._stop()
        client._req.close(linger=0)
        client._ctx.term()


def _worker():
    request = json.loads(sys.stdin.read(4096))
    try:
        result = _cached_value(**request)
        message = {'value': result}
    except Exception as error:
        message = {'error': type(error).__name__ + ': ' + str(error)[:1024]}
    print(RESULT_PREFIX + json.dumps(message, ensure_ascii=True), flush=True)


def read_snapshot(host, port, path, timeout=5.0):
    """Return decoded cached JSON or raise; no network work escapes the deadline."""
    _validate(host, port, path, timeout)
    deadline = time.monotonic() + timeout
    command = [sys.executable, '-c',
               'import sys; sys.path.insert(0, sys.argv[1]); from inspect_hold import _worker; _worker()',
               str(Path(__file__).resolve().parent)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    request = json.dumps(dict(host=host, port=port, path=path, timeout=timeout))
    try:
        output, errors = process.communicate(request, timeout=max(0.001, deadline - time.monotonic()))
    except BaseException:
        process.kill()
        try:
            process.wait(timeout=0.25)
        finally:
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()
        raise
    if process.returncode:
        raise RuntimeError('Inspection client failed: ' + errors[-1024:])
    for line in output.splitlines():
        if line.startswith(RESULT_PREFIX):
            if len(line) > MAX_DIAGNOSTIC_BYTES + 4096:
                raise ValueError('Inspection response exceeds bounded size')
            result = json.loads(line[len(RESULT_PREFIX):])
            if 'error' in result:
                raise RuntimeError(result['error'])
            return result['value']
    raise RuntimeError('Inspection client returned no cached JSON')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--path', required=True)
    parser.add_argument('--timeout', type=float, default=5.0)
    args = parser.parse_args(argv)
    try:
        result = read_snapshot(args.host, args.port, args.path, args.timeout)
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print('Inspection failed: ' + str(error)[:1024], file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
