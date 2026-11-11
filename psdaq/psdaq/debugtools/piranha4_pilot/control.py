"""Run-scoped inspection control and an independent worker watchdog."""
import json
import math
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time


class Channel:
    def __init__(self, connection, campaign):
        self.connection = connection
        self.connection.setblocking(False)
        self.campaign = campaign
        self.buffer = b''

    def send(self, kind, trial, **fields):
        message = dict(fields, kind=kind, trial=trial, campaign=self.campaign)
        data = (json.dumps(message, allow_nan=False) + '\n').encode()
        if len(data) > 65536:
            raise ValueError('Oversized control message')
        deadline = time.monotonic() + 1.
        while data:
            if time.monotonic() >= deadline:
                raise TimeoutError('Control send deadline')
            if select.select([], [self.connection], [], .05)[1]:
                try:
                    data = data[self.connection.send(data):]
                except BlockingIOError:
                    pass

    def receive(self, timeout=0.):
        if b'\n' not in self.buffer:
            if not select.select([self.connection], [], [], max(0., timeout))[0]:
                return None
            chunk = self.connection.recv(65536)
            if not chunk:
                raise EOFError('Parent/child control channel closed')
            self.buffer += chunk
        if len(self.buffer) > 65536:
            raise ValueError('Oversized control buffer')
        if b'\n' not in self.buffer:
            return None
        raw, self.buffer = self.buffer.split(b'\n', 1)
        message = json.loads(raw)
        if (not isinstance(message, dict) or message.get('campaign') != self.campaign
                or type(message.get('trial')) is not int or not isinstance(message.get('kind'), str)):
            raise ValueError('Malformed or foreign control message')
        return message

    def wait(self, trial, kinds, deadline):
        while time.monotonic() < deadline:
            message = self.receive(min(.05, deadline - time.monotonic()))
            if message is not None:
                if message['trial'] != trial or message['kind'] not in kinds:
                    raise ValueError('Unexpected/stale control decision: ' + repr(message))
                return message
        raise TimeoutError('Control decision deadline expired')

    def close(self):
        self.connection.close()


class CleanupGuard:
    """Independent interpreter: bounds cleanup even if native code holds the GIL."""
    def __init__(self):
        read_fd, self.write_fd = os.pipe()
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), '--guard',
                 str(read_fd), str(os.getpid())], pass_fds=(read_fd,),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True)
        except BaseException:
            os.close(self.write_fd)
            raise
        finally:
            os.close(read_fd)

    def arm(self, deadline):
        if not math.isfinite(deadline):
            raise ValueError('Nonfinite watchdog deadline')
        os.write(self.write_fd, ('%.9f\n' % deadline).encode())

    def close(self):
        os.close(self.write_fd)
        try:
            self.process.wait(timeout=1.)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=1.)


def guard_main(read_fd, parent_pid):
    deadline = None
    pending = b''
    while True:
        remaining = None if deadline is None else max(0., deadline - time.monotonic())
        if not select.select([read_fd], [], [], remaining)[0]:
            if os.getppid() == parent_pid:
                os.kill(parent_pid, signal.SIGKILL)
            return
        chunk = os.read(read_fd, 4096)
        if not chunk:
            return
        pending += chunk
        while b'\n' in pending:
            raw, pending = pending.split(b'\n', 1)
            deadline = float(raw)
            if not math.isfinite(deadline):
                return


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--guard':
        raise SystemExit('Internal cleanup watchdog only')
    guard_main(int(sys.argv[2]), int(sys.argv[3]))
