from __future__ import annotations

import os
import signal
import subprocess
import time


def process_alive(proc: subprocess.Popen) -> bool:
    """Return True while a subprocess is still running."""

    return proc.poll() is None


def terminate_process_group(
    proc: subprocess.Popen,
    term_timeout_seconds: float = 15.0,
    kill_timeout_seconds: float = 5.0,
) -> None:
    """Terminate a subprocess group.

    SceneSmith may spawn child processes. Starting the subprocess with
    start_new_session=True creates a separate process group. This function
    sends SIGINT, then SIGTERM, then SIGKILL if needed.
    """

    if not process_alive(proc):
        return

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return

    for sig, timeout in [
        (signal.SIGINT, term_timeout_seconds),
        (signal.SIGTERM, kill_timeout_seconds),
    ]:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return

        start = time.time()
        while time.time() - start < timeout:
            if not process_alive(proc):
                return
            time.sleep(0.2)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
