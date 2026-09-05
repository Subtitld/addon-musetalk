#!/usr/bin/env python3
"""Thin provisioner + protocol proxy for the MuseTalk add-on.

The shipped executable is THIS bootstrap, frozen with only the Python standard
library — no torch, no CUDA — so the release archive is a few MB and fits far
under GitHub's 2 GiB release-asset limit. The heavy ML stack
(torch/diffusers/transformers/cv2/…) is installed at FIRST USE into a private
virtual-env inside the add-on cache, and the real worker
(:mod:`musetalk_addon._worker`) then runs inside that venv as a subprocess.

This is how MuseTalk ships on Linux at all: a fully-frozen CUDA torch bundle is
> 2 GiB, which GitHub refuses as a release asset. Provisioning at runtime keeps
the download small and lets the same architecture shrink every platform.

Flow:

  * Send the hello handshake immediately (stdlib only), so Subtitld's startup
    timeout is satisfied before anything is downloaded.
  * Validate each ``video.lipsync`` request's params up front — a malformed
    request fails fast with ``bad_params`` and never triggers a multi-GB
    install.
  * On the first valid request, provision the runtime venv (reporting progress
    in a small band so the host shows "Setting up…"), start the worker, and
    then transparently proxy every protocol frame in both directions.

Provisioning needs a base Python 3.10+ to build the venv from: it uses
``SUBTITLD_MUSETALK_PYTHON`` if set, else a suitable ``python3.x`` on PATH. The
CUDA torch wheels come from ``SUBTITLD_MUSETALK_TORCH_INDEX`` (default
https://download.pytorch.org/whl/cu121).

KNOWN LIMITATION: :func:`ensure_runtime` runs on the stdin-reading thread, so a
``shutdown`` sent *during* the multi-GB first-run install is not read until the
install finishes. On POSIX the host escalates to ``killpg`` on the add-on's
session, which reaps the pip child too. On Windows the host can only terminate
the bootstrap PID, so a pip grandchild can outlive it and keep downloading. Fix
would be to provision on a background thread and hold a handle to the pip
process to terminate it on shutdown.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import threading

from musetalk_addon._common import (
    ADDON_ID,
    ADDON_VERSION,
    PROTOCOL_VERSION,
    _AddonError,
    _DepMissing,
    _NetworkUnavailable,
    cache_root,
    log,
    send,
    send_raw,
    validate_params,
)

# Where the CUDA torch wheels come from. Overridable so a user on a different
# CUDA version (or an air-gapped mirror) can point elsewhere.
_PYTORCH_INDEX = os.environ.get(
    'SUBTITLD_MUSETALK_TORCH_INDEX', 'https://download.pytorch.org/whl/cu121')

# Candidate interpreter names, newest first. All must be >= 3.10.
_PY_CANDIDATES = ('python3.12', 'python3.11', 'python3.10', 'python3', 'python')


# ---------------------------------------------------------------------------
# Runtime location
# ---------------------------------------------------------------------------
def _runtime_root():
    d = os.path.join(cache_root(), 'runtime')
    os.makedirs(d, exist_ok=True)
    return d


def _venv_python(venv_dir):
    """Path to the venv's interpreter (POSIX ``bin/`` vs Windows ``Scripts/``)."""
    if os.name == 'nt':
        return os.path.join(venv_dir, 'Scripts', 'python.exe')
    return os.path.join(venv_dir, 'bin', 'python')


# ---------------------------------------------------------------------------
# Base-Python discovery
# ---------------------------------------------------------------------------
def _python_ok(exe):
    """True if ``exe`` is a usable venv-capable CPython >= 3.10."""
    if not exe:
        return False
    try:
        out = subprocess.run(
            [exe, '-c',
             'import sys,venv,ensurepip;'
             'print(sys.version_info[0],sys.version_info[1])'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    if out.returncode != 0:
        return False
    try:
        major, minor = (int(x) for x in out.stdout.split())
    except ValueError:
        return False
    return (major, minor) >= (3, 10)


def _find_base_python():
    """Locate a base Python to build the runtime venv from.

    Honours ``SUBTITLD_MUSETALK_PYTHON`` (raising a clear error if it is set but
    unusable), else scans PATH. Returns the interpreter path, or None if PATH
    has nothing suitable.
    """
    override = os.environ.get('SUBTITLD_MUSETALK_PYTHON')
    if override:
        if _python_ok(override):
            return override
        raise _DepMissing(
            f'SUBTITLD_MUSETALK_PYTHON={override!r} is not a usable Python 3.10+ '
            '(needs the venv + ensurepip modules).')
    for name in _PY_CANDIDATES:
        exe = shutil.which(name)
        if exe and _python_ok(exe):
            return exe
    return None


# ---------------------------------------------------------------------------
# The installable — our own package, carrying its ML dependencies
# ---------------------------------------------------------------------------
def _installable_target():
    """Return a pip target that installs ``musetalk_addon`` + its deps.

    Frozen: a wheel bundled beside the executable (``wheels/*.whl``). Dev: the
    repo checkout (``pip install <repo>``). The pip requirement carries the
    torch/diffusers/… dependency set, so installing it provisions everything.
    """
    candidates = []
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        candidates.append(os.path.join(meipass, 'wheels'))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(sys.executable)), 'wheels'))
    for d in candidates:
        hits = sorted(glob.glob(os.path.join(d, 'subtitld_addon_musetalk-*.whl')))
        if hits:
            return hits[-1]
    # Dev fallback: the repo root (two levels up from this file).
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isfile(os.path.join(repo, 'pyproject.toml')):
        return repo
    raise _DepMissing(
        'could not locate the MuseTalk add-on wheel to install; the bundle is '
        'incomplete (expected wheels/subtitld_addon_musetalk-*.whl).')


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------
def _run_streaming(cmd, progress, lo, hi, message):
    """Run ``cmd``, forwarding coarse progress in the [lo, hi] band.

    pip output is noisy and unstructured, so we don't try to parse exact
    percentages — we nudge the bar forward on each "Collecting/Downloading/
    Installing" line and surface the current package name as the message. On a
    non-zero exit the tail of the combined output is raised.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    tail = []
    value = lo
    step = (hi - lo) / 60.0  # ~60 interesting lines spans the band
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)
        head = line.split(' ', 1)[0]
        if head in ('Collecting', 'Downloading', 'Installing', 'Building', 'Using'):
            value = min(hi - 0.01, value + step)
            progress(value, f'{message}: {line[:80]}')
    proc.wait()
    if proc.returncode != 0:
        detail = '\n'.join(tail[-15:])
        raise _classify_pip_failure(proc.returncode, detail)


# Substrings that mark a pip failure as genuinely network-related. Anything
# else (a resolution failure, a source build that won't compile) is a local
# problem and must NOT be reported as `network_unavailable` — that sends the
# user off debugging their connection instead of their toolchain.
_NETWORK_MARKERS = (
    'connection', 'connectionerror', 'timed out', 'timeout', 'temporary failure',
    'name resolution', 'network is unreachable', 'proxy', 'ssl', 'certificate',
    'retries exceeded', 'failed to establish', 'newconnectionerror',
    'read timed out', 'remote end closed', 'econnreset',
)


def _classify_pip_failure(returncode, detail):
    """Map a failed pip run onto the right protocol error.

    ``network_unavailable`` only when the output actually looks like a transport
    problem; everything else is ``internal`` via ``_DepMissing`` (a resolution
    failure or an un-buildable dependency is a local/toolchain issue). The pip
    tail is preserved in the message either way, so the concrete cause is never
    lost.
    """
    blob = (detail or '').lower()
    prefix = 'runtime setup failed (pip exited %d):\n%s' % (returncode, detail)
    if any(marker in blob for marker in _NETWORK_MARKERS):
        return _NetworkUnavailable(prefix)
    return _DepMissing(prefix)


def ensure_runtime(progress):
    """Provision (or reuse) the private ML venv. Returns its interpreter path.

    Idempotent: a ``.provisioned`` marker holding the add-on version short-
    circuits re-installs. Progress is reported across [0, 1] within whatever
    band the caller maps it into.
    """
    root = _runtime_root()
    venv_dir = os.path.join(root, 'venv')
    marker = os.path.join(root, '.provisioned')
    py = _venv_python(venv_dir)

    if os.path.isfile(marker) and os.path.isfile(py):
        try:
            if open(marker, encoding='utf-8').read().strip() == ADDON_VERSION:
                return py
        except OSError:
            pass

    progress(0.02, 'Setting up MuseTalk GPU runtime (first run)…')
    base = _find_base_python()
    if not base:
        raise _DepMissing(
            'MuseTalk needs a Python 3.10+ interpreter to set up its GPU runtime, '
            'and none was found on PATH. Install python3 (3.10 or newer) or set '
            'SUBTITLD_MUSETALK_PYTHON to one.')

    # Fresh venv (remove a half-provisioned one from a previous failure).
    if os.path.isdir(venv_dir) and not (os.path.isfile(marker)):
        shutil.rmtree(venv_dir, ignore_errors=True)
    if not os.path.isfile(py):
        progress(0.05, 'Creating virtual environment…')
        try:
            subprocess.run([base, '-m', 'venv', venv_dir], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            raise _DepMissing(f'failed to create the runtime venv: {exc}')
        py = _venv_python(venv_dir)

    _run_streaming([py, '-m', 'pip', 'install', '--upgrade', 'pip', 'wheel'],
                   progress, 0.05, 0.10, 'Preparing installer')

    target = _installable_target()
    progress(0.10, 'Installing PyTorch + MuseTalk (downloads several GB, once)…')
    _run_streaming(
        [py, '-m', 'pip', 'install', target, '--extra-index-url', _PYTORCH_INDEX],
        progress, 0.10, 0.98, 'Installing runtime')

    with open(marker, 'w', encoding='utf-8') as fh:
        fh.write(ADDON_VERSION)
    progress(1.0, 'Runtime ready')
    return py


# ---------------------------------------------------------------------------
# Worker subprocess + transparent proxy
# ---------------------------------------------------------------------------
def _start_worker(venv_python):
    env = os.environ.copy()
    env['SUBTITLD_MUSETALK_WORKER'] = '1'
    return subprocess.Popen(
        [venv_python, '-m', 'musetalk_addon', '--worker'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
        text=True, bufsize=1, env=env)


class _InFlight:
    """The request ids forwarded to the worker that have no terminal frame yet.

    The worker can die in ways it cannot report itself — a CUDA abort, a
    segfault in a native op, the OS OOM-killer. Those leave the host waiting
    forever on a request that will never be answered, so the bootstrap tracks
    what is outstanding and synthesizes an ``error`` frame for each one when the
    worker's stdout hits EOF with the process gone.
    """

    def __init__(self):
        self._ids = []
        self._lock = threading.Lock()

    def add(self, req_id):
        if req_id is None:
            return
        with self._lock:
            if req_id not in self._ids:
                self._ids.append(req_id)

    def discard(self, req_id):
        with self._lock:
            if req_id in self._ids:
                self._ids.remove(req_id)

    def drain(self):
        with self._lock:
            pending, self._ids = self._ids, []
            return pending


def _pump_worker_stdout(worker, in_flight):
    """Copy the worker's protocol frames through to the host.

    Terminal frames (``result`` / ``error``) clear the request from
    ``in_flight``. If the stream ends while requests are still outstanding, the
    worker died without reporting — emit an error for each so the host is never
    left hanging.
    """
    try:
        for line in worker.stdout:
            line = line.strip()
            if not line:
                continue
            frame = _safe_json(line)
            if frame is not None and frame.get('type') in ('result', 'error'):
                in_flight.discard(frame.get('id'))
            send_raw(line)
    except (ValueError, OSError):
        pass

    # Stream closed. Anything still outstanding will never be answered.
    pending = in_flight.drain()
    if not pending:
        return
    rc = worker.poll()
    detail = 'exit code %s' % ('unknown' if rc is None else rc)
    for req_id in pending:
        send({'id': req_id, 'type': 'error', 'code': 'internal',
              'message': 'the MuseTalk runtime worker exited unexpectedly (%s) — '
                         'this usually means a CUDA/driver crash or the process ran '
                         'out of memory. Try a smaller "batch_size".' % detail})


def run_bootstrap():
    send({
        'type': 'hello',
        'protocol': PROTOCOL_VERSION,
        'addon': ADDON_ID,
        'version': ADDON_VERSION,
        'capabilities': [{'task': 'video.lipsync'}],
    })

    worker = None  # subprocess.Popen, once provisioned + started
    in_flight = _InFlight()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        # Transparent pass-through once the worker is live.
        if worker is not None and worker.poll() is None:
            frame = _safe_json(line)
            ftype = frame.get('type') if frame else None
            if ftype == 'video.lipsync':
                in_flight.add(frame.get('id'))
            try:
                worker.stdin.write(line + '\n')
                worker.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                # The worker died between the poll() above and this write. Do
                # not drop the frame silently — answer the request ourselves.
                if ftype == 'video.lipsync':
                    in_flight.discard(frame.get('id'))
                    send({'id': frame.get('id'), 'type': 'error', 'code': 'internal',
                          'message': 'the MuseTalk runtime worker is no longer running; '
                                     'the request could not be delivered.'})
                worker = None
            if ftype == 'shutdown':
                break
            continue

        frame = _safe_json(line)
        if frame is None:
            continue
        ftype = frame.get('type')

        if ftype == 'ready':
            log('handshake complete')
            continue
        if ftype == 'shutdown':
            break
        if ftype == 'cancel':
            continue  # nothing running yet
        if ftype != 'video.lipsync':
            continue

        req_id = frame.get('id')
        params = frame.get('params') or {}

        def progress(value, message=''):
            send({'id': req_id, 'type': 'progress',
                  'value': max(0.0, min(1.0, round(float(value), 4))),
                  'message': message})

        # 1. Fast param validation — no provisioning for a doomed request.
        try:
            validate_params(params)
        except _AddonError as exc:
            send({'id': req_id, 'type': 'error', 'code': exc.code, 'message': str(exc)})
            continue

        # 2. Provision the runtime (once). Progress lands in the first 5%.
        try:
            venv_python = ensure_runtime(lambda v, m='': progress(0.05 * v, m))
        except _AddonError as exc:
            send({'id': req_id, 'type': 'error', 'code': exc.code, 'message': str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001
            send({'id': req_id, 'type': 'error', 'code': 'internal', 'message': repr(exc)})
            continue

        # 3. Launch the worker and hand off to transparent proxying.
        worker = _start_worker(venv_python)
        in_flight.add(req_id)
        threading.Thread(target=_pump_worker_stdout, args=(worker, in_flight),
                         daemon=True).start()
        try:
            worker.stdin.write(line + '\n')  # forward the triggering request
            worker.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            in_flight.discard(req_id)
            send({'id': req_id, 'type': 'error', 'code': 'internal',
                  'message': 'runtime worker failed to start'})
            worker = None

    # Clean shutdown of the worker if one is running.
    if worker is not None and worker.poll() is None:
        try:
            worker.stdin.write('{"type":"shutdown"}\n')
            worker.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            pass
        try:
            worker.wait(timeout=15)
        except subprocess.TimeoutExpired:
            worker.kill()
    log('exiting')


def _safe_json(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None
