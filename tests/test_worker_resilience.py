"""Regression tests for the bootstrap proxy's failure handling.

Two behaviours are easy to get wrong and impossible to notice until a user is
staring at a frozen progress bar:

  1. If the worker dies *hard* (CUDA abort, segfault, OOM-killer) it cannot
     report anything itself. The bootstrap must notice its stdout closed with a
     request still outstanding and synthesize a terminal ``error`` frame, or the
     host waits forever.
  2. A ``cancel`` sent while a lip-sync is running must actually be observed.
     The worker reads stdin on a background thread precisely so a cancel can
     land mid-job; if it ever goes back to reading on the pipeline thread,
     cancellation silently becomes a no-op.

Both are driven with a fake worker / fake pipeline — no torch, no GPU, no
model download — so they run anywhere.
"""

import json
import queue
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from musetalk_addon import _bootstrap  # noqa: E402
from musetalk_addon import _worker  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Worker dies mid-request -> bootstrap synthesizes an error frame
# ---------------------------------------------------------------------------
def test_worker_hard_crash_yields_error_frame(monkeypatch, capsys):
    """A worker that exits without answering must not leave the host hanging."""
    # A "worker" that emits one progress frame then dies hard, exactly like a
    # CUDA abort: no result, no error, just a closed pipe and a non-zero exit.
    crasher = subprocess.Popen(
        [sys.executable, '-c', textwrap.dedent('''
            import sys
            sys.stdout.write('{"id":"r1","type":"progress","value":0.5,"message":"working"}\\n')
            sys.stdout.flush()
            sys.exit(139)  # simulate a hard death (segfault-ish exit code)
        ''')],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    in_flight = _bootstrap._InFlight()
    in_flight.add('r1')
    _bootstrap._pump_worker_stdout(crasher, in_flight)

    frames = [json.loads(ln) for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    # The worker's own progress frame is relayed...
    assert any(f.get('type') == 'progress' and f.get('id') == 'r1' for f in frames)
    # ...and the request is terminated on its behalf.
    errors = [f for f in frames if f.get('type') == 'error']
    assert len(errors) == 1, f'expected exactly one synthesized error, got {frames}'
    assert errors[0]['id'] == 'r1'
    assert errors[0]['code'] == 'internal'
    assert 'exited unexpectedly' in errors[0]['message']
    # Nothing left outstanding.
    assert in_flight.drain() == []


def test_clean_result_is_not_double_reported(monkeypatch, capsys):
    """A worker that answers properly must NOT also get a synthesized error."""
    good = subprocess.Popen(
        [sys.executable, '-c', textwrap.dedent('''
            import sys
            sys.stdout.write('{"id":"r1","type":"result","data":{"output_path":"/tmp/x.mp4"}}\\n')
            sys.stdout.flush()
        ''')],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    in_flight = _bootstrap._InFlight()
    in_flight.add('r1')
    _bootstrap._pump_worker_stdout(good, in_flight)

    frames = [json.loads(ln) for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert [f['type'] for f in frames] == ['result']
    assert in_flight.drain() == []


# ---------------------------------------------------------------------------
# 2. Cancel arrives *while* the pipeline runs
# ---------------------------------------------------------------------------
def test_cancel_is_observed_during_a_running_job(monkeypatch):
    """The pipeline must see a cancel that is sent after the job has started."""
    started = threading.Event()
    observed = queue.Queue()

    def fake_run_lipsync(params, progress, is_cancelled):
        started.set()
        # Poll like the real UNet loop does between batches.
        for _ in range(200):  # ~2s worst case
            if is_cancelled():
                observed.put(True)
                raise _worker._Cancelled('cancelled by host')
            time.sleep(0.01)
        observed.put(False)
        return '/tmp/never.mp4'

    monkeypatch.setattr(_worker, 'run_lipsync', fake_run_lipsync)

    # Feed the worker a request, then a cancel *after* it is already running.
    r, w = __import__('os').pipe()
    reader = __import__('os').fdopen(r, 'r')
    writer = __import__('os').fdopen(w, 'w')
    monkeypatch.setattr(sys, 'stdin', reader)

    sent = []
    monkeypatch.setattr(_worker, 'send', lambda f: sent.append(f))

    loop = threading.Thread(target=_worker.run_worker_loop,
                            kwargs={'emit_hello': False}, daemon=True)
    loop.start()

    writer.write(json.dumps({'id': 'j1', 'type': 'video.lipsync',
                             'params': {}}) + '\n')
    writer.flush()
    assert started.wait(timeout=5), 'job never started'
    # The job is running NOW — this is the frame that used to sit unread.
    writer.write(json.dumps({'type': 'cancel', 'target': 'j1'}) + '\n')
    writer.flush()

    was_cancelled = observed.get(timeout=5)
    assert was_cancelled, 'cancel sent mid-job was never observed by the pipeline'

    writer.write(json.dumps({'type': 'shutdown'}) + '\n')
    writer.flush()
    loop.join(timeout=5)

    # The cancellation surfaces as a terminal error frame for that request.
    terminal = [f for f in sent if f.get('type') in ('result', 'error')]
    assert terminal and terminal[-1]['id'] == 'j1'
    assert terminal[-1]['type'] == 'error'


# ---------------------------------------------------------------------------
# 3. pip failures are classified, not blanket-blamed on the network
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('detail, expected_code', [
    ('ERROR: Could not install packages due to an OSError: '
     'HTTPSConnectionPool(host=\'pypi.org\', port=443): Read timed out.', 'network_unavailable'),
    ('ERROR: Failed building wheel for llvmlite\n'
     'error: command \'gcc\' failed with exit status 1', 'internal'),
    ('ERROR: Could not find a version that satisfies the requirement torch==2.9',
     'internal'),
])
def test_pip_failure_classification(detail, expected_code):
    exc = _bootstrap._classify_pip_failure(1, detail)
    assert exc.code == expected_code
    assert detail.splitlines()[0] in str(exc)  # the real cause is preserved
