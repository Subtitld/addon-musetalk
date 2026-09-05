"""Protocol smoke test — spawns the add-on and drives the JSON-line loop.

Exercises the THIN bootstrap (the shipped executable): the hello handshake, the
fast ``bad_params`` path (params are validated before any provisioning), and a
clean provisioning-failure error. None of these need torch, CUDA, a venv build,
or a model download — the test runs offline in CI with **none of the ML stack
installed**, and we force provisioning to fail instantly by pointing the base
interpreter at a bogus path. A real lip-sync test needs an NVIDIA GPU + several
GB of weights and is left to manual runs.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Provisioning is forced to fail fast; the resulting error is a clean protocol
# frame rather than a crash. `_DepMissing` maps to the `internal` code.
_ERROR_CODES = {'gpu_unavailable', 'model_missing', 'network_unavailable', 'internal', 'oom'}


def _spawn():
    env = dict(os.environ)
    # Bogus base interpreter -> ensure_runtime() raises before any download.
    env['SUBTITLD_MUSETALK_PYTHON'] = str(ROOT / 'no-such-python')
    # Make param validation deterministic without a real ffmpeg on PATH.
    env['SUBTITLD_FFMPEG_EXECUTABLE'] = sys.executable
    return subprocess.Popen(
        [sys.executable, '-m', 'musetalk_addon'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        cwd=str(ROOT), env=env,
    )


def _send(proc, frame):
    proc.stdin.write(json.dumps(frame) + '\n')
    proc.stdin.flush()


def _recv(proc):
    return json.loads(proc.stdout.readline())


def test_protocol_loop(tmp_path):
    proc = _spawn()
    try:
        # --- handshake ---
        hello = _recv(proc)
        assert hello['type'] == 'hello'
        assert hello['protocol'] == 1
        assert hello['addon'] == 'musetalk'
        assert hello['version'] == '0.2.0'
        assert any(c.get('task') == 'video.lipsync' for c in hello['capabilities'])

        _send(proc, {'type': 'ready', 'protocol': 1})

        # --- missing video_path -> bad_params, no provisioning ---
        _send(proc, {'id': 'a', 'type': 'video.lipsync',
                     'params': {'audio_path': str(tmp_path / 'a.wav'),
                                'output_path': str(tmp_path / 'out.mp4')}})
        r = _recv(proc)
        assert r['id'] == 'a' and r['type'] == 'error'
        assert r['code'] == 'bad_params'

        # --- valid params but no provisionable runtime -> clean error, loop alive ---
        video = tmp_path / 'in.mp4'
        audio = tmp_path / 'in.wav'
        video.write_bytes(b'\x00\x00\x00\x18ftypmp42')  # non-empty stub
        audio.write_bytes(b'RIFF\x00\x00\x00\x00WAVE')
        _send(proc, {'id': 'b', 'type': 'video.lipsync',
                     'params': {'video_path': str(video), 'audio_path': str(audio),
                                'output_path': str(tmp_path / 'out.mp4')}})
        # A "Setting up…" progress frame may precede the error; find the terminal one.
        while True:
            r = _recv(proc)
            if r.get('type') == 'progress':
                assert r['id'] == 'b'
                assert 0.0 <= r['value'] <= 1.0
                continue
            break
        assert r['id'] == 'b' and r['type'] == 'error'
        assert r['code'] in _ERROR_CODES

        # --- shutdown ---
        _send(proc, {'type': 'shutdown'})
        proc.stdin.close()
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
