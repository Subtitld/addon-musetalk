"""Stdlib-only shared code for the MuseTalk add-on.

Imported by BOTH the thin bootstrap (frozen, no ML stack) and the heavy worker
(runs inside the provisioned cache venv). Nothing here may import torch / cv2 /
numpy or anything outside the standard library — that guarantee is what lets
the bootstrap speak the protocol and validate params before a single byte of
the multi-gigabyte ML runtime is downloaded.
"""

import json
import os
import shutil
import sys
import threading

ADDON_ID = 'musetalk'
ADDON_VERSION = '0.2.0'
PROTOCOL_VERSION = 1

# Host stdout can have two writers in the bootstrap: the main thread (progress
# / errors it generates itself) and the thread pumping the worker's frames
# through. Serialise every write so two frames can never interleave into one
# corrupt line.
_STDOUT_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Protocol I/O helpers
# ---------------------------------------------------------------------------
def log(*args):
    print('[musetalk]', *args, file=sys.stderr, flush=True)


def send(frame):
    send_raw(json.dumps(frame, ensure_ascii=False, separators=(',', ':')))


def send_raw(line):
    """Write one already-serialised protocol frame to host stdout."""
    if not line.endswith('\n'):
        line += '\n'
    with _STDOUT_LOCK:
        sys.stdout.write(line)
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Typed errors -> protocol error codes
# ---------------------------------------------------------------------------
class _AddonError(Exception):
    code = 'internal'


class _BadParams(_AddonError):
    code = 'bad_params'


class _DepMissing(_AddonError):
    # An ML dependency (torch/diffusers/musetalk source) or the runtime itself
    # could not be provisioned.
    code = 'internal'


class _GpuUnavailable(_AddonError):
    code = 'gpu_unavailable'


class _ModelMissing(_AddonError):
    code = 'model_missing'


class _NetworkUnavailable(_AddonError):
    code = 'network_unavailable'


class _Oom(_AddonError):
    code = 'oom'


class _Cancelled(_AddonError):
    code = 'internal'


# ---------------------------------------------------------------------------
# Environment helpers shared by bootstrap + worker
# ---------------------------------------------------------------------------
def cache_root():
    root = os.environ.get('SUBTITLD_ADDON_CACHE') or os.path.join(
        os.path.expanduser('~'), '.cache', 'subtitld', 'musetalk')
    os.makedirs(root, exist_ok=True)
    return root


def resolve_ffmpeg():
    """ffmpeg from env (Subtitld sets SUBTITLD_FFMPEG_EXECUTABLE) else PATH."""
    exe = os.environ.get('SUBTITLD_FFMPEG_EXECUTABLE')
    if exe and (os.path.isfile(exe) or shutil.which(exe)):
        return exe
    return shutil.which('ffmpeg')


def validate_params(params):
    """Cheap, ML-free validation of a ``video.lipsync`` request.

    Runs in the bootstrap BEFORE provisioning the multi-GB runtime, so a
    malformed request fails fast with a clean ``bad_params`` error instead of
    triggering a huge download. Raises ``_BadParams`` on any problem; returns
    ``None`` on success. The heavy worker calls this again as defence in depth.
    """
    video_path = params.get('video_path')
    audio_path = params.get('audio_path')
    output_path = params.get('output_path')

    if not video_path or not os.path.isfile(video_path):
        raise _BadParams(f'video_path not found: {video_path!r}')
    if not audio_path or not os.path.isfile(audio_path):
        raise _BadParams(f'audio_path not found: {audio_path!r}')
    if not output_path:
        raise _BadParams('output_path is required')
    if not resolve_ffmpeg():
        raise _BadParams(
            'ffmpeg not found — set SUBTITLD_FFMPEG_EXECUTABLE or install ffmpeg.')
