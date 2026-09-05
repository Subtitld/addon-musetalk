#!/usr/bin/env python3
"""MuseTalk lip-sync add-on for Subtitld (task: ``video.lipsync``).

Speaks Subtitld's JSON-line add-on protocol on stdin/stdout. Lip-sync is done
with **MuseTalk** (https://github.com/TMElyralab/MuseTalk, MIT-licensed) — a
generative, audio-conditioned mouth-inpainting pipeline built on Stable
Diffusion's VAE + a UNet + Whisper audio features. It is GPU/CUDA only and
needs several GB of model weights, downloaded once on first use and cached
locally under ``~/.cache/subtitld/musetalk/`` (override with
``SUBTITLD_ADDON_CACHE``).

Wire protocol (one JSON object per line):
  add-on -> host : {"type":"hello","protocol":1,"addon":..,"version":..,"capabilities":[..]}
  host -> add-on : {"type":"ready",..}
  host -> add-on : {"id":..,"type":"video.lipsync","params":{"video_path","audio_path","output_path","options"}}
  add-on -> host : {"id":..,"type":"progress","value":<0..1>,"message":..}
  add-on -> host : {"id":..,"type":"result","data":{"output_path": "<path>"}}
  add-on -> host : {"id":..,"type":"error","code":..,"message":..}
  host -> add-on : {"id":..,"type":"cancel","target":"<rid>"}
  host -> add-on : {"type":"shutdown"}

stderr is free-form logging; stdout carries only protocol frames.

IMPORTANT: every heavy dependency (torch, diffusers, transformers, cv2, and the
``musetalk`` package itself) is imported LAZILY inside :func:`run_lipsync`, so
the hello/handshake/error protocol loop runs — and the smoke tests pass —
without any of the ML stack installed.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

ADDON_ID = 'musetalk'
ADDON_VERSION = '0.1.0'
PROTOCOL_VERSION = 1

# The model host / HuggingFace occasionally 403 the default urllib User-Agent,
# so send a browser-like one on the urllib download fallback path.
_USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Subtitld-Addon'

# MuseTalk's source lives in a git repo (it is not a PyPI package). If the
# ``musetalk`` package is not importable we shallow-clone it into the cache.
_MUSETALK_GIT = 'https://github.com/TMElyralab/MuseTalk'

# ---------------------------------------------------------------------------
# Model manifest — mirrors MuseTalk's ``download_weights.sh`` / models/ tree:
#
#   models/
#   ├── musetalk/         musetalk.json, pytorch_model.bin      (MuseTalk 1.0)
#   ├── musetalkV15/      musetalk.json, unet.pth               (MuseTalk 1.5)
#   ├── sd-vae/           config.json, diffusion_pytorch_model.bin
#   ├── whisper/          config.json, pytorch_model.bin, preprocessor_config.json
#   ├── dwpose/           dw-ll_ucoco_384.pth
#   ├── syncnet/          latentsync_syncnet.pt                 (eval only)
#   └── face-parse-bisent/ 79999_iter.pth, resnet18-5c106cde.pth
#
# Each entry downloads a set of files. HuggingFace repos are fetched via
# huggingface_hub if available (best: resumable, hashed), else via a plain
# ``resolve/main`` URL with urllib. ``local_dir`` is the sub-directory of
# ``models/`` the files land in ("." keeps the repo's own subpaths).
# ---------------------------------------------------------------------------
_MODEL_SPECS = [
    {'repo': 'TMElyralab/MuseTalk', 'local_dir': '.',
     'files': ['musetalkV15/musetalk.json', 'musetalkV15/unet.pth',
               'musetalk/musetalk.json', 'musetalk/pytorch_model.bin']},
    {'repo': 'stabilityai/sd-vae-ft-mse', 'local_dir': 'sd-vae',
     'files': ['config.json', 'diffusion_pytorch_model.bin']},
    {'repo': 'openai/whisper-tiny', 'local_dir': 'whisper',
     'files': ['config.json', 'pytorch_model.bin', 'preprocessor_config.json']},
    {'repo': 'yzd-v/DWPose', 'local_dir': 'dwpose',
     'files': ['dw-ll_ucoco_384.pth']},
    # SyncNet is used for evaluation, not the forward inference path; fetch it
    # too so the models/ tree matches MuseTalk exactly. Marked optional so a
    # download failure here does not abort inference.
    {'repo': 'ByteDance/LatentSync', 'local_dir': 'syncnet', 'optional': True,
     'files': ['latentsync_syncnet.pt']},
    # ResNet18 backbone for the BiSeNet face-parser (plain HTTP).
    {'url': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
     'dest': 'face-parse-bisent/resnet18-5c106cde.pth'},
    # BiSeNet face-parsing weights live on Google Drive; needs gdown or a
    # manual drop. Marked optional so a headless box without gdown still gets
    # a clear, actionable error rather than crashing mid-download.
    {'gdrive_id': '154JgKpzCPW82qINcVieuPH3fZ2e0P812',
     'dest': 'face-parse-bisent/79999_iter.pth'},
]


# ---------------------------------------------------------------------------
# Protocol I/O helpers
# ---------------------------------------------------------------------------
def log(*args):
    print('[musetalk]', *args, file=sys.stderr, flush=True)


def send(frame):
    sys.stdout.write(json.dumps(frame, ensure_ascii=False, separators=(',', ':')) + '\n')
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Typed errors -> protocol error codes
# ---------------------------------------------------------------------------
class _AddonError(Exception):
    code = 'internal'


class _BadParams(_AddonError):
    code = 'bad_params'


class _DepMissing(_AddonError):
    # An ML dependency (torch/diffusers/musetalk source) is not installed.
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
# Filesystem / environment helpers
# ---------------------------------------------------------------------------
def _cache_root():
    root = os.environ.get('SUBTITLD_ADDON_CACHE') or os.path.join(
        os.path.expanduser('~'), '.cache', 'subtitld', 'musetalk')
    os.makedirs(root, exist_ok=True)
    return root


def _models_dir():
    d = os.path.join(_cache_root(), 'models')
    os.makedirs(d, exist_ok=True)
    return d


def _resolve_ffmpeg():
    """ffmpeg from env (Subtitld sets SUBTITLD_FFMPEG_EXECUTABLE) else PATH."""
    exe = os.environ.get('SUBTITLD_FFMPEG_EXECUTABLE')
    if exe and (os.path.isfile(exe) or shutil.which(exe)):
        return exe
    return shutil.which('ffmpeg')


def _http_download(url, dest, timeout=600):
    """Stream ``url`` to ``dest`` with a browser User-Agent."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': _USER_AGENT})
    tmp = dest + '.part'
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, 'wb') as fh:
            shutil.copyfileobj(resp, fh, length=1 << 20)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise _NetworkUnavailable(f'download failed for {url}: {exc}')
    os.replace(tmp, dest)


# ---------------------------------------------------------------------------
# Model + source acquisition
# ---------------------------------------------------------------------------
def _iter_model_targets():
    """Flatten ``_MODEL_SPECS`` into (kind, meta, local_relpath) work items."""
    for spec in _MODEL_SPECS:
        if 'repo' in spec:
            for f in spec['files']:
                local = f if spec['local_dir'] == '.' else os.path.join(spec['local_dir'], f)
                yield ('hf', {'repo': spec['repo'], 'file': f,
                              'local_dir': spec['local_dir'],
                              'optional': spec.get('optional', False)}, local)
        elif 'url' in spec:
            yield ('url', {'url': spec['url']}, spec['dest'])
        elif 'gdrive_id' in spec:
            yield ('gdrive', {'id': spec['gdrive_id']}, spec['dest'])


def _ensure_models(progress, lo=0.05, hi=0.40):
    """Download MuseTalk's weights into ``models/`` once. Emits progress in
    the [lo, hi] band. Raises ``_NetworkUnavailable`` / ``_ModelMissing``."""
    models_dir = _models_dir()
    try:
        from huggingface_hub import hf_hub_download
        have_hf = True
    except ImportError:
        have_hf = False

    targets = list(_iter_model_targets())
    total = len(targets)
    for i, (kind, meta, local) in enumerate(targets):
        dest = os.path.join(models_dir, local)
        frac = lo + (hi - lo) * (i / max(1, total))
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            continue
        progress(frac, f'Downloading model: {local}')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            if kind == 'hf':
                if have_hf:
                    target_dir = models_dir if meta['local_dir'] == '.' \
                        else os.path.join(models_dir, meta['local_dir'])
                    hf_hub_download(repo_id=meta['repo'], filename=meta['file'],
                                    local_dir=target_dir)
                else:
                    url = f"https://huggingface.co/{meta['repo']}/resolve/main/{meta['file']}"
                    _http_download(url, dest)
            elif kind == 'url':
                _http_download(meta['url'], dest)
            elif kind == 'gdrive':
                try:
                    import gdown
                except ImportError:
                    raise _ModelMissing(
                        'the BiSeNet face-parsing weights (79999_iter.pth) require '
                        '`gdown` or a manual download from Google Drive into '
                        f'{dest} — see the README.')
                gdown.download(id=meta['id'], output=dest, quiet=True)
        except _AddonError:
            if meta.get('optional'):
                log(f'optional model {local} failed to download; continuing')
                continue
            raise
        if not (os.path.isfile(dest) and os.path.getsize(dest) > 0) and not meta.get('optional'):
            raise _ModelMissing(f'model file missing after download: {local}')
    progress(hi, 'Models ready')


def _ensure_musetalk_source():
    """Make the ``musetalk`` package importable. Returns the cache root that
    contains ``models/`` (used as cwd so MuseTalk's relative ``./models``
    paths resolve). Clones the MuseTalk repo on first use if needed."""
    cache_root = _cache_root()
    try:
        import musetalk  # noqa: F401
        return cache_root
    except ImportError:
        pass

    repo_dir = os.path.join(cache_root, 'MuseTalk')
    if not os.path.isdir(os.path.join(repo_dir, 'musetalk')):
        git = shutil.which('git')
        if not git:
            raise _ModelMissing(
                'the MuseTalk source package is not installed and `git` is not '
                f'available to fetch it. Install MuseTalk or place its source at {repo_dir}.')
        log('cloning MuseTalk source (shallow)...')
        try:
            subprocess.run([git, 'clone', '--depth', '1', _MUSETALK_GIT, repo_dir],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:
            raise _NetworkUnavailable(f'failed to clone MuseTalk source: {exc}')
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    return cache_root


# ---------------------------------------------------------------------------
# The lip-sync pipeline (faithful to MuseTalk's scripts/inference.py)
# ---------------------------------------------------------------------------
def run_lipsync(params, progress, is_cancelled):
    """Generate a lip-synced video. Returns the output path on success.

    All heavy imports happen inside this function so the protocol loop can run
    without the ML stack. The pipeline mirrors MuseTalk's official
    ``scripts/inference.py`` (main branch):

      audio -> Whisper features -> per-frame face crop -> VAE encode ->
      UNet (audio-conditioned) -> VAE decode -> BiSeNet blend back -> ffmpeg mux
    """
    # ---- 1. Validate params (no heavy imports) --------------------------
    video_path = params.get('video_path')
    audio_path = params.get('audio_path')
    output_path = params.get('output_path')
    options = params.get('options') or {}

    if not video_path or not os.path.isfile(video_path):
        raise _BadParams(f'video_path not found: {video_path!r}')
    if not audio_path or not os.path.isfile(audio_path):
        raise _BadParams(f'audio_path not found: {audio_path!r}')
    if not output_path:
        raise _BadParams('output_path is required')

    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg:
        raise _BadParams(
            'ffmpeg not found — set SUBTITLD_FFMPEG_EXECUTABLE or install ffmpeg.')

    version = str(options.get('version') or 'v15').lower()
    version = 'v15' if version in ('v15', 'v1.5', '1.5') else 'v1'
    bbox_shift = int(options.get('bbox_shift', 0))
    extra_margin = int(options.get('extra_margin', 10))
    batch_size = max(1, int(options.get('batch_size', 8)))
    left_pad = int(options.get('audio_padding_length_left', 2))
    right_pad = int(options.get('audio_padding_length_right', 2))
    req_fps = options.get('fps')
    use_fp16 = bool(options.get('use_float16', True))

    # ---- 2. Lazy torch import + GPU gate --------------------------------
    progress(0.01, 'Loading PyTorch...')
    try:
        import torch
    except ImportError as exc:
        raise _DepMissing(
            f'PyTorch is not installed ({exc}). MuseTalk needs an NVIDIA GPU with '
            'CUDA plus the torch/diffusers/transformers stack — see the README.')

    if not torch.cuda.is_available():
        raise _GpuUnavailable(
            'no CUDA device found. MuseTalk requires an NVIDIA GPU (CUDA); it '
            'cannot run on CPU. See the README for GPU requirements.')

    device = torch.device('cuda')
    weight_dtype = torch.float16 if use_fp16 else torch.float32

    # ---- 3. Ensure source + weights -------------------------------------
    progress(0.03, 'Preparing MuseTalk...')
    cache_root = _ensure_musetalk_source()
    _ensure_models(progress, lo=0.05, hi=0.40)

    # MuseTalk resolves several weights via relative ``./models/...`` paths;
    # run from the cache root that holds models/ and restore cwd afterwards.
    prev_cwd = os.getcwd()
    os.chdir(cache_root)
    tmpdir = tempfile.mkdtemp(prefix='musetalk_')
    try:
        import cv2
        import numpy as np
        try:
            from musetalk.utils.utils import load_all_model, datagen, get_video_fps, get_file_type
            from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
            from musetalk.utils.blending import get_image
            from musetalk.utils.audio_processor import AudioProcessor
            from musetalk.utils.face_parsing import FaceParsing
            from transformers import WhisperModel
        except ImportError as exc:
            raise _DepMissing(
                f'a MuseTalk / ML dependency failed to import ({exc}). The MuseTalk '
                'face/pose stack (mmcv/mmpose/mmdet) can be hard to build — see the '
                'README "Known limitations".')

        # ---- 4. Load models --------------------------------------------
        progress(0.42, 'Loading models onto GPU...')
        if version == 'v15':
            unet_model_path = os.path.join('models', 'musetalkV15', 'unet.pth')
            unet_config = os.path.join('models', 'musetalkV15', 'musetalk.json')
        else:
            unet_model_path = os.path.join('models', 'musetalk', 'pytorch_model.bin')
            unet_config = os.path.join('models', 'musetalk', 'musetalk.json')

        vae, unet, pe = load_all_model(
            unet_model_path=unet_model_path,
            vae_type='sd-vae-ft-mse',
            unet_config=unet_config,
            device=device,
        )
        pe = pe.to(device)
        vae.vae = vae.vae.to(device)
        unet.model = unet.model.to(device)
        if weight_dtype == torch.float16:
            pe = pe.half()
            vae.vae = vae.vae.half()
            unet.model = unet.model.half()
        timesteps = torch.tensor([0], device=device)

        # Whisper audio encoder + MuseTalk's audio feature processor.
        audio_processor = AudioProcessor(feature_extractor_path=os.path.join('models', 'whisper'))
        whisper = WhisperModel.from_pretrained(os.path.join('models', 'whisper'))
        whisper = whisper.to(device=device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False)

        # BiSeNet face parser (version-specific blend mode).
        fp = FaceParsing() if version != 'v15' else FaceParsing(
            left_cheek_width=int(options.get('left_cheek_width', 90)),
            right_cheek_width=int(options.get('right_cheek_width', 90)),
        )
        parsing_mode = 'jaw' if version == 'v15' else 'raw'

        # ---- 5. Audio -> Whisper feature chunks ------------------------
        progress(0.46, 'Extracting audio features...')
        # fps: honour request, else derive from the source video (default 25).
        ftype = get_file_type(video_path)
        if req_fps:
            fps = float(req_fps)
        elif ftype == 'video':
            fps = get_video_fps(video_path)
        else:
            fps = 25.0

        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features, device, weight_dtype, whisper, librosa_length,
            fps=fps, audio_padding_length_left=left_pad, audio_padding_length_right=right_pad,
        )

        # ---- 6. Frames -> face crops -> VAE latents --------------------
        progress(0.49, 'Detecting faces...')
        if ftype == 'video':
            frames_dir = os.path.join(tmpdir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            subprocess.run([ffmpeg, '-y', '-v', 'fatal', '-i', video_path,
                            '-start_number', '0', os.path.join(frames_dir, '%08d.png')],
                           check=True)
            input_img_list = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
        elif ftype == 'image':
            input_img_list = [video_path]
        else:  # a directory of frames
            input_img_list = sorted(
                glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]')))
        if not input_img_list:
            raise _BadParams(f'no frames extracted from {video_path!r}')

        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if version == 'v15':
                y2 = min(y2 + extra_margin, frame.shape[0])
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)

        # Cycle (forward + reverse) so a short clip can cover a longer audio.
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

        # ---- 7. UNet generation loop -----------------------------------
        progress(0.50, 'Generating lip-synced frames...')
        video_num = len(whisper_chunks)
        gen = datagen(whisper_chunks, input_latent_list_cycle, batch_size)
        res_frame_list = []
        done = 0
        for whisper_batch, latent_batch in gen:
            if is_cancelled():
                raise _Cancelled('cancelled by host')
            audio_feature_batch = pe(whisper_batch.to(device))
            latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)
            try:
                pred_latents = unet.model(
                    latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                recon = vae.decode_latents(pred_latents)
            except torch.cuda.OutOfMemoryError as exc:
                raise _Oom(f'CUDA out of memory ({exc}). Lower "batch_size" and retry.')
            except RuntimeError as exc:
                if 'out of memory' in str(exc).lower():
                    raise _Oom(f'CUDA out of memory ({exc}). Lower "batch_size" and retry.')
                raise
            for res_frame in recon:
                res_frame_list.append(res_frame)
            done += len(recon)
            progress(0.50 + 0.40 * (done / max(1, video_num)), 'Generating lip-synced frames...')

        # ---- 8. Blend the generated mouth back into each frame ---------
        progress(0.90, 'Blending frames...')
        result_dir = os.path.join(tmpdir, 'result')
        os.makedirs(result_dir, exist_ok=True)
        for i, res_frame in enumerate(res_frame_list):
            bbox = coord_list_cycle[i % len(coord_list_cycle)]
            ori_frame = frame_list_cycle[i % len(frame_list_cycle)].copy()
            x1, y1, x2, y2 = bbox
            if version == 'v15':
                y2 = min(y2 + extra_margin, ori_frame.shape[0])
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception:  # noqa: BLE001 — degenerate bbox; skip this frame
                continue
            combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2],
                                      mode=parsing_mode, fp=fp)
            cv2.imwrite(os.path.join(result_dir, f'{str(i).zfill(8)}.png'), combine_frame)

        # ---- 9. Encode frames + mux the original audio -----------------
        progress(0.96, 'Encoding video...')
        temp_video = os.path.join(tmpdir, 'temp.mp4')
        subprocess.run([ffmpeg, '-y', '-v', 'warning', '-r', str(fps), '-f', 'image2',
                        '-i', os.path.join(result_dir, '%08d.png'),
                        '-vcodec', 'libx264', '-vf', 'format=yuv420p', '-crf', '18',
                        temp_video], check=True)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        progress(0.98, 'Muxing audio...')
        subprocess.run([ffmpeg, '-y', '-v', 'warning', '-i', audio_path, '-i', temp_video,
                        '-c:v', 'copy', '-c:a', 'aac', '-shortest', output_path], check=True)

        if not (os.path.isfile(output_path) and os.path.getsize(output_path) > 0):
            raise _AddonError('ffmpeg produced no output file')

        progress(1.0, 'Done')
        return output_path
    except subprocess.CalledProcessError as exc:
        raise _AddonError(f'ffmpeg failed: {exc}')
    finally:
        os.chdir(prev_cwd)
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Protocol loop
# ---------------------------------------------------------------------------
def main():
    send({
        'type': 'hello',
        'protocol': PROTOCOL_VERSION,
        'addon': ADDON_ID,
        'version': ADDON_VERSION,
        'capabilities': [{'task': 'video.lipsync'}],
    })

    cancelled = set()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            continue
        ftype = frame.get('type')

        if ftype == 'ready':
            log('handshake complete')
            continue
        if ftype == 'shutdown':
            log('shutdown requested')
            break
        if ftype == 'cancel':
            target = frame.get('target')
            if target:
                cancelled.add(target)  # best-effort; checked between UNet batches
            continue

        if ftype == 'video.lipsync':
            req_id = frame.get('id')
            params = frame.get('params') or {}

            def progress(value, message=''):
                send({'id': req_id, 'type': 'progress',
                      'value': max(0.0, min(1.0, round(float(value), 4))),
                      'message': message})

            def is_cancelled():
                return req_id in cancelled

            try:
                out = run_lipsync(params, progress, is_cancelled)
                send({'id': req_id, 'type': 'result', 'data': {'output_path': out}})
            except _AddonError as exc:
                send({'id': req_id, 'type': 'error', 'code': exc.code, 'message': str(exc)})
            except Exception as exc:  # noqa: BLE001
                send({'id': req_id, 'type': 'error', 'code': 'internal', 'message': repr(exc)})
            finally:
                cancelled.discard(req_id)
            continue

    log('exiting')


if __name__ == '__main__':
    main()
