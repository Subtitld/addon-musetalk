# MuseTalk lip-sync add-on for Subtitld

Generative, high-quality **lip-sync** for [Subtitld](https://github.com/Subtitld/subtitld),
powered by **[MuseTalk](https://github.com/TMElyralab/MuseTalk)** (MIT-licensed,
commercial use OK).

- Serves the `video.lipsync` task over Subtitld's add-on protocol.
- Given a **video** (or a still image) and an **audio** track, MuseTalk regenerates
  the speaker's mouth so it matches the audio — audio-conditioned mouth inpainting
  built on Stable Diffusion's VAE + a UNet, with Whisper audio features and a
  BiSeNet face parser for seamless blending. Output is H.264 + AAC.

## Requirements — please read

> [!IMPORTANT]
> **An NVIDIA GPU with CUDA is required.** MuseTalk is a torch/diffusers pipeline
> and **cannot run on CPU** — the add-on detects this and returns a clear
> `gpu_unavailable` error if no CUDA device is present. There is **no macOS build**
> (CUDA-only); Linux and Windows only.

> [!IMPORTANT]
> **The PyTorch runtime installs itself on first use.** The shipped add-on is a
> tiny stdlib-only launcher (a few MB). The first time you run a lip-sync, it
> builds a private virtual-env under `~/.cache/subtitld/musetalk/runtime/` and
> pip-installs its own wheel — pulling PyTorch (CUDA), diffusers, transformers,
> opencv, etc. — into it (**several GB, downloaded once**). This is what lets
> MuseTalk ship on Linux at all: a fully-bundled CUDA torch build is > 2 GiB,
> past GitHub's release-asset limit. Provisioning needs **Python 3.10+** on
> `PATH` (or point `SUBTITLD_MUSETALK_PYTHON` at one) and a network connection.

> [!IMPORTANT]
> **Large model download on first use (~several GB).** After the runtime is set
> up, the MuseTalk weights, the SD-VAE, Whisper-tiny, DWPose, and the BiSeNet
> face-parsing models download once into `~/.cache/subtitld/musetalk/models/`
> (override the cache root with the `SUBTITLD_ADDON_CACHE` environment variable).
> After that, inference is fully local.

`ffmpeg` is used to split frames and mux audio; the add-on resolves it from
`SUBTITLD_FFMPEG_EXECUTABLE` (set by Subtitld) or from `PATH`.

## Usage

Install it from Subtitld's **Add-ons** dialog, then use it wherever Subtitld exposes
the `video.lipsync` task. The add-on takes:

- `video_path` — the source video (25 fps recommended), still image, or a folder of frames.
- `audio_path` — the driving audio.
- `output_path` — where the lip-synced `.mp4` is written.
- `options` — see the configurable fields below.

The first run provisions the PyTorch runtime (see *Requirements*), downloads the
models, and — if the MuseTalk source package is not already installed —
shallow-clones it into the cache. All of that happens once, with progress
reported to Subtitld; subsequent runs start straight into inference and are
fully local.

### First-run environment variables

| Variable | Purpose |
|----------|---------|
| `SUBTITLD_ADDON_CACHE` | Root for the runtime venv + model weights (default `~/.cache/subtitld/musetalk/`). |
| `SUBTITLD_MUSETALK_PYTHON` | Base Python 3.10+ used to build the runtime venv. Defaults to a `python3.x` found on `PATH`. |
| `SUBTITLD_MUSETALK_TORCH_INDEX` | Extra pip index for the CUDA torch wheels (default `https://download.pytorch.org/whl/cu121`). Point it at a different CUDA build or a mirror if needed. |
| `SUBTITLD_FFMPEG_EXECUTABLE` | ffmpeg binary (Subtitld sets this; otherwise `PATH` is used). |

## Options

| Option | Default | Notes |
|--------|---------|-------|
| `version` | `v15` | `v15` (MuseTalk 1.5, sharper) or `v1` (1.0). |
| `bbox_shift` | `0` | Vertical face-crop shift to tune mouth openness (MuseTalk 1.0). |
| `extra_margin` | `10` | Extra jaw pixels below the crop (MuseTalk 1.5). |
| `batch_size` | `8` | Frames per UNet pass. Lower it if you hit `oom`. |
| `fps` | `25` | Output frame rate. MuseTalk is trained at 25 fps. |
| `use_float16` | `true` | fp16 to cut VRAM. |

## Error codes

`bad_params` (bad/missing paths or no ffmpeg), `gpu_unavailable` (no CUDA device),
`model_missing` (a required weight or the MuseTalk source could not be obtained),
`network_unavailable` (download/clone failed), `oom` (CUDA out of memory — lower
`batch_size`), `internal` (anything else, including a missing ML dependency).

## Development

```bash
pip install -e '.[dev]'
python -m musetalk_addon          # runs the add-on (speaks the protocol on stdio)
pytest                            # protocol smoke tests (no torch / no model download)
```

The smoke tests deliberately run **without** the ML stack — every heavy import
(`torch`, `diffusers`, `transformers`, `cv2`, and the `musetalk` package) is lazy,
so the handshake and error paths work even on a machine with no GPU and no torch.

Build the thin bootstrap bundle (the heavy stack is NOT frozen — it installs at
runtime):

```bash
pip install -e '.[build]'
python -m build --wheel                    # -> dist/subtitld_addon_musetalk-*.whl
pyinstaller pyinstaller.spec --noconfirm   # -> dist/musetalk-addon/ (bundles the wheel under wheels/)
```

Releases are built for linux/Windows by the `Release` GitHub Action on a `v*` tag
and published to the Subtitld add-ons catalog.

## Known limitations

- **CUDA-only, large first-run download.** See *Requirements* above.
- **First run needs a Python 3.10+ toolchain and network.** The runtime venv is
  built with a base Python found on `PATH` (or `SUBTITLD_MUSETALK_PYTHON`). If no
  suitable interpreter is found you get a clear `internal` error explaining how to
  fix it — no partial/half-broken install.
- **Face/pose deps are hard to build.** MuseTalk's landmark/pose stack
  (`mmcv` / `mmpose` / `mmdet`) is notoriously difficult to compile. It is pulled
  in at runtime by the provisioned venv and imported lazily by MuseTalk at
  inference time. If it fails to import you get a clear `internal` error pointing
  here. Installing a matching `mmcv`/`mmpose`/`mmdet` set for your torch+CUDA
  version is the fiddliest part of the setup.
- **BiSeNet face-parsing weight** (`79999_iter.pth`) is hosted on Google Drive; the
  add-on uses `gdown` to fetch it. If `gdown` is unavailable, download it manually
  into `~/.cache/subtitld/musetalk/models/face-parse-bisent/`.
- **Quitting during the first-run install (Windows).** Provisioning runs on the
  bootstrap's protocol thread, so a `shutdown` sent while the multi-GB install is
  in flight is not read until it finishes. On Linux/macOS the host reaps the whole
  process group, which stops `pip` too; on Windows only the bootstrap is
  terminated, so a `pip` child can keep downloading in the background until it
  completes. Re-running the add-on afterwards simply resumes/reuses what landed.

## License

MIT (this add-on and MuseTalk itself). The downloaded models carry their own
licenses (SD-VAE, Whisper, DWPose, BiSeNet, LatentSync/SyncNet).
