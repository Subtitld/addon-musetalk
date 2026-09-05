# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Subtitld MuseTalk lip-sync add-on (thin bootstrap).

Produces `dist/musetalk-addon/` — a small --onedir bundle holding the launcher
binary (`musetalk-addon[.exe]`) and NOTHING heavy. The launcher is the
stdlib-only bootstrap (:mod:`musetalk_addon._bootstrap`); the torch / diffusers
/ transformers / cv2 stack is NOT frozen here. Instead the add-on's own wheel
(built into `dist/` before this spec runs) is bundled under `wheels/`, and on
first use the bootstrap pip-installs that wheel — pulling its ML dependencies —
into a private cache venv. This keeps the release archive a few MB (a fully
frozen CUDA torch bundle is > 2 GiB, which GitHub refuses as a release asset).

Build order (see .github/workflows/release.yml):

    python -m build --wheel          # -> dist/subtitld_addon_musetalk-*.whl
    pyinstaller pyinstaller.spec --noconfirm

Model weights are NEVER bundled: they download once at first run into the
add-on cache.
"""

# ruff: noqa: F821  # PyInstaller injects Analysis/PYZ/EXE/COLLECT at runtime.

from __future__ import annotations

import glob
from pathlib import Path

SPEC_ROOT = Path(SPECPATH).resolve()

# Bundle our own wheel (built into dist/ beforehand) so the bootstrap can
# pip-install it — and, transitively, the ML stack — at first run.
_wheels = sorted(glob.glob(str(SPEC_ROOT / 'dist' / 'subtitld_addon_musetalk-*.whl')))
datas = [(w, 'wheels') for w in _wheels]

# All three package modules must be frozen so the bootstrap's imports resolve.
hiddenimports = [
    'musetalk_addon',
    'musetalk_addon._common',
    'musetalk_addon._bootstrap',
    'musetalk_addon._worker',
]

a = Analysis(
    [str(SPEC_ROOT / 'musetalk_addon' / '__main__.py')],
    pathex=[str(SPEC_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # The heavy ML stack is provisioned at runtime, never frozen — exclude
        # it so a stray function-level import can't drag it (or a multi-GB CUDA
        # runtime) into the bundle.
        'torch', 'torchvision', 'torchaudio',
        'diffusers', 'transformers', 'accelerate', 'huggingface_hub',
        'cv2', 'numpy', 'scipy', 'librosa', 'soundfile', 'einops',
        'omegaconf', 'gdown', 'musetalk',
        # The add-on speaks stdio JSON only — no GUI toolkit needed.
        'PySide6', 'PyQt6', 'PyQt5',
        'tkinter', 'Tkinter', '_tkinter',
        'matplotlib', 'pandas',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='musetalk-addon',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # stdio protocol — no window.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='musetalk-addon',
)
