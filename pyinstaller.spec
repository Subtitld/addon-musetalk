# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Subtitld MuseTalk lip-sync add-on.

Produces `dist/musetalk-addon/` — a self-contained --onedir bundle holding the
launcher binary (`musetalk-addon[.exe]`) plus the bundled torch / diffusers /
transformers / opencv runtimes MuseTalk needs.

Run from the addon root:

    pyinstaller pyinstaller.spec --noconfirm

The release workflow zips `dist/musetalk-addon/` together with manifest.json /
LICENSE / README.md into the platform-tagged archive that Subtitld's
AddonsDialog installs.

NOTE (size / build): the CUDA torch stack makes this bundle very large
(multiple GB) and slow to build; CI runners may need extra disk / timeout
tuning. The MuseTalk face/pose deps (mmcv / mmpose / mmdet) are hard to build
and are imported lazily by MuseTalk at inference time — they are NOT frozen
here; see README "Known limitations". Model weights are NEVER bundled: they are
downloaded once at first run into the add-on cache.
"""

# ruff: noqa: F821  # PyInstaller injects Analysis/PYZ/EXE/COLLECT at runtime.

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

SPEC_ROOT = Path(SPECPATH).resolve()

# torch / torchvision / opencv ship compiled extension modules + shared
# libraries (and CUDA runtimes) next to their Python packages; collect them so
# the frozen bundle can import them. transformers / diffusers are pure-python
# but pull in many lazily-imported submodules.
binaries = (
    collect_dynamic_libs('torch')
    + collect_dynamic_libs('torchvision')
    + collect_dynamic_libs('cv2')
)
hiddenimports = (
    ['torch', 'torchvision', 'torchaudio', 'cv2', 'numpy']
    + collect_submodules('diffusers')
    + collect_submodules('transformers')
    + collect_submodules('librosa')
)

a = Analysis(
    [str(SPEC_ROOT / 'musetalk_addon' / '__main__.py')],
    pathex=[str(SPEC_ROOT)],
    binaries=binaries,
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
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
