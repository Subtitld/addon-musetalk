"""Subtitld MuseTalk lip-sync add-on (GPU/CUDA, torch-based).

Ships as a thin stdlib-only bootstrap that provisions the heavy ML stack into a
private cache venv on first use — see :mod:`musetalk_addon._bootstrap`.
"""
from musetalk_addon._common import ADDON_VERSION as __version__  # noqa: F401
