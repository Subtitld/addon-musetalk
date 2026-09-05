#!/usr/bin/env python3
"""Entry point for the Subtitld MuseTalk lip-sync add-on.

Two modes share one package:

  * **bootstrap** (default) — the tiny, stdlib-only executable that ships in the
    release archive. It handshakes, validates params, provisions the heavy ML
    runtime into a cache venv on first use, then proxies to the worker. See
    :mod:`musetalk_addon._bootstrap`.
  * **worker** (``--worker`` or ``SUBTITLD_MUSETALK_WORKER=1``) — the heavy half
    that imports torch and runs MuseTalk. It only ever runs inside the
    provisioned venv, launched by the bootstrap. See
    :mod:`musetalk_addon._worker`.
"""

import os
import sys


def main():
    worker_mode = ('--worker' in sys.argv[1:]
                   or os.environ.get('SUBTITLD_MUSETALK_WORKER') == '1')
    if worker_mode:
        from musetalk_addon._worker import run_worker_loop
        run_worker_loop(emit_hello=False)
    else:
        from musetalk_addon._bootstrap import run_bootstrap
        run_bootstrap()


if __name__ == '__main__':
    main()
