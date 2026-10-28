#!/usr/bin/env python3
"""Plain launcher used by run.py: runs the pinned, unmodified CRAD_v1.3.main().

Usage (run.py builds this line; do not call by hand unless debugging):
  python crad_launch.py SRC RUNTIME_JSON -- <explicit CRAD_v1.3 CLI>

It only observes which model was loaded and writes runtime.json (also on failure).
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import launch_common as lc


def main() -> int:
    if "--" not in sys.argv or sys.argv.index("--") != 3:
        raise SystemExit("usage: crad_launch.py SRC RUNTIME_JSON -- <crad args>")
    src, rt_path = Path(sys.argv[1]), Path(sys.argv[2])
    crad_args = sys.argv[4:]
    t0 = time.time()
    err = None
    try:
        v13, crad_base = lc.load_v13_module(src)
        lc.install_model_probe(crad_base)
        sys.argv = ["CRAD_v1.3.py"] + crad_args
        v13.main()
    except SystemExit as e:
        if e.code not in (None, 0):
            err = f"SystemExit: {e.code}"
            raise
    except BaseException:
        err = traceback.format_exc()
        raise
    finally:
        lc.write_json(rt_path, lc.runtime_info(t0, err))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
