#!/usr/bin/env python3
"""Rung 3 of the GPU ladder: run ANY lab script against the CPU mock model.

Usage:
  CRAD_MOCK_MODEL=1 python mock_run.py dose_probe.py captures/<id> --turn 3 --direction R
  CRAD_MOCK_MODEL=1 python mock_run.py patch_probe.py captures/<id> --source 0 --target 1

capture.py / run.py already go through the launchers (which honour CRAD_MOCK_MODEL);
dose_probe.py and patch_*.py build crad_base.RecogModel directly, so this wrapper
installs the mock on the class before handing control to the script.

Mock outputs are wiring evidence only: random fp16 weights, character tokenizer,
no real decisions.  Never run this inside the real lab folder you will publish
from; use a scratch copy.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB / "src"))

import launch_common as lc  # noqa: E402

if not lc.is_mock():
    raise SystemExit("mock_run.py: set CRAD_MOCK_MODEL=1 (refusing to touch a real model through the mock path)")
if len(sys.argv) < 2:
    raise SystemExit(__doc__)

import crad_base  # noqa: E402

lc.install_model_probe(crad_base)
script = str(LAB / sys.argv[1])
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
