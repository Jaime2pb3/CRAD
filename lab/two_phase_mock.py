#!/usr/bin/env python3
"""Rung 3 for two_phase.py: plan -> run -> analyze on the CPU mock model, in a throw-away copy.

Usage:  CRAD_MOCK_MODEL=1 python two_phase_mock.py        (a few minutes on CPU, zero GPU)

Reuses mock_ladder.py's planted decisions (the mock never writes "access code is").
Checks the wiring only: the gates (reproduction, self-patch, read-back), every mode, the
downstream readout and the verdict file. Numbers mean nothing by design.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
from mock_ladder import PLANTED, SKIP  # noqa: E402


def main() -> int:
    if os.environ.get("CRAD_MOCK_MODEL") != "1":
        raise SystemExit("two_phase_mock.py: run with CRAD_MOCK_MODEL=1")
    tmp = Path(tempfile.mkdtemp(prefix="two_phase_mock_"))
    work = tmp / "lab"
    shutil.copytree(LAB, work, ignore=shutil.ignore_patterns(*SKIP, "*.ipynb", "*.zip"))
    (work / "_planted.py").write_text(PLANTED)
    cfg = json.loads((work / "configs" / "qwen2.5-7b_T0.7_cal0_L14-20_s42.json").read_text())
    cfg["name"] = "MOCK_wiring"
    cfg["args"].update({"n-tokens": 24, "draws": 8, "n-perm": 20})
    (work / "configs" / "mock.json").write_text(json.dumps(cfg, indent=2))
    py, ok = sys.executable, True

    def step(name, cmd, want_rc=0):
        print(f"\n[two_phase-mock] {name}\n$ {' '.join(cmd)}", flush=True)
        p = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
        good = p.returncode == want_rc
        print("\n".join((p.stdout + p.stderr).strip().splitlines()[-12:]))
        if not good:
            print(f"[two_phase-mock] RED at '{name}' (exit {p.returncode}). Full output:\n"
                  + p.stdout[-6000:] + p.stderr[-6000:])
        return good

    ok = step("capture on the mock (INCONSISTENT expected: mock stamp)",
              [py, "capture.py", "configs/mock.json", "--superset", "13", "28"], want_rc=1)
    if ok:
        cap = sorted((work / "captures").glob("*_MOCK_wiring_cap13-28"))[-1]
        cj = json.loads((cap / "capture.json").read_text())
        ok = cj["problems"] == ["MOCK model: wiring test only, never data"]
        cj["problems"] = []
        (cap / "capture.json").write_text(json.dumps(cj, indent=2))        # temp copy only
        rel = str(cap.relative_to(work))
        ok = ok and step("two_phase selftest", [py, "two_phase.py", "selftest"])
        ok = ok and step("two_phase plan (planted)", [py, "_planted.py", "two_phase.py", "plan", rel,
                                                      "--max-pairs", "6"])
        ok = ok and step("two_phase run (planted)", [py, "_planted.py", "two_phase.py", "run", rel,
                                                     "--max-pairs", "6"])
        runs = sorted((cap / "two_phase").glob("2*")) if ok else []
        ok = ok and bool(runs) and step("two_phase analyze", [py, "two_phase.py", "analyze",
                                                              str(runs[-1].relative_to(work))])
        if ok:
            v = json.loads((runs[-1] / "verdict.json").read_text())
            ok = all(k in v for k in ("G0", "H1", "H2", "H3a", "H3b", "reading"))
    shutil.rmtree(tmp, ignore_errors=True)
    print("\n[two_phase-mock] GREEN: wiring verified on CPU. Numbers are meaningless by design."
          if ok else "\n[two_phase-mock] RED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
