#!/usr/bin/env python3
"""Instrument fingerprint for runs/ and captures/ (Colab rebuild, 2026-10-10).

Two outputs with different fingerprints were produced by a different instrument,
software stack or hardware.  canonical_hash() returns "sha256:<hex>" over the
canonical JSON of the fingerprint (sorted keys, no whitespace).

Note for comparisons with the RTX 3060 series (2026-09-24: torch 2.14.0+cu130,
transformers 5.17.0, bitsandbytes 0.50.2, driver 595.91): a Colab fingerprint
WILL differ, by design.  Same numbers across fingerprints are a replication
result, not an assumption.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import launch_common as lc

SCHEMA = "crad-fingerprint-colab-0.1"


def _sha(p: Path):
    p = Path(p)
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def collect(lab: Path, src: Path, cfg: dict, manifest) -> dict:
    lab, src = Path(lab), Path(src)
    pins = json.loads((src / "PINNED.json").read_text()) if (src / "PINNED.json").exists() else {}
    a = cfg["args"]
    return {
        "schema": SCHEMA,
        "mock": lc.is_mock(),
        "source": {
            "instrument_commit": pins.get("commit"),
            "crad_v1_3": _sha(src / "CRAD_v1.3.py"),
            "crad_base": _sha(src / "crad_base.py"),
            "run_py": _sha(lab / "run.py"),
            "fingerprint_py": _sha(lab / "fingerprint.py"),
            "launch_common": _sha(lab / "launch_common.py"),
            "crad_launch": _sha(lab / "crad_launch.py"),
        },
        "config": {"name": cfg["name"], "sha256": "sha256:" + hashlib.sha256(_canon(cfg).encode()).hexdigest()},
        "model": {"id": a["model"], "revision": (manifest or {}).get("revision"),
                  "quantization_requested": "fp16" if a["no4bit"] else "nf4"},
        "env": lc.versions(),
        "gpu": lc.gpu_info(),
    }


def missing_required(fp: dict, needs_gpu: bool = True) -> list[str]:
    miss = []
    for k in ("crad_v1_3", "crad_base", "run_py", "launch_common"):
        if not fp["source"].get(k):
            miss.append(f"source.{k}")
    for k in ("torch", "numpy"):
        if not fp["env"].get(k):
            miss.append(f"env.{k}")
    if not fp.get("mock"):
        if not fp["env"].get("transformers"):
            miss.append("env.transformers")
        if not fp["model"].get("revision"):
            miss.append("model.revision")
        if fp["model"]["quantization_requested"] == "nf4" and not fp["env"].get("bitsandbytes"):
            miss.append("env.bitsandbytes")
    if needs_gpu and not fp["gpu"].get("available") and not fp.get("mock"):   # mock = CPU wiring test
        miss.append("gpu.name (no CUDA device)")
    return miss


def canonical_hash(fp: dict) -> str:
    return "sha256:" + hashlib.sha256(_canon(fp).encode()).hexdigest()
