#!/usr/bin/env python3
"""CRAD lab runner — Colab rebuild (2026-10-10).

TwoQuarks Research — Luis Jaime Ledesma Perez

The original run.py stayed on the RTX 3060 machine.  This one is rebuilt against
the interface capture.py, dose_probe.py, patch_probe.py, patch_batch.py and
crad_contract.py consume (SRC, preflight, cli_args, consistency, extract, fmt,
run_tag, die), keeping the lab's rules:

  * the instrument (src/CRAD_v1.3.py, src/crad_base.py, commit eb8217b) runs
    UNMODIFIED and is hash-pinned in src/PINNED.json;
  * every CRAD option is written explicitly in the config (no silent defaults);
  * every run gets its own folder, generated RUN.md and a fingerprint.

Usage:
  python run.py configs/<config>.json        one run -> runs/<run_id>/
  python run.py --selftest                   CPU, offline: pins, configs, CLI mapping, extract

Config format:
  {"name": "...", "description": "...", "model_revision": "<HF commit prefix or null>",
   "args": {<every CRAD_v1.3 option, CLI spelling without "--">}}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
SRC = LAB / "src"
RUNS = LAB / "runs"
CONFIGS = LAB / "configs"
PIN_FILE = SRC / "PINNED.json"

# Every option of CRAD_v1.3.main() except --selftest / --out (owned by the lab).
CRAD_ARGS = (
    "model", "layer-lo", "layer-hi", "temperature-mode", "temp", "calibration-temp",
    "analysis-temp", "force-deterministic-repeats", "calibration-draws", "draws", "n-tokens",
    "focus-from-turn", "secret", "pre-window", "post-window", "action-threshold",
    "action-layer-frac", "min-contiguous-layers", "refractory-tokens", "baseline-pre-tokens",
    "hold-horizon", "hold-ratio-min", "release-ratio-max", "hold-layer-frac", "hold-eps-frac",
    "min-baseline-support", "min-baseline-layer-frac", "n-perm", "seed", "no4bit",
)
BOOL_ARGS = ("force-deterministic-repeats", "no4bit")
NULLABLE_ARGS = ("calibration-temp", "analysis-temp")


def die(msg: str):
    raise SystemExit(f"[run] {msg}")


def fmt(x, n: int = 4) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return str(x)


def sha256_file(p: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_tag(cfg: dict) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", cfg["name"]).strip("-")


def cli_args(args: dict) -> list[str]:
    out: list[str] = []
    for k in CRAD_ARGS:
        v = args[k]
        if k in BOOL_ARGS:
            if v:
                out.append(f"--{k}")
            continue
        if v is None:
            continue
        out += [f"--{k}", str(v)]
    return out


def effective_temps(args: dict) -> dict:
    base = 0.0 if args["temperature-mode"] == "off" else float(args["temp"])
    cal, ana = args.get("calibration-temp"), args.get("analysis-temp")
    return {"calibration": base if cal is None else float(cal),
            "analysis": base if ana is None else float(ana)}


def validate_config(cfg: dict) -> list[str]:
    probs = []
    for key in ("name", "args"):
        if key not in cfg:
            probs.append(f"config has no '{key}'")
    if probs:
        return probs
    a = cfg["args"]
    missing = [k for k in CRAD_ARGS if k not in a]
    unknown = [k for k in a if k not in CRAD_ARGS]
    if missing:
        probs.append(f"options not written explicitly: {missing}")
    if unknown:
        probs.append(f"unknown options: {unknown}")
    if missing or unknown:
        return probs
    for k in BOOL_ARGS:
        if not isinstance(a[k], bool):
            probs.append(f"'{k}' must be true/false")
    for k in CRAD_ARGS:
        if a[k] is None and k not in NULLABLE_ARGS:
            probs.append(f"'{k}' is null; only {NULLABLE_ARGS} may be null")
    if a["temperature-mode"] not in ("on", "off"):
        probs.append("temperature-mode must be 'on' or 'off'")
    if not (0 <= int(a["layer-lo"]) < int(a["layer-hi"])):
        probs.append("invalid layer band")
    return probs


def check_pins() -> list[str]:
    if not PIN_FILE.exists():
        return [f"{PIN_FILE} missing"]
    pins = json.loads(PIN_FILE.read_text())["files"]
    probs = []
    for name, want in pins.items():
        p = SRC / name
        if not p.exists():
            probs.append(f"pinned file missing: src/{name}")
        elif sha256_file(p) != want:
            probs.append(f"src/{name} does not match its pin (instrument modified?)")
    return probs


def preflight(cfg: dict):
    """Refuse to run anything that is not the pinned instrument with an explicit config.
    Returns the model manifest {"model", "revision"} (None for selftest configs)."""
    probs = check_pins() + validate_config(cfg)
    if probs:
        die("preflight failed:\n  - " + "\n  - ".join(probs))
    if cfg.get("selftest"):
        return None
    import launch_common as lc
    if lc.is_mock():
        return {"model": cfg["args"]["model"], "revision": "mock-00000000", "mock": True}
    from huggingface_hub import HfApi, snapshot_download
    model = cfg["args"]["model"]
    main_sha = HfApi().model_info(model, revision="main").sha
    pin = cfg.get("model_revision")
    if pin and not main_sha.startswith(pin):
        die(f"{model}: config pins revision {pin}, but 'main' is now {main_sha}. CRAD_v1.3 loads "
            "'main' (it takes no revision argument), so the pin cannot be honoured without "
            "editing the instrument. Decide explicitly: set model_revision to null (new series) "
            "or stop.")
    print(f"[run] model {model} @ {main_sha} (pin {pin or 'none'}): ensuring local snapshot", flush=True)
    snapshot_download(model, revision=main_sha)
    return {"model": model, "revision": main_sha, "pin": pin}


def consistency(cfg: dict, result: dict, rt: dict) -> list[str]:
    """Did CRAD run what the config says, on the hardware the lab assumes?"""
    a, m = cfg["args"], result["meta"]
    probs = []
    if (m["layer_lo"], m["layer_hi"]) != (a["layer-lo"], a["layer-hi"]):
        probs.append(f"band ran [{m['layer_lo']},{m['layer_hi']}) != config [{a['layer-lo']},{a['layer-hi']})")
    t = effective_temps(a)
    if (m["effective_calibration_temp"], m["effective_analysis_temp"]) != (t["calibration"], t["analysis"]):
        probs.append(f"temperatures ran cal={m['effective_calibration_temp']} ana={m['effective_analysis_temp']} "
                     f"!= config cal={t['calibration']} ana={t['analysis']}")
    for meta_key, arg_key in (("n_tokens", "n-tokens"), ("seed", "seed"), ("model", "model")):
        if m.get(meta_key) != a[arg_key]:
            probs.append(f"{meta_key} ran {m.get(meta_key)} != config {a[arg_key]}")
    if rt.get("mock"):
        probs.append("MOCK model: wiring test only, never data")
    else:
        want = "fp16" if a["no4bit"] else "nf4"
        if rt.get("quantization") != want:
            probs.append(f"quantization loaded {rt.get('quantization')} != config {want}")
    if rt.get("offload"):
        probs.append(f"model partly offloaded to CPU/disk: {rt.get('device_map')}")
    if not rt.get("model_commit"):
        probs.append("loaded model commit unknown")
    if rt.get("error"):
        probs.append(f"launcher reported an error: {str(rt['error'])[:200]}")
    return probs


def extract(result: dict) -> dict:
    """Small summary.json from CRAD's result.jsonl (robust to draws with event None)."""
    cal = result["calibration"]
    g = cal["geometry"]
    draws = []
    for d in result["draws"]:
        ev = d.get("event") or {}
        an = d["analysis"]
        draws.append({
            "draw": d["draw"], "label": d["label"], "final_label_raw": d["final_label_raw"],
            "event_global_token": ev.get("event_global_token"),
            "is_pseudo_event": bool(ev.get("is_pseudo_event", False)),
            "n_frames": d["n_frames"], "n_candidates": an["n_candidates"], "n_override": an["n_override"],
            "n_consistent": an["n_consistent"], "n_ambiguous": an["n_ambiguous"],
            "features": d["features"],
            "candidates": [{k: e.get(k) for k in ("global_token", "token", "classification", "rel_token",
                                                   "action_R_mean", "hold_fraction")}
                           for e in an["candidate_events"]],
        })
    s = result["summary"]
    return {"meta": result["meta"], "status": s["status"],
            "geometry": {k: g.get(k) for k in ("cos_LR_median", "cos_LR_min", "cos_LR_max",
                                               "antipodal_fraction", "coactivation_rate_all_states")},
            "counts": cal["counts"], "n_comply": s["n_comply"], "n_control": s["n_control"],
            "permutation_gates": s["permutation_gates"], "draws": draws}


# ------------------------------------------------------------------ run mode --
def write_run_md(rd: Path, run_id: str, cfg: dict, fp_hash: str, rt: dict, summary, problems, rc) -> str:
    verdict = "FAILED" if rc != 0 else ("INCONSISTENT" if problems else "OK")
    L = [f"# Run `{run_id}`", "", "| Field | Value |", "|---|---|",
         f"| Verdict | **{verdict}** |", f"| Config | `{cfg['name']}` |", f"| Exit code | {rc} |",
         f"| Fingerprint | `{fp_hash}` |", f"| Model commit | `{rt.get('model_commit') or '—'}` |",
         f"| GPU | {(rt.get('gpu') or {}).get('name', '—')} |",
         f"| Wall time | {fmt(rt.get('wall_seconds'), 1)} s |", ""]
    if problems:
        L += ["## Problems", ""] + [f"- {p}" for p in problems] + [""]
    if summary:
        g = summary["geometry"]
        L += ["## CRAD readings", "",
              f"cos(L,R) median {fmt(g['cos_LR_median'])} · coactivation {fmt(g['coactivation_rate_all_states'], 3)} "
              f"· status `{summary['status']}` · COMPLY {summary['n_comply']} / control {summary['n_control']}", "",
              "| draw | label raw | event token | candidates | OVERRIDE | n_pre_override |", "|---|---|---|---|---|---|"]
        for d in summary["draws"]:
            L.append(f"| {d['draw']} | {d['final_label_raw']} | {d['event_global_token']}"
                     f"{' (pseudo)' if d['is_pseudo_event'] else ''} | {d['n_candidates']} | {d['n_override']} "
                     f"| {fmt(d['features'].get('n_pre_override'), 0)} |")
    (rd / "RUN.md").write_text("\n".join(L) + "\n")
    return verdict


def run_main(config_path: str) -> int:
    import fingerprint as fpmod
    cfg = json.loads(Path(config_path).read_text())
    manifest = preflight(cfg)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{stamp}_{run_tag(cfg)}"
    rd = RUNS / run_id
    rd.mkdir(parents=True)
    fp = fpmod.collect(LAB, SRC, cfg, manifest)
    miss = fpmod.missing_required(fp, needs_gpu=not fp.get("mock"))
    fp_hash = fpmod.canonical_hash(fp)
    (rd / "fingerprint.json").write_text(json.dumps(fp, indent=2, sort_keys=True) + "\n")
    (rd / "fingerprint.sha256").write_text(fp_hash + "\n")
    (rd / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if miss:
        (rd / "RUN.md").write_text(f"# Run `{run_id}`\n\n**REFUSED** — missing fingerprint fields: {miss}\n")
        die(f"missing fingerprint fields {miss}")
    cmd = [sys.executable, str(LAB / "crad_launch.py"), str(SRC), str(rd / "runtime.json"), "--"] + \
        cli_args(cfg["args"]) + ["--out", str(rd / "result.jsonl")]
    (rd / "command.txt").write_text(" ".join(cmd) + "\n")
    print(f"[run] {run_id}\n[run] fingerprint {fp_hash}", flush=True)
    with open(rd / "stdout.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=rd)
        for line in proc.stdout:
            sys.stdout.write(line); log.write(line)
        rc = proc.wait()
    rt = json.loads((rd / "runtime.json").read_text()) if (rd / "runtime.json").exists() else {}
    summary, problems = None, []
    if rc == 0:
        result = json.loads((rd / "result.jsonl").read_text().splitlines()[0])
        summary = extract(result)
        (rd / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        problems = consistency(cfg, result, rt)
        if manifest and rt.get("model_commit") and rt["model_commit"] != manifest["revision"]:
            problems.append(f"loaded commit {rt['model_commit']} != manifest {manifest['revision']}")
    verdict = write_run_md(rd, run_id, cfg, fp_hash, rt, summary, problems, rc)
    idx = RUNS / "INDEX.md"
    if not idx.exists():
        idx.write_text("# CRAD runs index\n\nGenerated by run.py; never edited by hand.\n\n"
                       "| run | verdict | config | fingerprint |\n|---|---|---|---|\n")
    with open(idx, "a") as fh:
        fh.write(f"| [{run_id}]({run_id}/RUN.md) | {verdict} | {cfg['name']} | `{fp_hash[7:19]}` |\n")
    with open(RUNS / "run_log.jsonl", "a") as fh:
        fh.write(json.dumps({"ts": stamp, "event": "run", "run_id": run_id, "config": cfg["name"],
                             "instrument_hash": fp_hash, "verdict": verdict, "problems": problems}) + "\n")
    print(f"[run] {verdict}: {rd}/RUN.md")
    return 0 if verdict == "OK" else 1


# ------------------------------------------------------------------ selftest --
def selftest() -> int:
    tests = []
    tests.append(("instrument matches src/PINNED.json (eb8217b)", check_pins() == []))
    help_text = subprocess.run([sys.executable, str(SRC / "CRAD_v1.3.py"), "--help"],
                               capture_output=True, text=True, cwd=SRC).stdout
    flags_ok = all(f"--{k}" in help_text for k in CRAD_ARGS)
    tests.append(("every lab option is a real CRAD_v1.3 flag", flags_ok))
    v13_flags = set(re.findall(r"--([a-z0-9-]+)", help_text)) - {"help", "selftest", "out"}
    tests.append(("every CRAD_v1.3 flag is covered by the lab (none left to defaults)",
                  v13_flags <= set(CRAD_ARGS)))
    cfgs = sorted(CONFIGS.glob("*.json"))
    bad = {p.name: validate_config(json.loads(p.read_text())) for p in cfgs}
    tests.append((f"configs explicit and valid ({len(cfgs)} files)", cfgs and all(not v for v in bad.values())))
    a = json.loads(cfgs[0].read_text())["args"] if cfgs else {}
    if a:
        cl = cli_args(a)
        tests.append(("cli_args: no None/False leaks, booleans as bare flags",
                      "None" not in cl and "False" not in cl and "True" not in cl))
    fake = {"meta": {"seed": 1}, "calibration": {"counts": {}, "geometry": {"cos_LR_median": 0.1}},
            "summary": {"status": "ok", "n_comply": 0, "n_control": 1, "permutation_gates": {}},
            "draws": [{"draw": 0, "label": "NONCOMPLY", "final_label_raw": "REFUSE", "event": None,
                       "n_frames": 3, "features": {"n_pre_override": 0.0},
                       "analysis": {"n_candidates": 0, "n_override": 0, "n_consistent": 0, "n_ambiguous": 0,
                                    "candidate_events": []}}]}
    try:
        s = extract(fake)
        tests.append(("extract survives event=None (the old run.extract crash)", s["draws"][0]["event_global_token"] is None))
    except Exception:
        tests.append(("extract survives event=None (the old run.extract crash)", False))
    tests.append(("fmt / run_tag", fmt(None) == "—" and fmt(0.15234, 3) == "0.152" and run_tag({"name": "a b/c"}) == "a-b-c"))
    print("run.py (colab rebuild) selftest")
    for name, ok in tests:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    for name, probs in bad.items():
        for p in probs:
            print(f"    {name}: {p}")
    return 0 if all(ok for _, ok in tests) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="CRAD lab runner (Colab rebuild)")
    ap.add_argument("config", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.config:
        ap.error("config required")
    return run_main(a.config)


if __name__ == "__main__":
    raise SystemExit(main())
