#!/usr/bin/env python3
"""Capture the protocol v0.2 §10 replay artifact for one frozen config.

Usage:
  .venv/bin/python capture.py configs/<config>.json
  .venv/bin/python capture.py configs/<config>.json --compare-run runs/<run_id>
  .venv/bin/python capture.py configs/<config>.json --superset 13 20

It runs the UNMODIFIED CRAD_v1.3.py exactly like run.py does (same preflight,
same config, same explicit CLI), through capture_launch.py instead of
crad_launch.py, and files everything in captures/<capture_id>/:

  CAPTURE.md        generated summary + verdict (never hand-edited)
  config.json  command.txt  fingerprint.json  fingerprint.sha256
  runtime.json      what was really loaded
  stdout.log  result.jsonl  summary.json   (CRAD's own output, as in runs/)
  capture.npz       superset hidden states + token-level coordinates (immutable)
  capture.json      checks, hashes, superset, fp16 exactness

Checks (any failure -> INCONSISTENT; the folder stays, like runs/):
  - run.py's own consistency checks (band, temperature, quantization, offload, commit)
  - per analysis draw: captured frame count == CRAD's n_frames
  - calibration: N/L/R state and delta counts rebuilt with CRAD's OWN
    stack_states / within_turn_deltas from the band slice == CRAD's counts
  - every analysis frame has an emitted token ID
  - optional --compare-run: labels, event tokens, OVERRIDE counts and geometry
    must equal an earlier run of the same config (reproducibility evidence)

runs/ is never touched.  Captures have their own INDEX.md and log.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import fingerprint as fpmod
import run as runmod

LAB = Path(__file__).resolve().parent
CAPS = LAB / "captures"
DEFAULT_SUPERSET = (13, 20)  # protocol v0.2 §10


def sha256_file(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def load_v13(src: Path):
    """Import CRAD_v1.3.py (dotted filename) read-only, to reuse its own functions."""
    sys.path.insert(0, str(src))
    spec = importlib.util.spec_from_file_location("crad_v1_3_pinned", src / "CRAD_v1.3.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crad_v1_3_pinned"] = mod  # required for dataclasses in a dotted file
    spec.loader.exec_module(mod)
    return mod


def band_frames(z, mask: np.ndarray, lo: int, hi: int) -> list:
    """Rebuild CRAD-shaped frames (phi sliced to [lo,hi)) for the selected rows."""
    sup_lo, sup_hi = json.loads(str(z["meta"]))["superset"]
    if not (sup_lo <= lo < hi <= sup_hi):
        raise SystemExit(f"[capture] band [{lo},{hi}) outside captured superset [{sup_lo},{sup_hi})")
    a, b = lo - sup_lo, hi - sup_lo
    idx = np.flatnonzero(mask)
    phi, turn, glob = z["phi"], z["turn"], z["global_token"]
    return [{"phi": np.asarray(phi[i, a:b], dtype=np.float64),
             "turn_idx": int(turn[i]), "global_token": int(glob[i])} for i in idx]


def check_capture(npz: Path, result: dict, v13) -> tuple[list, dict]:
    problems, facts = [], {}
    with np.load(npz, allow_pickle=False) as zf:
        z = {k: zf[k] for k in zf.files}  # load once; npz re-reads on every key access
    meta = json.loads(str(z["meta"]))
    lo, hi = meta["crad_band"]
    kind, branch, draw = z["kind"], z["branch"], z["draw"]
    facts.update({"frames_total": int(len(kind)), "phi_dtype": meta["phi_dtype"],
                  "fp16_exact": meta["fp16_exact"], "superset": meta["superset"]})

    # analysis draws: frame counts and token IDs
    draws = result["draws"]
    for d in draws:
        k = int(d["draw"])
        n_cap = int(np.sum((kind == 1) & (draw == k)))
        if n_cap != int(d["n_frames"]):
            problems.append(f"draw {k}: captured {n_cap} frames, CRAD n_frames={d['n_frames']}")
    n_draws_cap = len(set(draw[kind == 1].tolist()))
    if n_draws_cap != len(draws):
        problems.append(f"captured {n_draws_cap} draws, CRAD has {len(draws)}")
    missing_ids = int(np.sum((kind == 1) & (z["token_id"] < 0)))
    if missing_ids:
        problems.append(f"{missing_ids} analysis frames without token ID")

    # calibration populations rebuilt with CRAD's own functions
    counts = result["calibration"].get("counts", {})
    rebuilt = {}
    for code, name in ((0, "N"), (1, "L"), (2, "R")):
        fr = band_frames(z, (kind == 0) & (branch == code), lo, hi)
        rebuilt[f"states_{name}"] = int(len(v13.stack_states(fr)))
        rebuilt[f"deltas_{name}"] = int(len(v13.within_turn_deltas(fr)))
    for k, v in rebuilt.items():
        if counts.get(k) != v:
            problems.append(f"calibration {k}: rebuilt {v}, CRAD {counts.get(k)}")
    facts["calibration_counts"] = rebuilt

    # value-level identity: CRAD's own calibration on the captured band must
    # reproduce CRAD's reported geometry exactly (not just the counts)
    pops = {}
    for code, name in ((0, "N"), (1, "L"), (2, "R")):
        fr = band_frames(z, (kind == 0) & (branch == code), lo, hi)
        pops[f"H_{name}"], pops[f"D_{name}"] = v13.stack_states(fr), v13.within_turn_deltas(fr)
    recal = v13.calibrate_from_arrays(pops["H_N"], pops["H_L"], pops["H_R"],
                                      pops["D_N"], pops["D_L"], pops["D_R"], layer_lo=lo, layer_hi=hi)
    g_crad = result["calibration"]["geometry"]
    for key in ("cos_LR_median", "coactivation_rate_all_states", "antipodal_fraction"):
        got, want = recal.geometry.get(key), g_crad.get(key)
        if got != want:
            problems.append(f"geometry {key}: recomputed from capture {got} != CRAD {want}")
    facts["geometry_recomputed"] = {k: recal.geometry.get(k) for k in ("cos_LR_median", "coactivation_rate_all_states")}
    return problems, facts


def safe_extract(result: dict) -> dict:
    """run.extract() crashes when a draw's "event" is None (no real event in the
    whole run, so no pseudo-event either): d.get("event", {}) returns None.
    run.py is left untouched; capture passes a copy with None -> {}."""
    fixed = dict(result)
    fixed["draws"] = [dict(d, event=(d.get("event") or {})) for d in result["draws"]]
    return runmod.extract(fixed)


def compare_to_run(summary: dict, run_dir: Path) -> list:
    ref = json.loads((run_dir / "summary.json").read_text())
    probs = []
    if len(ref["draws"]) != len(summary["draws"]):
        return [f"compare: {len(summary['draws'])} draws vs {len(ref['draws'])} in {run_dir.name}"]
    for a, b in zip(summary["draws"], ref["draws"]):
        for key in ("label", "final_label_raw", "event_global_token", "n_override", "n_candidates"):
            if a.get(key) != b.get(key):
                probs.append(f"compare draw {a['draw']}: {key} {a.get(key)} != {b.get(key)}")
    ga, gb = summary["geometry"]["cos_LR_median"], ref["geometry"]["cos_LR_median"]
    if ga != gb:
        probs.append(f"compare: cos_LR_median {ga} != {gb}")
    return probs


def write_md(cd: Path, cap_id: str, cfg: dict, fp_hash: str, rt: dict, summary, facts: dict,
             problems: list, rc: int, compared: str | None) -> str:
    verdict = "FAILED" if rc != 0 else ("INCONSISTENT" if problems else "OK")
    f = runmod.fmt
    L = [f"# Capture `{cap_id}`", "", "| Field | Value |", "|---|---|",
         f"| Verdict | **{verdict}** |", f"| Config | `{cfg['name']}` |", f"| Exit code | {rc} |",
         f"| Fingerprint | `{fp_hash}` |", f"| Model commit | `{rt.get('model_commit') or '—'}` |",
         f"| Superset captured | {facts.get('superset', '—')} |",
         f"| Frames | {facts.get('frames_total', '—')} ({facts.get('phi_dtype', '—')}, "
         f"fp16 exact: {facts.get('fp16_exact', '—')}) |",
         f"| Compared with | {compared or '—'} |",
         f"| Wall time | {f(rt.get('wall_seconds'), 1)} s |", ""]
    if problems:
        L += ["## Problems", ""] + [f"- {p}" for p in problems] + [""]
    if summary:
        g = summary["geometry"]
        L += ["## CRAD readings (unchanged instrument)", "",
              f"cos(L,R) median {f(g['cos_LR_median'])} · coactivation {f(g['coactivation_rate_all_states'], 3)} "
              f"· status `{summary['status']}`", "",
              "| draw | label raw | event token | OVERRIDE |", "|---|---|---|---|"]
        for d in summary["draws"]:
            L.append(f"| {d['draw']} | {d['final_label_raw']} | {d['event_global_token']}"
                     f"{' (pseudo)' if d['is_pseudo_event'] else ''} | {d['n_override']} |")
    L += ["", "Next: `.venv/bin/python triad_tokens.py captures/" + cap_id + "`", ""]
    (cd / "CAPTURE.md").write_text("\n".join(L))
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--superset", nargs=2, type=int, default=list(DEFAULT_SUPERSET))
    ap.add_argument("--compare-run", default=None)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if cfg.get("selftest"):
        runmod.die("capture needs a model config, not the selftest config")
    manifest = runmod.preflight(cfg)
    lo, hi = cfg["args"]["layer-lo"], cfg["args"]["layer-hi"]
    s_lo, s_hi = min(a.superset[0], lo), max(a.superset[1], hi)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    cap_id = f"{stamp}_{runmod.run_tag(cfg)}_cap{s_lo}-{s_hi}"
    cd = CAPS / cap_id
    cd.mkdir(parents=True)
    fp = fpmod.collect(LAB, runmod.SRC, cfg, manifest)
    fp["source"]["capture_launcher"] = sha256_file(LAB / "capture_launch.py")
    fp["source"]["capture_driver"] = sha256_file(Path(__file__))
    miss = fpmod.missing_required(fp, needs_gpu=True)
    fp_hash = fpmod.canonical_hash(fp)
    (cd / "fingerprint.json").write_text(json.dumps(fp, indent=2, sort_keys=True) + "\n")
    (cd / "fingerprint.sha256").write_text(fp_hash + "\n")
    (cd / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if miss:
        (cd / "CAPTURE.md").write_text(f"# Capture `{cap_id}`\n\n**REFUSED** — missing fingerprint fields: {miss}\n")
        runmod.die(f"missing fingerprint fields {miss}")

    crad = runmod.cli_args(cfg["args"]) + ["--out", str(cd / "result.jsonl")]
    cmd = [sys.executable, str(LAB / "capture_launch.py"), str(runmod.SRC), str(cd / "runtime.json"),
           str(cd / "capture.npz"), str(s_lo), str(s_hi), "--"] + crad
    (cd / "command.txt").write_text(" ".join(cmd) + "\n")
    print(f"[capture] {cap_id}\n[capture] fingerprint {fp_hash}")
    with open(cd / "stdout.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cd)
        for line in proc.stdout:
            sys.stdout.write(line); log.write(line)
        rc = proc.wait()
    rt = json.loads((cd / "runtime.json").read_text()) if (cd / "runtime.json").exists() else {}

    summary, problems, facts = None, [], {}
    if rc == 0:
        result = json.loads((cd / "result.jsonl").read_text().splitlines()[0])
        summary = safe_extract(result)
        (cd / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        problems = runmod.consistency(cfg, result, rt)
        if manifest and rt.get("model_commit") and rt["model_commit"] != manifest["revision"]:
            problems.append(f"loaded commit {rt['model_commit']} != manifest {manifest['revision']}")
        cap_probs, facts = check_capture(cd / "capture.npz", result, load_v13(runmod.SRC))
        problems += cap_probs
        if a.compare_run:
            problems += compare_to_run(summary, Path(a.compare_run))
        facts["capture_npz_sha256"] = sha256_file(cd / "capture.npz")
    (cd / "capture.json").write_text(json.dumps({"capture_id": cap_id, "superset": [s_lo, s_hi],
                                                 "crad_band": [lo, hi], "problems": problems,
                                                 "facts": facts}, indent=2) + "\n")
    verdict = write_md(cd, cap_id, cfg, fp_hash, rt, summary, facts, problems, rc, a.compare_run)
    idx = CAPS / "INDEX.md"
    if not idx.exists():
        idx.write_text("# CRAD captures index\n\nGenerated by capture.py; never edited by hand.\n\n"
                       "| capture | verdict | config | superset | fingerprint | npz sha256 |\n|---|---|---|---|---|---|\n")
    with open(idx, "a") as fh:
        fh.write(f"| [{cap_id}]({cap_id}/CAPTURE.md) | {verdict} | {cfg['name']} | [{s_lo},{s_hi}) "
                 f"| `{fp_hash[7:19]}` | `{facts.get('capture_npz_sha256', '—')[7:19]}` |\n")
    with open(CAPS / "capture_log.jsonl", "a") as fh:
        fh.write(json.dumps({"ts": stamp, "protocol_version": "0.2", "event": "capture", "capture_id": cap_id,
                             "config": cfg["name"], "instrument_hash": fp_hash, "verdict": verdict,
                             "problems": problems}) + "\n")
    print(f"[capture] {verdict}: {cd}/CAPTURE.md")
    return 0 if verdict == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
