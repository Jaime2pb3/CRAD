#!/usr/bin/env python3
"""Applies EXACTLY the test written in PREREG.md. Nothing else is tested.

Usage: .venv/bin/python analyze_prereg.py

- Data: runs whose config name is qwen2.5-7b_T0.7_L14-20_s43..s46 (seed 42 excluded).
- Feature: n_pre_override, as written by CRAD v1.3 into each run's summary.json.
- Groups: final_label_raw == REFUSE vs final_label_raw == COMPLY (the PREREG
  wording). CRAD's own "label" field only holds COMPLY/NONCOMPLY, so it is NOT
  used for grouping. Other raw labels (e.g. AMBIG) are dropped and reported; a
  non-COMPLY sensitivity view is printed as EXPLORATORY only.
- Refuses to decide if either group is empty or the statistic is not finite.

Revision 2 (2026-09-24): v1 grouped on the "label" field, found no REFUSE draws,
produced a NaN statistic and a spurious "H1 SUPPORTED". v1's output is kept
untouched in analysis/. Hypothesis, feature, statistic and test are unchanged.
- Statistic: mean(REFUSE) - mean(COMPLY), pooled.
- Null: 10000 label permutations within each seed. One-sided p. Alpha 0.05.
Writes analysis/<UTC>_prereg_override_direction.{md,json}.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
SEEDS = [43, 44, 45, 46]
N_PERM = 10000
PERM_RNG_SEED = 20260924
ALPHA = 0.05
SCRIPT_REVISION = 2
V1_SHA256 = "d7177e818398f33dc67cc5956369251036a37efcfafd6a510b463ea67c1ed82b"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runs() -> dict[int, dict]:
    found: dict[int, list] = {s: [] for s in SEEDS}
    for rd in sorted((LAB / "runs").iterdir()):
        cfgp = rd / "config.json"
        if not cfgp.exists():
            continue
        cfg = json.loads(cfgp.read_text())
        for s in SEEDS:
            if cfg.get("name") == f"qwen2.5-7b_T0.7_L14-20_s{s}":
                found[s].append(rd)
    runs = {}
    for s, dirs in found.items():
        if not dirs:
            raise SystemExit(f"[prereg] seed {s}: no run found. Run all four seeds first.")
        if len(dirs) > 1:
            raise SystemExit(f"[prereg] seed {s}: {len(dirs)} runs found {[d.name for d in dirs]}. "
                             "Ambiguous; decide which one counts BEFORE analysing.")
        rd = dirs[0]
        md = (rd / "RUN.md").read_text()
        if "| Verdict | **OK** |" not in md:
            raise SystemExit(f"[prereg] seed {s}: run {rd.name} is not OK. Not analysable.")
        summ = json.loads((rd / "summary.json").read_text())
        if summ["meta"]["seed"] != s:
            raise SystemExit(f"[prereg] seed {s}: summary says seed {summ['meta']['seed']}")
        runs[s] = {"dir": rd.name, "summary": summ}
    return runs


def stratified_test(groups: dict[int, tuple[np.ndarray, np.ndarray]], rng) -> dict:
    """groups[seed] = (values, is_refuse). Returns statistic, p, per-seed usability."""
    vals = np.concatenate([g[0] for g in groups.values()])
    lab = np.concatenate([g[1] for g in groups.values()])
    if lab.sum() == 0 or (~lab).sum() == 0:
        raise SystemExit(f"[prereg] ABORT: empty group (REFUSE={int(lab.sum())}, "
                         f"COMPLY={int((~lab).sum())}). No decision is made.")
    obs = vals[lab].mean() - vals[~lab].mean()
    if not np.isfinite(obs):
        raise SystemExit("[prereg] ABORT: non-finite statistic. No decision is made.")
    usable = {s: bool(g[1].sum() >= 2 and (~g[1]).sum() >= 2) for s, g in groups.items()}
    count = 0
    for _ in range(N_PERM):
        perm_lab = []
        for s, (v, l) in groups.items():
            perm_lab.append(rng.permutation(l) if usable[s] else l)
        pl = np.concatenate(perm_lab)
        if vals[pl].mean() - vals[~pl].mean() >= obs - 1e-12:
            count += 1
    p = (count + 1) / (N_PERM + 1)
    return {"statistic": float(obs), "p_one_sided": float(p), "usable_seeds": usable}


def decide(stat: float, p: float) -> str:
    if not (np.isfinite(stat) and np.isfinite(p)):
        raise SystemExit("[prereg] ABORT: non-finite result. No decision is made.")
    if stat <= 0:
        return "H1 REJECTED (statistic <= 0)"
    if p < ALPHA:
        return "H1 SUPPORTED (statistic > 0, p < 0.05)"
    return "INCONCLUSIVE (statistic > 0, p >= 0.05)"


def main() -> int:
    prereg = LAB / "PREREG.md"
    if not prereg.exists():
        raise SystemExit("[prereg] PREREG.md missing; the test is undefined without it.")
    runs = load_runs()

    table, primary, sens, labels_all = [], {}, {}, Counter()
    for s in SEEDS:
        draws = runs[s]["summary"]["draws"]
        labs = [d["final_label_raw"] for d in draws]
        crad_labs = [d["label"] for d in draws]
        mismatch = sum((a == "COMPLY") != (b == "COMPLY") for a, b in zip(labs, crad_labs))
        if mismatch:
            raise SystemExit(f"[prereg] seed {s}: {mismatch} draws where raw COMPLY "
                             "disagrees with CRAD label COMPLY. Stop and inspect.")
        labels_all.update(labs)
        v = np.array([d["features"]["n_pre_override"] for d in draws], dtype=float)
        keep = np.array([l in ("COMPLY", "REFUSE") for l in labs])
        is_ref = np.array([l == "REFUSE" for l in labs])
        primary[s] = (v[keep], is_ref[keep])
        sens[s] = (v, np.array([l != "COMPLY" for l in labs]))
        c, r = v[np.array(labs) == "COMPLY"], v[np.array(labs) == "REFUSE"]
        table.append({"seed": s, "run": runs[s]["dir"], "n_comply": int(len(c)),
                      "n_refuse": int(len(r)), "n_other": int((~keep).sum()),
                      "mean_comply": float(c.mean()) if len(c) else None,
                      "mean_refuse": float(r.mean()) if len(r) else None})

    rng = np.random.default_rng(PERM_RNG_SEED)
    res = stratified_test(primary, rng)
    verdict = decide(res["statistic"], res["p_one_sided"])
    res_s = stratified_test(sens, np.random.default_rng(PERM_RNG_SEED))

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = {"prereg_sha256": sha(prereg), "analysis_script_sha256": sha(Path(__file__)),
           "script_revision": SCRIPT_REVISION, "superseded_v1_sha256": V1_SHA256,
           "seeds": SEEDS, "n_perm": N_PERM, "perm_rng_seed": PERM_RNG_SEED,
           "labels_seen": dict(labels_all), "per_seed": table,
           "primary": {**res, "verdict": verdict},
           "exploratory_noncomply_vs_comply": res_s}
    adir = LAB / "analysis"
    adir.mkdir(exist_ok=True)
    (adir / f"{stamp}_prereg_override_direction.json").write_text(json.dumps(out, indent=2) + "\n")

    f = lambda x: "—" if x is None else f"{x:.3f}"
    L = ["# Preregistered test — OVERRIDE direction", "",
         f"- PREREG.md sha256: `{out['prereg_sha256']}`",
         f"- analyze_prereg.py sha256: `{out['analysis_script_sha256']}` (revision {SCRIPT_REVISION})",
         f"- Supersedes v1 `{V1_SHA256}`: v1 grouped on CRAD's `label` field (COMPLY/NONCOMPLY),",
         "  found zero REFUSE draws and reported a spurious decision from a NaN statistic.",
         "  Groups now use `final_label_raw`. Hypothesis, feature, statistic and test unchanged.",
         f"- Seeds: {SEEDS} (seed 42 excluded as discovery data)",
         f"- Permutations: {N_PERM} within-seed, rng seed {PERM_RNG_SEED}", "",
         "## Per seed", "", "| seed | run | COMPLY n | REFUSE n | other n | mean COMPLY | mean REFUSE |",
         "|---|---|---|---|---|---|---|"]
    for t in table:
        L.append(f"| {t['seed']} | {t['run']} | {t['n_comply']} | {t['n_refuse']} | {t['n_other']} "
                 f"| {f(t['mean_comply'])} | {f(t['mean_refuse'])} |")
    unusable = [s for s, u in res["usable_seeds"].items() if not u]
    L += ["", "## Primary (confirmatory)", "",
          f"- Statistic mean(REFUSE) − mean(COMPLY): **{res['statistic']:.3f}**",
          f"- One-sided p: **{res['p_one_sided']:.4f}**",
          f"- Seeds without permutations (<2 of a label): {unusable or 'none'}",
          f"- **Decision: {verdict}**", "",
          "## Exploratory only (no claim): non-COMPLY vs COMPLY", "",
          f"- statistic {res_s['statistic']:.3f}, p {res_s['p_one_sided']:.4f}", "",
          f"Labels seen: {dict(labels_all)}", ""]
    (adir / f"{stamp}_prereg_override_direction.md").write_text("\n".join(L))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
