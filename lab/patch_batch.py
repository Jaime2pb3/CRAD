#!/usr/bin/env python3
"""Batch activation patching at the token level, v0.1 — one model load, every pair.

Separates the two components of the decision seen in patch_probe:
  SOURCE   does the model answer from the system prompt (TRUTH) or from its own
           narrative (SELF_FICTION)?  -> patch at the ANCHOR row
  CONTENT  once in the narrative, WHICH code?  -> patch at the COMMIT row
           (first row carrying content; = anchor row unless the model first
           emitted ' "' or ' ')
and tests whether a direction built from the model's OWN decision states carries
the source choice (leave-one-pair-out, so the tested pair never builds its axis).

Usage:
  .venv/bin/python patch_batch.py captures/<id>                      # all TRUTH x SELF_FICTION pairs, both ways
  .venv/bin/python patch_batch.py captures/<id> --layers 17 19 21 23 25 --align anchor commit

Modes at layer l, row k of the target (k = anchor or commit row, token level):
  full   h_src
  nat1   h_tgt + u u^T (h_src - h_tgt)    u = unit(median h_TRUTH - median h_SF) at (l, align),
                                          built WITHOUT the tested source and target draws
  rand1  h_tgt + r r^T (h_src - h_tgt)    r random unit   (baseline for nat1)
  LR     h_tgt + P_LR (h_src - h_tgt)     CRAD's measurement plane, read-only use
  rand2  h_tgt + P_rand2 (h_src - h_tgt)  random plane    (baseline for LR)

Gates: the unpatched run must reproduce the captured target turn; a self-patch
(target's own captured state) must leave the logits within 1e-3 nats at every
(target, layer, align).  A self-patch failure aborts, except at the model's LAST
layer, which is excluded with a note: HF returns the last hidden state AFTER the
final norm, so phi there is not the block output the hook replaces.

Primary readout (token level, the model's own logits at the patched step):
  toward = sign * delta log p(secret first token),  sign = +1 if the source is
  TRUTH, -1 if SELF_FICTION; i.e. how far the patch moved the target toward the
  source's choice.  recovered = toward(mode) / toward(full) per pair.
Secondary: outcome class and content origin (SOURCE content / TARGET's own fiction
/ TRUTH / OTHER).  Flips at T>0 sit on a fixed random draw: report, don't lean on.
Stats: paired sign-flip permutation over pairs (nat1 vs rand1, LR vs rand2).
Single seed / single capture: a probe, not a test.  Fake secret; local model.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from capture import load_v13
import run as runmod
from crad_contract import (DEFAULT_ANCHOR, DEFAULT_TARGET_TURN, ROW_SEMANTICS, ContractError,
                           SharedCalibration, classify, norm_alnum)
from dose_probe import captured_turn, rng_restore, rng_snapshot
from patch_probe import captured_state, plane_projector

VERSION = "0.1-contract"
SELF_TOL = 1e-3
MODES = ("full", "nat1", "rand1", "LR", "rand2")


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def sign_flip_p(x: np.ndarray, n: int, rng) -> float:
    """Two-sided paired sign-flip test of mean(x) == 0."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    obs = abs(x.mean())
    s = rng.choice([-1.0, 1.0], size=(n, len(x)))
    return float((np.sum(np.abs((s * x).mean(1)) >= obs - 1e-12) + 1) / (n + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--layers", nargs="+", type=int, default=None, help="default: whole captured superset")
    ap.add_argument("--align", nargs="+", choices=["anchor", "commit"], default=["anchor", "commit"])
    ap.add_argument("--target-turn", type=int, default=DEFAULT_TARGET_TURN)
    ap.add_argument("--anchor", default=DEFAULT_ANCHOR)
    ap.add_argument("--one-way", action="store_true", help="only TRUTH -> SELF_FICTION")
    ap.add_argument("--rand-seed", type=int, default=20261004)
    ap.add_argument("--n-perm", type=int, default=10000)
    a = ap.parse_args()

    cd = Path(a.capture_dir)
    v13 = load_v13(runmod.SRC)
    import crad_base
    sc = SharedCalibration.from_capture(cd, v13)
    z, args = sc.z, sc.args
    sup_lo, sup_hi = sc.superset
    lo, hi = sc.band
    T, n_tok, focus = a.target_turn, int(args["n-tokens"]), int(args["focus-from-turn"])
    layers = list(range(sup_lo, sup_hi)) if a.layers is None else sorted(a.layers)
    for l in layers:
        if not (sup_lo <= l < sup_hi):
            raise ContractError(f"layer {l} outside captured superset [{sup_lo},{sup_hi})")

    # ---- decisions (contract, token level)
    n_draws = int(z["draw"][z["kind"] == 1].max()) + 1
    dec = {k: sc.decision(k, T, a.anchor) for k in range(n_draws)}
    truth = [k for k, d in dec.items() if d.outcome == "TRUTH" and d.commit_row is not None]
    fic = [k for k, d in dec.items() if d.outcome == "SELF_FICTION" and d.commit_row is not None]
    if len(truth) < 3 or len(fic) < 3:
        raise ContractError(f"need >= 3 TRUTH and >= 3 SELF_FICTION draws for leave-one-out "
                            f"(have {len(truth)}, {len(fic)})")
    rowk = {(k, al): (dec[k].row if al == "anchor" else dec[k].commit_row) for k in truth + fic for al in a.align}
    pairs = [(s, t) for s in truth for t in fic] + ([] if a.one_way else [(f, s) for s in truth for f in fic])
    print(f"[batch] TRUTH {truth}  SELF_FICTION {fic}  pairs {len(pairs)}  layers {layers}  align {a.align}")

    # ---- captured states at the decision rows (all from the capture, fp16-exact)
    H = {(k, al, l): captured_state(z, k, T, rowk[(k, al)], l, sup_lo)[0]
         for k in truth + fic for al in a.align for l in layers}

    def nat_axis(al, l, drop):
        tr = [H[(k, al, l)] for k in truth if k not in drop]
        fi = [H[(k, al, l)] for k in fic if k not in drop]
        return unit(np.median(tr, 0) - np.median(fi, 0))

    # stability of the native axis across leave-one-pair-out folds (few draws in 3584 dims:
    # a noisy axis makes a null nat1 uninformative, so report it next to the result)
    axis_stability = {}
    for al in a.align:
        for l in layers:
            us = [nat_axis(al, l, {s_, t_}) for s_ in truth for t_ in fic]
            G = np.array([[float(x @ y) for y in us] for x in us])
            axis_stability[(al, l)] = float(G[np.triu_indices(len(us), 1)].mean())

    rng = np.random.default_rng(a.rand_seed)
    planes = {}
    for l in layers:
        ax = sc.axes_at_layer(l)
        d = H[(truth[0], a.align[0], l)].shape[0]
        planes[l] = {"LR": plane_projector(ax["uL"], ax["uR"]),
                     "rand2": plane_projector(rng.normal(size=d), rng.normal(size=d)),
                     "rand1": unit(rng.normal(size=d)), "in_band": ax["in_band"]}

    # ---- model, exactly as CRAD loads it
    v13.seed_everything(int(args["seed"]))
    mw = crad_base.RecogModel(args["model"], use_4bit=not args["no4bit"])
    torch = mw.torch
    n_layers = int(mw.lm.config.num_hidden_layers)
    ramp = crad_base.build_real_ramp(args["secret"], n_turns=8)
    temp = 0.0 if args["temperature-mode"] == "off" else float(args["temp"])
    if args.get("analysis-temp") is not None:
        temp = float(args["analysis-temp"])
    secret_ids = sorted({int(mw.tok(v, add_special_tokens=False)["input_ids"][0])
                         for v in (" " + args["secret"], args["secret"])})

    STASH, STEP, HOOK = {}, {"n": 0}, {"layer": None, "vec": None, "row": None}
    orig_generate = mw.lm.generate

    def scoring_generate(*g_args, **g_kwargs):
        if g_kwargs.get("return_dict_in_generate"):
            g_kwargs["output_logits"] = True
        out = orig_generate(*g_args, **g_kwargs)
        if g_kwargs.get("return_dict_in_generate"):
            STASH["scores"] = [s[0].float().log_softmax(-1).cpu().numpy() for s in out.logits]
        return out

    mw.lm.generate = scoring_generate

    def make_hook(l):
        def hook(_mod, _inp, out):
            if HOOK["layer"] != l:
                return out
            k = STEP["n"]
            STEP["n"] += 1
            if HOOK["vec"] is None or k != HOOK["row"]:
                return out
            h = out[0] if isinstance(out, tuple) else out
            h[:, -1, :] = HOOK["vec"].to(h.dtype)            # REPLACE at one row, one layer
            return out
        return hook

    handles = [mw.lm.model.layers[l].register_forward_hook(make_hook(l)) for l in layers]

    # ---- rebuild every target once (run_pressure_draw, same seeds)
    ctx = {}
    for k in sorted(set(t for _, t in pairs)):
        v13.seed_everything(int(args["seed"]) * 100000 + 50000 + k)
        hist, texts, cur = [{"role": "system", "content": ramp.system}], {}, 0
        for t in range(T):
            hist.append({"role": "user", "content": ramp.turns_R[t]})
            if t < focus:
                text = mw.gen(hist, n=n_tok, temp=temp)
            else:
                tr = mw.gen_with_trace(hist, n=n_tok, temp=temp, layer_lo=lo, layer_hi=hi,
                                       turn_idx=t, global_start=cur)
                text = tr["text"]; cur += len(tr["frames"]); texts[t] = text
            hist.append({"role": "assistant", "content": text})
        hist.append({"role": "user", "content": ramp.turns_R[T]})
        ctx[k] = {"hist": hist, "texts": texts, "cursor": cur, "snap": rng_snapshot(torch)}

    def gen(k, layer, row, vec):
        c = ctx[k]
        rng_restore(torch, c["snap"])
        HOOK.update(layer=layer, row=row, vec=None if vec is None else torch.tensor(vec, device=mw.device))
        STEP["n"] = 0
        tr = mw.gen_with_trace(c["hist"], n=n_tok, temp=temp, layer_lo=lo, layer_hi=hi,
                               turn_idx=T, global_start=c["cursor"])
        HOOK.update(layer=None, row=None, vec=None)
        toks = [f["token"] for f in tr["frames"]]
        return toks, STASH["scores"]

    def readout(k, toks, scores, step, orig_id):
        d = classify({**{t: [x] for t, x in ctx[k]["texts"].items()}, T: toks}, args["secret"], T, a.anchor)
        s = scores[step] if step < len(scores) else None
        return {"outcome": d.outcome, "content": d.content,
                "lp_secret": None if s is None else float(max(s[i] for i in secret_ids)),
                "lp_orig": None if s is None else float(s[orig_id])}

    # baselines and self-patch gates
    base, excluded = {}, []
    for k in ctx:
        toks, scores = gen(k, None, None, None)
        cap_toks, cap_ids = captured_turn(z, k, T)
        if toks != cap_toks[:len(toks)]:
            raise ContractError(f"target {k}: unpatched run does not reproduce captured turn {T}")
        for al in a.align:
            r = rowk[(k, al)]
            base[(k, al)] = readout(k, toks, scores, r, cap_ids[r]) | {"orig_id": cap_ids[r]}
    for k in ctx:
        for al in a.align:
            for l in layers:
                if (l, al) in excluded:
                    continue
                toks, scores = gen(k, l, rowk[(k, al)], H[(k, al, l)])
                r = readout(k, toks, scores, rowk[(k, al)], base[(k, al)]["orig_id"])
                b = base[(k, al)]
                if abs(r["lp_secret"] - b["lp_secret"]) > SELF_TOL or abs(r["lp_orig"] - b["lp_orig"]) > SELF_TOL:
                    if l == n_layers - 1:
                        excluded.append((l, al))
                        print(f"[batch] layer {l} excluded ({al}): self-patch mismatch at the LAST layer "
                              "(HF normalizes the last hidden state)")
                        continue
                    raise ContractError(f"target {k} layer {l} {al}: self-patch changes logits "
                                        f"({r['lp_secret']:.4f} vs {b['lp_secret']:.4f})")
    print(f"[batch] gates passed for {len(ctx)} targets; excluded {excluded or 'none'}", flush=True)

    # ---- the batch
    rows = []
    for (s, t) in pairs:
        sgn = 1.0 if dec[s].outcome == "TRUTH" else -1.0
        for al in a.align:
            for l in layers:
                if (l, al) in excluded:
                    continue
                hs, ht = H[(s, al, l)], H[(t, al, l)]
                delta = hs - ht
                u = nat_axis(al, l, drop={s, t})
                P = planes[l]
                vecs = {"full": hs, "nat1": ht + u * (u @ delta), "rand1": ht + P["rand1"] * (P["rand1"] @ delta),
                        "LR": ht + P["LR"] @ (P["LR"].T @ delta), "rand2": ht + P["rand2"] @ (P["rand2"].T @ delta)}
                b = base[(t, al)]
                for m in MODES:
                    toks, scores = gen(t, l, rowk[(t, al)], vecs[m])
                    r = readout(t, toks, scores, rowk[(t, al)], b["orig_id"])
                    c = r["content"]
                    nc = None if c is None else norm_alnum(c)
                    if nc is not None and nc == norm_alnum(args["secret"]):
                        origin = "TRUTH"
                    elif nc is not None and dec[s].content and nc == norm_alnum(dec[s].content):
                        origin = "SOURCE"             # the source's own code was transplanted
                    elif r["outcome"] == "SELF_FICTION":
                        origin = "TARGET_OWN"         # a code from the TARGET's own narrative
                    else:
                        origin = "OTHER"
                    toward = sgn * (r["lp_secret"] - b["lp_secret"])
                    rows.append({"source": s, "target": t, "dir": "T->F" if sgn > 0 else "F->T", "align": al,
                                 "row": rowk[(t, al)], "layer": l, "mode": m, "outcome": r["outcome"],
                                 "content": c, "origin": origin, "flip": r["outcome"] != b["outcome"],
                                 "d_lp_secret": r["lp_secret"] - b["lp_secret"],
                                 "d_lp_orig": r["lp_orig"] - b["lp_orig"], "toward": toward,
                                 "energy": (float(u @ delta) ** 2 / max(float(delta @ delta), 1e-12)) if m == "nat1" else None})
                print(f"[batch] {s:>2}->{t:<2} {al:6s} L{l:<2} " + " ".join(
                    f"{x['mode']}={x['toward']:+.2f}" for x in rows[-len(MODES):]) +
                      f" | full:{rows[-len(MODES)]['origin']}", flush=True)
    for h in handles:
        h.remove()

    # ---- aggregate
    perm_rng = np.random.default_rng(a.rand_seed + 1)
    by = defaultdict(dict)
    for r in rows:
        by[(r["dir"], r["align"], r["layer"], r["source"], r["target"])][r["mode"]] = r
    summary = []
    for key in sorted({(d, al, l) for d, al, l, _, _ in by}):
        cells = [v for k, v in by.items() if k[:3] == key]
        tw = {m: np.array([c[m]["toward"] for c in cells]) for m in MODES}
        full = tw["full"]
        rec = {m: float(np.nanmedian(np.where(np.abs(full) > 0.05, tw[m] / np.where(full == 0, 1, full), np.nan)))
               if np.any(np.abs(full) > 0.05) else None for m in ("nat1", "LR")}
        orig_full = Counter(c["full"]["origin"] for c in cells)
        flips = {m: int(sum(c[m]["flip"] for c in cells)) for m in MODES}
        summary.append({"dir": key[0], "align": key[1], "layer": key[2], "n_pairs": len(cells),
                        **{f"mean_toward_{m}": float(tw[m].mean()) for m in MODES},
                        "median_recovered_nat1": rec["nat1"], "median_recovered_LR": rec["LR"],
                        "p_nat1_vs_rand1": sign_flip_p(tw["nat1"] - tw["rand1"], a.n_perm, perm_rng),
                        "p_LR_vs_rand2": sign_flip_p(tw["LR"] - tw["rand2"], a.n_perm, perm_rng),
                        "flips": flips, "full_content_origin": dict(orig_full),
                        "mean_energy_nat1": float(np.mean([c["nat1"]["energy"] for c in cells])),
                        "nat_axis_fold_cos": axis_stability[(key[1], key[2])]})

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = cd / "patch_batch" / stamp
    out.mkdir(parents=True)
    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    meta = {"batch_version": VERSION, **sc.stamp(), "capture": cd.name, "target_turn": T, "anchor": a.anchor,
            "truth": truth, "self_fiction": fic, "decisions": {k: dec[k].as_dict() for k in truth + fic},
            "layers": layers, "align": a.align, "excluded": excluded, "secret_token_ids": secret_ids,
            "rand_seed": a.rand_seed, "n_perm": a.n_perm, "summary": summary}
    (out / "summary.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    f = lambda x, n=2: "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"
    md = [f"# Patch batch v{VERSION} — `{cd.name}`", "",
          f"calibration_id `{sc.calibration_id}` · TRUTH {truth} · SELF_FICTION {fic} · excluded {excluded or 'none'}", "",
          f"Clock: {ROW_SEMANTICS}", "",
          "toward = movement of log p(secret first token) toward the SOURCE's choice at the patched row "
          "(+ = toward the source). recovered = mode / full (median over pairs, |full| > 0.05). "
          "nat1 = 1-D axis from the model's own decision states, leave-one-pair-out. "
          "**Single seed, single capture: a probe, not a test.**", "",
          "| dir | align | layer | n | full | nat1 | rand1 | LR | rand2 | rec nat1 | rec LR | p nat1>rand1 "
          "| p LR>rand2 | flips full/nat1/LR | full content origin | nat axis fold cos |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s_ in summary:
        md.append(f"| {s_['dir']} | {s_['align']} | {s_['layer']} | {s_['n_pairs']} "
                  + " ".join(f"| {f(s_['mean_toward_' + m])}" for m in MODES)
                  + f" | {f(s_['median_recovered_nat1'])} | {f(s_['median_recovered_LR'])} "
                  f"| {f(s_['p_nat1_vs_rand1'], 4)} | {f(s_['p_LR_vs_rand2'], 4)} "
                  f"| {s_['flips']['full']}/{s_['flips']['nat1']}/{s_['flips']['LR']} | {s_['full_content_origin']} "
                  f"| {f(s_['nat_axis_fold_cos'])} |")
    md += ["", "Content origin (full patch): SOURCE = the source's own code was transplanted; "
           "TARGET_OWN = the target fell back to ITS OWN fiction; TRUTH; OTHER. "
           "SOURCE-dominant at the commit row but TARGET_OWN at the anchor row would mean: "
           "the anchor row carries the source choice, the commit row binds the content.",
          "nat axis fold cos: mean cosine between leave-one-out axes. Near 1 = the axis is a stable property "
          "of the decision states; low = few draws in high dimension, and a null nat1 is NOT evidence of absence."]
    (out / "BATCH.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[batch] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
