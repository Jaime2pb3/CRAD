#!/usr/bin/env python3
"""Activation patching at the decision row, v0.1 — bound to crad_contract.

Question: does the model's OWN state at the decision row carry the choice
(TRUTH vs SELF_FICTION), and how much of that causal content lies inside the
span of CRAD's measurement axes {L, R}?  L/R are used only to SPLIT the patch,
never as a direction we inject.

Usage:
  .venv/bin/python patch_probe.py captures/<id> --source 0 --target 2
  .venv/bin/python patch_probe.py captures/<id> --source 2 --target 0      # reverse
  options: --layers 13 14 15 16 17 18 19   --target-turn 7   --anchor "access code is"

Method (one generation of the target turn per condition, same RNG snapshot):
  The target draw is rebuilt exactly as CRAD ran it (run_pressure_draw, same seeds)
  up to its target turn.  At generation step d (the target's decision row, contract
  clock: step d = row d) a forward hook on decoder block l REPLACES the residual
  at the last position:
      none      no hook                                  (sanity: = captured tokens)
      self      h_tgt  (the target's own captured state)  (sanity: = none; proves the
                                                          hook writes where phi is read)
      full      h_src
      LR        h_tgt + P_LR  (h_src - h_tgt)       P_LR  = projector on span{uL, uR}
      orth      h_tgt + (I - P_LR)(h_src - h_tgt)
      rand2     h_tgt + P_rand(h_src - h_tgt)        P_rand = random 2-D subspace (baseline for LR)
  h_src / h_tgt are the CAPTURED states (capture.npz, fp16-exact) of source and target
  at their decision rows, layer l.  phi[t, l] = output of block l = what the hook sees.

Readouts at step d (non-circular, the model's own logits): log p of the secret's
first token, log p of the token the target really chose, top-5; then the contract
outcome class of the whole generated turn.  energy_LR = ||P_LR delta||^2/||delta||^2,
to be read against energy_rand2 (~2/d for a random plane).

Interpretation guide (no claim; single pair, single seed):
  full flips, LR flips, orth does not  -> the choice-relevant content is inside span{L,R}
  full flips, orth flips, LR does not  -> L/R describe behaviour; the mechanism is elsewhere
  full does not flip at any layer      -> the decision is not carried by this position alone
Fake secret on a local model; nothing leaves the machine.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from capture import load_v13
import run as runmod
from crad_contract import (DEFAULT_ANCHOR, DEFAULT_TARGET_TURN, ROW_SEMANTICS, ContractError,
                           SharedCalibration, classify)
from dose_probe import captured_turn, rng_restore, rng_snapshot

VERSION = "0.1-contract"
SELF_TOL = 1e-3          # nats; self-patch must reproduce the unpatched logits


def captured_state(z, draw, turn, row, layer, sup_lo):
    sel = np.flatnonzero((z["kind"] == 1) & (z["draw"] == draw) & (z["turn"] == turn)
                         & (z["token_in_turn"] == row))
    if len(sel) != 1:
        raise ContractError(f"draw {draw} turn {turn} row {row}: {len(sel)} captured frames (need 1)")
    return z["phi"][sel[0], layer - sup_lo].astype(np.float64), str(z["token"][sel[0]])


def plane_projector(u1, u2):
    q, _ = np.linalg.qr(np.stack([u1, u2], axis=1))
    return q          # [d,2]; P x = q @ (q.T @ x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--source", type=int, required=True, help="draw whose state is copied in")
    ap.add_argument("--target", type=int, required=True, help="draw that is regenerated")
    ap.add_argument("--layers", nargs="+", type=int, default=None, help="default: whole captured superset")
    ap.add_argument("--target-turn", type=int, default=DEFAULT_TARGET_TURN)
    ap.add_argument("--anchor", default=DEFAULT_ANCHOR)
    ap.add_argument("--rand-seed", type=int, default=20261004)
    a = ap.parse_args()

    cd = Path(a.capture_dir)
    v13 = load_v13(runmod.SRC)
    import crad_base
    sc = SharedCalibration.from_capture(cd, v13)
    z, args = sc.z, sc.args
    sup_lo, sup_hi = sc.superset
    lo, hi = sc.band
    T, n_tok, focus = a.target_turn, int(args["n-tokens"]), int(args["focus-from-turn"])
    layers = list(range(sup_lo, sup_hi)) if a.layers is None else a.layers
    for l in layers:
        if not (sup_lo <= l < sup_hi):
            raise ContractError(f"layer {l} outside captured superset [{sup_lo},{sup_hi})")

    # ---- decisions (contract) and captured states: all checked before loading the model
    ds, dt = sc.decision(a.source, T, a.anchor), sc.decision(a.target, T, a.anchor)
    for name, d in (("source", ds), ("target", dt)):
        if d.row is None:
            raise ContractError(f"{name} draw has no decision row in turn {T} ({d.outcome})")
    if ds.outcome == dt.outcome:
        print(f"[patch] WARNING: source and target share outcome {ds.outcome}; a flip cannot be observed")
    cap_T_toks, cap_T_ids = captured_turn(z, a.target, T)
    orig_id = cap_T_ids[dt.row]
    rng = np.random.default_rng(a.rand_seed)
    geo = {}
    for l in layers:
        h_src, tok_s = captured_state(z, a.source, T, ds.row, l, sup_lo)
        h_tgt, tok_t = captured_state(z, a.target, T, dt.row, l, sup_lo)
        ax = sc.axes_at_layer(l)
        Q = plane_projector(ax["uL"], ax["uR"])
        Qr = plane_projector(rng.normal(size=h_src.shape), rng.normal(size=h_src.shape))
        delta = h_src - h_tgt
        p_lr, p_r = Q @ (Q.T @ delta), Qr @ (Qr.T @ delta)
        nd = float(delta @ delta)
        geo[l] = {"h_src": h_src, "h_tgt": h_tgt, "p_lr": p_lr, "p_rand": p_r, "in_band": ax["in_band"],
                  "delta_norm": nd ** 0.5, "rel_delta": nd ** 0.5 / max(np.linalg.norm(h_tgt), 1e-12),
                  "energy_LR": float(p_lr @ p_lr) / max(nd, 1e-12),
                  "energy_rand2": float(p_r @ p_r) / max(nd, 1e-12)}

    # ---- model, exactly as CRAD loads it
    v13.seed_everything(int(args["seed"]))
    mw = crad_base.RecogModel(args["model"], use_4bit=not args["no4bit"])
    torch = mw.torch
    ramp = crad_base.build_real_ramp(args["secret"], n_turns=8)
    temp = 0.0 if args["temperature-mode"] == "off" else float(args["temp"])
    if args.get("analysis-temp") is not None:
        temp = float(args["analysis-temp"])
    first_secret_ids = sorted({int(mw.tok(v, add_special_tokens=False)["input_ids"][0])
                               for v in (" " + args["secret"], args["secret"])})

    STASH, STEP, HOOK = {}, {"n": 0}, {"layer": None, "vec": None}
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
            if HOOK["vec"] is None or k != dt.row:
                return out
            h = out[0] if isinstance(out, tuple) else out
            h[:, -1, :] = HOOK["vec"].to(h.dtype)          # REPLACE, not add
            return out
        return hook

    handles = [mw.lm.model.layers[l].register_forward_hook(make_hook(l)) for l in layers]

    # ---- rebuild the target draw up to its target turn (run_pressure_draw, same calls)
    v13.seed_everything(int(args["seed"]) * 100000 + 50000 + a.target)
    history = [{"role": "system", "content": ramp.system}]
    texts, cursor = {}, 0
    for t in range(T):
        history.append({"role": "user", "content": ramp.turns_R[t]})
        if t < focus:
            text = mw.gen(history, n=n_tok, temp=temp)
        else:
            tr = mw.gen_with_trace(history, n=n_tok, temp=temp, layer_lo=lo, layer_hi=hi,
                                   turn_idx=t, global_start=cursor)
            text = tr["text"]; cursor += len(tr["frames"]); texts[t] = text
        history.append({"role": "assistant", "content": text})
    history.append({"role": "user", "content": ramp.turns_R[T]})
    snap = rng_snapshot(torch)

    def run(layer, vec):
        rng_restore(torch, snap)
        HOOK["layer"], STEP["n"] = layer, 0
        HOOK["vec"] = None if vec is None else torch.tensor(vec, device=mw.device)
        tr = mw.gen_with_trace(history, n=n_tok, temp=temp, layer_lo=lo, layer_hi=hi,
                               turn_idx=T, global_start=cursor)
        HOOK["layer"], HOOK["vec"] = None, None
        toks = [f["token"] for f in tr["frames"]]
        sc_d = STASH["scores"][dt.row]
        top = np.argsort(sc_d)[::-1][:5]
        dec = classify({**{t: [x] for t, x in texts.items()}, T: toks}, args["secret"], T, a.anchor)
        return {"toks": toks, "outcome": dec.outcome, "content": dec.content,
                "lp_secret": float(max(sc_d[i] for i in first_secret_ids)),
                "lp_orig": float(sc_d[orig_id]),
                "top5": [(mw.tok.decode([int(i)]), round(float(sc_d[i]), 3)) for i in top]}

    base = run(None, None)
    if base["toks"] != cap_T_toks[:len(base["toks"])]:
        for h in handles:
            h.remove()
        raise ContractError(f"unpatched run does not reproduce captured turn {T}: "
                            f"{cap_T_toks[:6]} vs {base['toks'][:6]}")
    print(f"[patch] baseline {base['outcome']} {base['content']} lp_secret={base['lp_secret']:.3f} "
          f"lp_orig={base['lp_orig']:.3f} top5={base['top5']}", flush=True)

    rows = []
    for l in layers:
        g = geo[l]
        selfp = run(l, g["h_tgt"])
        if abs(selfp["lp_secret"] - base["lp_secret"]) > SELF_TOL or abs(selfp["lp_orig"] - base["lp_orig"]) > SELF_TOL:
            for h in handles:
                h.remove()
            raise ContractError(f"layer {l}: self-patch changes the logits ({selfp['lp_secret']:.4f} vs "
                                f"{base['lp_secret']:.4f}); the hook does not write where phi is read")
        conds = {"full": g["h_src"], "LR": g["h_tgt"] + g["p_lr"],
                 "orth": g["h_tgt"] + (g["h_src"] - g["h_tgt"] - g["p_lr"]), "rand2": g["h_tgt"] + g["p_rand"]}
        for mode, vec in conds.items():
            r = run(l, vec)
            row = {"layer": l, "in_band": g["in_band"], "mode": mode, "outcome": r["outcome"],
                   "content": r["content"], "flip": r["outcome"] != base["outcome"],
                   "lp_secret": r["lp_secret"], "d_lp_secret": r["lp_secret"] - base["lp_secret"],
                   "lp_orig": r["lp_orig"], "d_lp_orig": r["lp_orig"] - base["lp_orig"], "top5": r["top5"],
                   "energy_LR": g["energy_LR"], "energy_rand2": g["energy_rand2"],
                   "rel_delta": g["rel_delta"], "self_patch_ok": True}
            rows.append(row)
            print(f"[patch] L{l:<2} {mode:5s} {r['outcome']:12s} {str(r['content'])[:14]:14s} "
                  f"dlp_secret={row['d_lp_secret']:+.3f} dlp_orig={row['d_lp_orig']:+.3f} "
                  f"top1={r['top5'][0]}", flush=True)
    for h in handles:
        h.remove()

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = cd / "patch_probe" / f"{stamp}_src{a.source}_tgt{a.target}"
    out.mkdir(parents=True)
    meta = {"probe_version": VERSION, **sc.stamp(), "capture": cd.name, "target_turn": T, "anchor": a.anchor,
            "source": ds.as_dict() | {"draw": a.source}, "target": dt.as_dict() | {"draw": a.target},
            "baseline": {k: v for k, v in base.items() if k != "toks"}, "first_secret_token_ids": first_secret_ids,
            "original_token_id": int(orig_id), "rand_seed": a.rand_seed, "self_tol": SELF_TOL, "rows": rows}
    (out / "patch.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    md = [f"# Patch probe v{VERSION} — source draw {a.source} ({ds.outcome} {ds.content}) → "
          f"target draw {a.target} ({dt.outcome} {dt.content}), decision row {dt.row} of turn {T}", "",
          f"calibration_id `{sc.calibration_id}` · rulers {sc.agreement}", "",
          f"Clock: {ROW_SEMANTICS}", "",
          f"Baseline: {base['outcome']} {base['content']} · log p(secret) {base['lp_secret']:.3f} · "
          f"log p(original) {base['lp_orig']:.3f} · top5 {base['top5']}", "",
          "Self-patch passed at every layer (hook writes where phi is read).", "",
          "**Single pair, single seed: a probe, not a test. energy_LR must be read against energy_rand2.**", "",
          "| layer | mode | outcome | content | flip | Δ log p secret | Δ log p original | top1 | energy LR | energy rand2 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['layer']}{'' if r['in_band'] else '*'} | {r['mode']} | {r['outcome']} | {r['content']} "
                  f"| {'**yes**' if r['flip'] else 'no'} | {r['d_lp_secret']:+.2f} | {r['d_lp_orig']:+.2f} "
                  f"| {r['top5'][0][0]!r} | {r['energy_LR']:.4f} | {r['energy_rand2']:.5f} |")
    md += ["", "\\* off-band layer: L/R by the contract's median construction on the capture superset."]
    (out / "PATCH.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[patch] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
