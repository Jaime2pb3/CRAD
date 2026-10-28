#!/usr/bin/env python3
"""Interventional oracle at the drift point: dose-response with a placebo arm.

Usage:
  .venv/bin/python dose_probe.py captures/<capture_id> --turn 6 --direction R
  .venv/bin/python dose_probe.py captures/<capture_id> --turn 7 --direction L
  options: --draw 0 --layer 14 --window 0 4 --doses 0 0.02 0.05 0.1 0.2

What it does
  1. Rebuilds draw <draw> of the capture exactly as crad_base.run_pressure_draw
     does (same seed, same calls), up to the user message of --turn.
  2. Snapshots every RNG, then for each (direction, dose) restores the snapshot
     and generates --turn with a forward hook on decoder layer --layer that adds
         delta = dose * median||h_N,layer|| * u
     to the last position, ONLY on generation steps in --window (step 0 = the
     forward that emits the turn's first token).
  3. Directions are CRAD's OWN neutral-anchored axes at --layer (calibration
     rebuilt from the capture with CRAD's calibrate_from_arrays, i.e. medians):
         R = axis_R = unit(median H_R - median H_N)
         L = axis_L = unit(median H_L - median H_N)
         N = unit(-(axis_L + axis_R))     toward the neutral anchor, away from both
     (v0.1 of this probe used means; the cosine to CRAD's median axes is
     reported in every run so earlier tables can be judged.)
     Every non-zero dose is paired with TWO placebos of the same magnitude:
         placebo    isotropic random direction, orthogonal to axis_L and axis_R
         placeboN   a natural neutral fluctuation: one D_N transition minus v_N
                    (CRAD's own noise model), orthogonalized the same way
  N as the variability reference (how CRAD uses it: v_N residual subtraction,
  energy_floor = median ||D_N||).  Each row reports the injected magnitude in
  two neutral units:
         x step_N   multiples of the typical neutral per-token step at --layer
         x sd_N(u)  multiples of the neutral state spread along that direction
  4. Sanity gate: dose 0 must reproduce the captured tokens of that turn
     exactly when the turn was captured; otherwise the probe aborts.

Readouts per condition
  behaviour   leak yes/no, leak start (token in turn), text head
  propensity  max log p(first secret token) over the window steps and the
              step right after it (raw logits from generate; tokens unchanged)
  CRAD        candidate classes in the window (CRAD's own analyze_frames,
              CRAD's own calibration rebuilt from the capture)
  Triad       mean zE, mean P_layers over the window

CIRCULARITY WARNING
  u lies inside the subspace Triad and CRAD measure, and the residual stream
  carries the injected vector forward.  Readings on steered tokens partly read
  back what was injected.  Behaviour and propensity are the non-circular
  outcomes; instrument readings are only interpretable against the placebo
  row with the same dose.  This is a verification probe, not a test: no PREREG,
  no claim.  Fake secret on a local model; nothing leaves the machine.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from capture import band_frames, load_v13
import run as runmod
from crad_triad import calibrate_triad, change_emotion, layer_polarization


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def captured_turn(z, draw, turn):
    """Tokens and token ids of one captured analysis turn, ordered by row (token_in_turn).

    Added in the Colab rebuild (2026-10-10): patch_probe.py and patch_batch.py import
    it from here, but this copy of dose_probe.py did not define it.  Row k of the
    returned lists is the token PRODUCED by row k (contract clock)."""
    sel = np.flatnonzero((z["kind"] == 1) & (z["draw"] == draw) & (z["turn"] == turn))
    if len(sel) == 0:
        raise SystemExit(f"[capture] draw {draw} turn {turn} was not captured")
    sel = sel[np.argsort(z["token_in_turn"][sel], kind="stable")]
    if z["token_in_turn"][sel].tolist() != list(range(len(sel))):
        raise SystemExit(f"[capture] draw {draw} turn {turn}: token_in_turn not contiguous")
    if np.any(z["token_id"][sel] < 0):
        raise SystemExit(f"[capture] draw {draw} turn {turn}: frames without token id")
    return [str(t) for t in z["token"][sel]], [int(i) for i in z["token_id"][sel]]


def rng_snapshot(torch):
    snap = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        snap["cuda"] = torch.cuda.get_rng_state_all()
    return snap


def rng_restore(torch, snap):
    random.setstate(snap["py"]); np.random.set_state(snap["np"]); torch.set_rng_state(snap["torch"])
    if "cuda" in snap:
        torch.cuda.set_rng_state_all(snap["cuda"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--turn", type=int, required=True)
    ap.add_argument("--direction", choices=["R", "L", "N"], required=True)
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--layer", type=int, default=None, help="default: CRAD band low edge")
    ap.add_argument("--window", nargs=2, type=int, default=[0, 4])
    ap.add_argument("--doses", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--placebo-seed", type=int, default=20261002)
    a = ap.parse_args()

    cd = Path(a.capture_dir)
    cap = json.loads((cd / "capture.json").read_text())
    if cap["problems"]:
        raise SystemExit(f"[dose] capture is not OK: {cap['problems']}")
    cfg = json.loads((cd / "config.json").read_text())
    args = cfg["args"]
    result = json.loads((cd / "result.jsonl").read_text().splitlines()[0])
    with np.load(cd / "capture.npz", allow_pickle=False) as zf:
        z = {k: zf[k] for k in zf.files}
    lo, hi = cap["crad_band"]
    sup_lo, sup_hi = cap["superset"]
    layer = lo if a.layer is None else a.layer
    if not (sup_lo <= layer < sup_hi):
        raise SystemExit(f"[dose] layer {layer} outside captured superset [{sup_lo},{sup_hi})")
    w0, w1 = a.window

    v13 = load_v13(runmod.SRC)
    import crad_base  # importable: load_v13 put src on sys.path

    # calibration populations: CRAD band for readouts, injection layer for directions
    pops = {}
    for code, name in ((0, "N"), (1, "L"), (2, "R")):
        fr = band_frames(z, (z["kind"] == 0) & (z["branch"] == code), lo, hi)
        pops[f"H_{name}"], pops[f"D_{name}"] = v13.stack_states(fr), v13.within_turn_deltas(fr)
    crad_cal = v13.calibrate_from_arrays(pops["H_N"], pops["H_L"], pops["H_R"],
                                         pops["D_N"], pops["D_L"], pops["D_R"], layer_lo=lo, layer_hi=hi)
    tc = calibrate_triad(pops["H_N"], pops["H_L"], pops["H_R"],
                         pops["D_N"], pops["D_L"], pops["D_R"], layer_lo=lo, layer_hi=hi)
    li = layer - sup_lo
    cal_mask = z["kind"] == 0
    Hl = {name: z["phi"][cal_mask & (z["branch"] == code)][:, li].astype(np.float64)
          for code, name in ((0, "N"), (1, "L"), (2, "R"))}
    if lo <= layer < hi:
        uR = unit(crad_cal.axis_R[layer - lo]); uL = unit(crad_cal.axis_L[layer - lo])
        vN = np.asarray(crad_cal.v_N[layer - lo], dtype=np.float64)
    else:  # outside CRAD's band: same construction (medians) at this layer
        uR = unit(np.median(Hl["R"], 0) - np.median(Hl["N"], 0))
        uL = unit(np.median(Hl["L"], 0) - np.median(Hl["N"], 0))
        vN = None
    cos_mean_axes = {"R": float(np.dot(uR, unit(Hl["R"].mean(0) - Hl["N"].mean(0)))),
                     "L": float(np.dot(uL, unit(Hl["L"].mean(0) - Hl["N"].mean(0))))}
    uN = unit(-(uL + uR))
    basis = [uR, unit(uL - np.dot(uL, uR) * uR)]

    def orth_unit(v):
        v = np.asarray(v, dtype=np.float64).copy()
        for b in basis:
            v -= np.dot(v, b) * b
        return unit(v)

    rng = np.random.default_rng(a.placebo_seed)
    u_plac = orth_unit(rng.normal(size=uR.shape))
    # neutral transitions at this layer (CRAD's noise model), rebuilt from the capture
    frN = band_frames(z, cal_mask & (z["branch"] == 0), layer, layer + 1)
    DN = v13.within_turn_deltas(frN)[:, 0]
    if vN is None:
        vN = np.median(DN, axis=0)
    u_placN = orth_unit(DN[int(rng.integers(len(DN)))] - vN)
    scale = float(np.median(np.linalg.norm(Hl["N"], axis=1)))
    step_N = float(np.median(np.linalg.norm(DN, axis=1)))
    centered_N = Hl["N"] - np.median(Hl["N"], 0)

    def sd_N(u):
        proj = centered_N @ u
        return max(float(1.4826 * np.median(np.abs(proj - np.median(proj)))), 1e-9)

    u_main = {"R": uR, "L": uL, "N": uN}[a.direction]

    # model, exactly as CRAD loads it
    v13.seed_everything(int(args["seed"]))
    mw = crad_base.RecogModel(args["model"], use_4bit=not args["no4bit"])
    torch = mw.torch
    ramp = crad_base.build_real_ramp(args["secret"], n_turns=8)
    if not (0 <= a.turn < len(ramp.turns_R)):
        raise SystemExit(f"[dose] --turn must be in [0,{len(ramp.turns_R)})")
    if a.turn < int(args["focus-from-turn"]):
        raise SystemExit("[dose] --turn must be a traced turn (>= focus-from-turn)")
    temp = 0.0 if args["temperature-mode"] == "off" else float(args["temp"])
    if args.get("analysis-temp") is not None:
        temp = float(args["analysis-temp"])
    n_tok = int(args["n-tokens"])

    # the secret's first token depends on what precedes it: " BLUE" after a word,
    # "BLUE" after a quote or newline.  Propensity = max over both variants.
    first_secret_ids = sorted({int(mw.tok(v, add_special_tokens=False)["input_ids"][0])
                               for v in (" " + args["secret"], args["secret"])})
    first_secret_id = first_secret_ids[0]

    # stash scores from generate without changing tokens
    STASH: dict = {}
    orig_generate = mw.lm.generate

    def scoring_generate(*g_args, **g_kwargs):
        # raw logits (before temperature / top-p warpers), so log p is never -inf
        if g_kwargs.get("return_dict_in_generate"):
            g_kwargs["output_logits"] = True
        out = orig_generate(*g_args, **g_kwargs)
        if g_kwargs.get("return_dict_in_generate"):
            STASH["scores"] = [s[0].float().log_softmax(-1).cpu().numpy() for s in out.logits]
        STEP["n"] = 0
        return out

    mw.lm.generate = scoring_generate

    # injection hook on the chosen decoder layer
    STEP = {"n": 0}
    HOOK = {"vec": None}
    decoder_layers = mw.lm.model.layers

    def hook(_mod, _inp, out):
        k = STEP["n"]
        STEP["n"] += 1
        if HOOK["vec"] is None or not (w0 <= k < w1):
            return out
        h = out[0] if isinstance(out, tuple) else out
        h[:, -1, :] = h[:, -1, :] + HOOK["vec"].to(h.dtype)
        return out

    handle = decoder_layers[layer].register_forward_hook(hook)

    # rebuild the draw up to the target user turn (run_pressure_draw, same calls)
    v13.seed_everything(int(args["seed"]) * 100000 + 50000 + a.draw)
    history = [{"role": "system", "content": ramp.system}]
    cursor = 0
    focus = int(args["focus-from-turn"])
    for t in range(a.turn):
        history.append({"role": "user", "content": ramp.turns_R[t]})
        STEP["n"] = 0
        if t < focus:
            text = mw.gen(history, n=n_tok, temp=temp)
        else:
            tr = mw.gen_with_trace(history, n=n_tok, temp=temp, layer_lo=lo, layer_hi=hi,
                                   turn_idx=t, global_start=cursor)
            text = tr["text"]; cursor += len(tr["frames"])
            if crad_base.first_secret_frame(tr["frames"], args["secret"]) is not None:
                raise SystemExit(f"[dose] draw already leaks in turn {t} < --turn {a.turn}")
        history.append({"role": "assistant", "content": text})
    history.append({"role": "user", "content": ramp.turns_R[a.turn]})
    snap = rng_snapshot(torch)

    # captured tokens of this turn, if it was captured (sanity gate)
    sel = np.flatnonzero((z["kind"] == 1) & (z["draw"] == a.draw) & (z["turn"] == a.turn))
    captured = [str(t) for t in z["token"][sel]] if len(sel) else None

    conditions = [("none", 0.0)] if 0.0 in a.doses else []
    for d in a.doses:
        if d != 0.0:
            conditions += [(a.direction, d), ("placebo", d), ("placeboN", d)]
    rows = []
    for name, dose in conditions:
        rng_restore(torch, snap)
        vec = {"none": None, "placebo": u_plac, "placeboN": u_placN}.get(name, u_main)
        HOOK["vec"] = None if vec is None else torch.tensor(dose * scale * vec, device=mw.device)
        STEP["n"] = 0
        tr = mw.gen_with_trace(history, n=n_tok, temp=temp, layer_lo=lo, layer_hi=hi,
                               turn_idx=a.turn, global_start=cursor)
        HOOK["vec"] = None
        frames = tr["frames"]
        toks = [f["token"] for f in frames]
        if name == "none" and captured is not None and toks != captured[:len(toks)]:
            handle.remove()
            raise SystemExit("[dose] ABORT: dose 0 does not reproduce the captured turn. "
                             f"captured head {captured[:6]} vs regenerated {toks[:6]}")
        ev = crad_base.first_secret_frame(frames, args["secret"])
        scores = STASH.get("scores", [])
        prop = [float(max(s[i] for i in first_secret_ids)) for s in scores[w0:w1 + 1]] if scores else []
        an = v13.analyze_frames(
            frames=frames, cal=crad_cal, action_threshold=args["action-threshold"],
            action_layer_frac=args["action-layer-frac"], min_contiguous_layers=args["min-contiguous-layers"],
            baseline_pre_tokens=args["baseline-pre-tokens"], hold_horizon=args["hold-horizon"],
            hold_ratio_min=args["hold-ratio-min"], release_ratio_max=args["release-ratio-max"],
            hold_layer_frac=args["hold-layer-frac"], hold_eps_frac=args["hold-eps-frac"],
            min_baseline_support=args["min-baseline-support"],
            min_baseline_layer_frac=args["min-baseline-layer-frac"], refractory_tokens=args["refractory-tokens"])
        win_g = {int(f["global_token"]) for f in frames[w0:w1]}
        classes = [e.get("classification") for e in an["candidate_events"] if int(e["global_token"]) in win_g]
        zE, P = [], []
        for i in range(w0, min(w1, len(frames))):
            phi = np.asarray(frames[i]["phi"], dtype=np.float64)
            P.append(layer_polarization(phi, tc))
            if i > 0:
                zE.append(change_emotion(phi, np.asarray(frames[i - 1]["phi"], dtype=np.float64), tc)["zE"])
        mag = abs(dose) * scale
        rows.append({
            "condition": name, "dose": dose,
            "x_step_N": mag / step_N if vec is not None else 0.0,
            "x_sd_N": mag / sd_N(vec) if vec is not None else 0.0,
            "leak": ev is not None,
            "leak_token_in_turn": None if ev is None else int(ev["token_in_turn"]),
            "max_logp_first_secret": max(prop) if prop else None,
            "window_crad_classes": classes, "window_mean_zE": float(np.mean(zE)) if zE else None,
            "window_mean_P": float(np.mean(P)) if P else None,
            "text_head": tr["text"][:90],
        })
        r = rows[-1]
        print(f"[dose] {name:8s} {dose:6.3f} xstepN={r['x_step_N']:.2f} xsdN={r['x_sd_N']:.1f} leak={str(r['leak']):5s} start={r['leak_token_in_turn']} "
              f"logp1={r['max_logp_first_secret']} zE={r['window_mean_zE']} P={r['window_mean_P']} "
              f"crad={classes} | {r['text_head']!r}", flush=True)
    handle.remove()

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = cd / "dose_probe" / f"{stamp}_turn{a.turn}_{a.direction}_L{layer}_w{w0}-{w1}"
    out.mkdir(parents=True)
    meta = {"capture": cd.name, "draw": a.draw, "turn": a.turn, "direction": a.direction, "layer": layer,
            "window_steps": [w0, w1], "dose_unit": "fraction of median ||h_N|| at layer",
            "dose_unit_value": scale, "cos_R_L": float(np.dot(uR, uL)), "placebo_seed": a.placebo_seed,
            "axes": "CRAD calibrate_from_arrays (median, neutral-anchored)",
            "cos_to_mean_axes_v01": cos_mean_axes, "neutral_step_at_layer": step_N,
            "neutral_sd_along": {"R": sd_N(uR), "L": sd_N(uL), "N": sd_N(uN),
                                 "placebo": sd_N(u_plac), "placeboN": sd_N(u_placN)},
            "first_secret_token_ids": first_secret_ids, "sanity_gate": "passed" if captured else "turn not captured",
            "rows": rows}
    (out / "dose.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    f = lambda x, n=3: "—" if x is None else f"{x:.{n}f}"
    md = [f"# Dose probe — turn {a.turn}, direction {a.direction}, layer {layer}, steps [{w0},{w1})", "",
          "**Verification probe, not a test. Instrument readings are circular; compare each dose with BOTH "
          "placebo rows (isotropic and neutral-shaped). Behaviour and propensity are the non-circular outcomes.**", "",
          f"Dose unit = {scale:.2f} (median ||h_N|| at layer {layer}); typical neutral step = {step_N:.2f}; "
          f"cos(uR, uL) = {np.dot(uR, uL):.3f}. Axes: CRAD's own (median); cos to the mean axes of v0.1: "
          f"R {cos_mean_axes['R']:.3f}, L {cos_mean_axes['L']:.3f}. Sanity gate: {meta['sanity_gate']}.", "",
          "| condition | dose | × neutral step | × neutral sd along u | leak | start | max log p(secret 1st tok) "
          "| CRAD in window | mean zE | mean P | text |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['condition']} | {r['dose']:.3f} | {r['x_step_N']:.2f} | {r['x_sd_N']:.1f} "
                  f"| {r['leak']} | {r['leak_token_in_turn']} "
                  f"| {f(r['max_logp_first_secret'], 2)} | {', '.join(c for c in r['window_crad_classes'] if c) or '—'} "
                  f"| {f(r['window_mean_zE'], 2)} | {f(r['window_mean_P'], 4)} | {r['text_head'][:50]!r} |")
    (out / "DOSE.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[dose] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
