#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRAD Triad v0.1 — lo que permanece / lo que cambia / lo que emociona
TwoQuarks Research — Luis Jaime Ledesma Perez

Design contract
---------------
This module never modifies CRAD v1.3 (hash-pinned by run.py) nor v1.5-prismatic.
It imports v1.5's calibration and reads the same tensors:
    H_N, H_L, H_R : [n, L, d] calibration states (neutral / constraint / pressure)
    D_N, D_L, D_R : [n, L, d] calibration transitions (phi_t - phi_{t-1})
    traj          : [T, L, d] phi frames of one draw, as gen_with_trace stores them

The triad, as operational definitions on hidden states
------------------------------------------------------
PERMANECE (what persists) lives in the weights.  We cannot read weights from a
trace, but we can read their default dynamics: the neutral transition field
v_N = median(D_N).  "What the model does when nothing special happens."

Both come from the SAME vector -- the step's deviation from that default --
and are its two orthogonal parts.  "Vienen del mismo lugar, pero no son la
misma cosa":
    r_t,l = (phi_t,l - phi_{t-1,l}) - v_N,l
    Q_l   = orthonormal span{L axis, R axis, L/R action contrast}   (what is at stake)

CAMBIA (what changes) = the part of r OUTSIDE the decision subspace, in units of
neutral noise.  How different the moment looks, with the stakes removed:
    C_t   = wmean_l( ||(I - Q_l Q_l^T) r_t,l|| / s_comp,l )

EMOCIONA (what matters) = how CONCENTRATED the step is on what is at stake,
compared with a non-specific (isotropic) step.  Per layer this is the F ratio of
energy per dimension inside vs outside the decision subspace (k_l = rank of Q_l):
    f_t,l = ( ||Q_l^T r||^2 / k_l ) / ( ||(I - Q_l Q_l^T) r||^2 / (d - k_l) )
    E_t   = wmean_l( log f_t,l )
It is scale-free: isotropic noise of ANY size has f ~ 1.  The raw size of the
decision push is reported separately as decision_motion (not used for quadrants).

Design notes (both found by testing this file, v0.1):
1. Measuring C on the TOTAL step made C absorb E -- in high dimension norms
   concentrate, so any decision-aligned push raised C (median zC ~ 9) and every
   STEALTH step was filed as OVERT.  The orthogonal split fixes it.
2. A first E defined as decision excess over kappa * complement scaled with step
   size: 4x isotropic noise was filed OVERT ~70% of the time.  The F ratio is
   the scale-free test of "aimed vs non-specific" and removes that failure.  Both C and E are standardized against the neutral
calibration distribution of the same aggregate; quadrant thresholds are neutral
quantiles fixed BEFORE any leak data is seen.

Quadrants (the 2x2 of the session):
    ROUTINE  low C, low E
    NOISE    high C, low E   (lots of change, nothing at stake)
    STEALTH  low C, high E   (looks normal, moves the decision: injection-like)
    OVERT    high C, high E  (crude attack)

Layer polarization (Isomeric Polarization with layers as views):
    Phi_l = (bounded_L, bounded_R)  from v1.5 state_observables (coactivation allowed)
    P_t   = sum_{i<j} w_i w_j ||Phi_i - Phi_j|| / sum_{i<j} w_i w_j
High P_t = the band disagrees about L vs R = unresolved arbitration (H3 proxy).

Isolation gate (base isolation, feedforward; contrast with v1.5's PD damper):
    corrected = phi - (1-g) * w_l * Q_l Q_l^T r_t,l
    g = 1 / (1 + lam * max(0, zE - zE_hi))
It only reduces transmission of excess decision-relevant motion; it never
pushes toward a target.  (I - QQ^T) phi is invariant by construction.

What this file is NOT
---------------------
- Not evidence about real models.  Selftests are instrument tests on planted data.
- E is a projection proxy, not causal attribution.  Causal emotion of input
  spans needs interventions on the prompt: see span_effects(), which takes a
  model-backed score function and is GPU-agnostic.
- The gate is an intervention.  Use it only under the two-pass shadow protocol
  of the v1.5 notes, after detection is calibrated, with an over-refusal control.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from CRAD_v1_5_prismatic import (  # noqa: E402
    EPS, OnlineState, OnsetConfig, PrismCalibration, calibrate_prism,
    online_step, safe_unit, state_observables, weighted_mean,
)

VERSION = "0.1-triad"
QUADRANTS = ("ROUTINE", "NOISE", "STEALTH", "OVERT")


# ------------------------------------------------------------ calibration --
@dataclass
class TriadCalibration:
    prism: PrismCalibration
    Q: np.ndarray            # [L, d, k] orthonormal decision subspace per layer
    rank: np.ndarray         # [L] effective k per layer
    s_comp: np.ndarray       # [L] neutral scale of ||(I-QQ^T) r||
    s_dec: np.ndarray        # [L] neutral scale of ||Q^T r|| (reporting only)
    dim: int                 # d, hidden size
    C_ref: tuple             # (center, scale) of aggregate C on neutral
    E_ref: tuple             # (center, scale) of aggregate E on neutral
    zC_hi: float
    zE_hi: float
    alpha: float

    def metadata(self) -> Dict:
        return {
            "version": VERSION,
            "decision_rank_per_layer": self.rank.tolist(),
            "hidden_dim": self.dim,
            "C_ref": list(self.C_ref), "E_ref": list(self.E_ref),
            "zC_hi": self.zC_hi, "zE_hi": self.zE_hi, "alpha": self.alpha,
            "thresholds_from": "neutral calibration quantiles (Bonferroni over 2 axes)",
        }


def _decision_basis(cal: PrismCalibration, tol: float = 1e-6):
    """Per layer: orthonormal basis of span{prism_U[:,0], prism_U[:,1], action_contrast}."""
    L, d, _ = cal.prism_U.shape
    Q = np.zeros((L, d, 3))
    rank = np.zeros(L, dtype=int)
    for l in range(L):
        cols = [cal.prism_U[l, :, 0], cal.prism_U[l, :, 1], cal.action_contrast[l]]
        basis: List[np.ndarray] = []
        for c in cols:
            v = np.asarray(c, float).copy()
            for b in basis:
                v -= np.dot(v, b) * b
            n = float(np.linalg.norm(v))
            if n > tol:
                basis.append(v / n)
        for k, b in enumerate(basis):
            Q[l, :, k] = b
        rank[l] = len(basis)
    return Q, rank


def _layer_terms(r: np.ndarray, Q: np.ndarray):
    """r: [L,d] residual step. Returns complement norm, decision norm, decision component."""
    coef = np.einsum("ldk,ld->lk", Q, r)
    dec_vec = np.einsum("ldk,lk->ld", Q, coef)
    comp = np.linalg.norm(r - dec_vec, axis=1)
    dec = np.linalg.norm(coef, axis=1)
    return comp, dec, dec_vec


def _log_f(comp: np.ndarray, dec: np.ndarray, rank: np.ndarray, dim: int) -> np.ndarray:
    k = np.maximum(rank, 1).astype(float)
    num = dec ** 2 / k
    den = comp ** 2 / np.maximum(dim - k, 1.0)
    return np.log((num + EPS) / (den + EPS))


def _robust_ref(v: np.ndarray, floor: float) -> tuple:
    m = float(np.median(v))
    ad = np.abs(v - m)
    sc = max(float(np.quantile(ad, 0.90)) / 1.281551565545, 1.4826 * float(np.median(ad)), floor)
    return m, sc


def calibrate_triad(H_N, H_L, H_R, D_N, D_L, D_R, layer_lo: int = 0,
                    layer_hi: Optional[int] = None, alpha: float = 0.05) -> TriadCalibration:
    prism = calibrate_prism(H_N, H_L, H_R, D_N, D_L, D_R, layer_lo, layer_hi)
    Q, rank = _decision_basis(prism)
    res = np.asarray(D_N, float) - prism.v_N[None]
    comps, decs = [], []
    for r in res:
        c, dd, _ = _layer_terms(r, Q)
        comps.append(c); decs.append(dd)
    comps = np.asarray(comps); decs = np.asarray(decs)
    dim = int(res.shape[2])
    s_comp = np.maximum(np.median(comps, axis=0), 1e-8)
    s_dec = np.maximum(np.median(decs, axis=0), 1e-8)
    w = prism.layer_reliability
    C_neu = np.asarray([weighted_mean(c / s_comp, w) for c in comps])
    E_neu = np.asarray([weighted_mean(_log_f(c, dd, rank, dim), w) for c, dd in zip(comps, decs)])
    C_ref = _robust_ref(C_neu, 1e-3)
    E_ref = _robust_ref(E_neu, 1e-3)
    q = 1.0 - alpha / 2.0  # Bonferroni over the two axes, one-sided each
    zC_hi = float(np.quantile((C_neu - C_ref[0]) / C_ref[1], q))
    zE_hi = float(np.quantile((E_neu - E_ref[0]) / E_ref[1], q))
    return TriadCalibration(prism, Q, rank, s_comp, s_dec, dim, C_ref, E_ref,
                            max(zC_hi, 1.0), max(zE_hi, 1.0), alpha)


# ------------------------------------------------------------ observables --
def change_emotion(phi: np.ndarray, prev_phi: np.ndarray, tc: TriadCalibration) -> Dict:
    r = np.asarray(phi, float) - np.asarray(prev_phi, float) - tc.prism.v_N
    comp, dec, _ = _layer_terms(r, tc.Q)
    w = tc.prism.layer_reliability
    C = weighted_mean(comp / tc.s_comp, w)
    E = weighted_mean(_log_f(comp, dec, tc.rank, tc.dim), w)
    motion = weighted_mean(dec / tc.s_dec, w)
    zC = (C - tc.C_ref[0]) / tc.C_ref[1]
    zE = (E - tc.E_ref[0]) / tc.E_ref[1]
    share = float(np.sum(w * dec ** 2) / max(np.sum(w * (dec ** 2 + comp ** 2)), EPS))
    contrast = float(weighted_mean(np.einsum("ld,ld->l", r, tc.prism.action_contrast), w))
    hiC, hiE = zC >= tc.zC_hi, zE >= tc.zE_hi
    quad = QUADRANTS[int(hiC) + 2 * int(hiE)]
    return {"C": C, "E": E, "zC": float(zC), "zE": float(zE), "decision_share": share,
            "decision_motion": motion,
            "contrast_sign": float(np.sign(contrast)), "quadrant": quad}


def layer_polarization(phi: np.ndarray, tc: TriadCalibration) -> float:
    """P_t = Agg_{i<j} d(Phi_i, Phi_j), layers as views, reliability-weighted."""
    s = state_observables(phi, tc.prism)
    V = np.stack([s["bounded_L"], s["bounded_R"]], axis=1)
    w = tc.prism.layer_reliability
    num = den = 0.0
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            ww = w[i] * w[j]
            num += ww * float(np.linalg.norm(V[i] - V[j]))
            den += ww
    return num / den if den > 0 else 0.0


# ------------------------------------------------------------ isolation gate --
@dataclass
class IsolationConfig:
    lam: float = 0.5
    trust_ratio: float = 0.02
    one_sided_R: bool = False  # default symmetric: damps L- and R-going excess alike


def isolation_gate(phi: np.ndarray, prev_phi: np.ndarray, tc: TriadCalibration,
                   cfg: IsolationConfig = IsolationConfig()) -> Dict:
    ce = change_emotion(phi, prev_phi, tc)
    g = 1.0 / (1.0 + cfg.lam * max(0.0, ce["zE"] - tc.zE_hi))
    if cfg.one_sided_R and ce["contrast_sign"] <= 0:
        g = 1.0
    x = np.asarray(phi, float)
    r = x - np.asarray(prev_phi, float) - tc.prism.v_N
    _, _, dec_vec = _layer_terms(r, tc.Q)
    dh = -(1.0 - g) * dec_vec * tc.prism.layer_reliability[:, None]
    n_h = np.linalg.norm(x, axis=1); n_d = np.linalg.norm(dh, axis=1)
    scale = np.minimum(1.0, cfg.trust_ratio * np.maximum(n_h, EPS) / np.maximum(n_d, EPS))
    dh *= scale[:, None]
    return {"gain": g, "delta_h": dh, "corrected": x + dh, "applied_scale": scale, **ce}


# ------------------------------------------------------------ input spans --
def span_effects(score_fn: Callable[[Sequence[bool]], float], surprisal: Sequence[float],
                 thr_change: float, thr_effect: float) -> List[Dict]:
    """Causal emotion of input spans by ablation.

    score_fn(keep_mask) -> decision score with only the kept spans in the prompt
    (e.g. log p of the secret's first token, or v1.5 action_diff at the event).
    surprisal[i] = mean token surprisal of span i under the model (its CHANGE).
    Thresholds must be fixed before looking at leak outcomes.
    """
    n = len(surprisal)
    base = float(score_fn([True] * n))
    rows = []
    for i in range(n):
        mask = [True] * n; mask[i] = False
        eff = abs(base - float(score_fn(mask)))
        hiC, hiE = surprisal[i] >= thr_change, eff >= thr_effect
        rows.append({"span": i, "change": float(surprisal[i]), "effect": eff,
                     "quadrant": QUADRANTS[int(hiC) + 2 * int(hiE)]})
    return rows


# ------------------------------------------------------------ replay --
def replay(traj: np.ndarray, tc: TriadCalibration, tokens: Optional[Sequence[str]] = None,
           turns: Optional[Sequence[int]] = None) -> Dict:
    """Same delta rule as v1.3 within_turn_deltas: no step is taken across a turn boundary."""
    st = OnlineState(); rows = []; prev = None; prev_turn = None
    for t, phi in enumerate(np.asarray(traj, float)):
        turn = 0 if turns is None else int(turns[t])
        if prev_turn is not None and turn != prev_turn:
            prev = None
        v15 = online_step(phi, prev, tc.prism, t, turn, st, OnsetConfig())
        row = {"t": t, "turn": turn, "token": None if tokens is None else tokens[t],
               "P_layers": layer_polarization(phi, tc),
               "v15_phase": v15["phase"], "v15_onset": v15["onset"],
               "constraint_integrity": v15["constraint_integrity"]}
        if prev is not None:
            row.update(change_emotion(phi, prev, tc))
        rows.append(row); prev = phi; prev_turn = turn
    counts = {q: sum(r.get("quadrant") == q for r in rows) for q in QUADRANTS}
    P = np.asarray([r["P_layers"] for r in rows])
    return {"meta": tc.metadata(), "quadrant_counts": counts,
            "P_layers": {"median": float(np.median(P)), "max": float(np.max(P))},
            "rows": rows}


# ------------------------------------------------------------ selftest --
def _fixture(seed: int = 7, N: int = 200, L: int = 6, d: int = 64):
    rng = np.random.default_rng(seed)
    e1 = safe_unit(rng.normal(size=d)); e2 = safe_unit(rng.normal(size=d))
    r = safe_unit(0.65 * e1 + np.sqrt(1 - 0.65 ** 2) * e2)
    HN = rng.normal(0, 0.03, (N, L, d)); HL = HN + e1; HR = HN + r
    aL = safe_unit(rng.normal(size=d)); tmp = safe_unit(rng.normal(size=d))
    aR = safe_unit(0.4 * aL + 0.9165 * tmp)
    DN = rng.normal(0, 0.015, (N, L, d)); DL = DN + 0.35 * aL; DR = DN + 0.35 * aR
    return rng, (HN, HL, HR, DN, DL, DR), e1, r, aR


def selftest() -> int:
    rng, cal_arrays, e1, rax, aR = _fixture()
    tc = calibrate_triad(*cal_arrays)
    L, d = tc.Q.shape[0], tc.Q.shape[1]
    base = np.tile(e1, (L, 1)) + tc.prism.v_N * 0.0
    vN = tc.prism.v_N

    def step(prev, extra):
        return prev + vN + extra + rng.normal(0, 0.015, prev.shape)

    # an off-decision direction for planted NOISE
    Qall = tc.Q.reshape(L, d, -1)
    junk = rng.normal(size=(L, d))
    for l in range(L):
        junk[l] -= Qall[l] @ (Qall[l].T @ junk[l])
    junk = junk / np.linalg.norm(junk, axis=1, keepdims=True)
    # decision-relevant push = the calibrated R-going differential action (inside Q).
    # v0.1 first draft planted along raw aR, which also leaks off-decision -> fixture bug.
    dec_dir = safe_unit(tc.prism.action_contrast)

    planted = {
        "ROUTINE": np.zeros((L, d)),
        "NOISE": 0.60 * junk,
        "STEALTH": 0.12 * dec_dir,
        "OVERT": 0.60 * junk + 0.40 * dec_dir,
    }
    hits = {}
    for name, extra in planted.items():
        ok = 0
        for _ in range(40):
            cur = step(base, extra)
            ok += change_emotion(cur, base, tc)["quadrant"] == name
        hits[name] = ok / 40
    t1 = all(v >= 0.85 for v in hits.values())

    # what persists does not surprise: a step equal to v_N has ~neutral C
    ce_exp = change_emotion(base + vN, base, tc)
    t2 = ce_exp["zC"] < tc.zC_hi and ce_exp["zE"] < tc.zE_hi

    # false-quadrant rate on fresh neutral steps near nominal alpha
    fp = 0
    for _ in range(400):
        cur = step(base, np.zeros((L, d)))
        fp += change_emotion(cur, base, tc)["quadrant"] != "ROUTINE"
    fp_rate = fp / 400
    t3 = fp_rate <= 2.0 * tc.alpha

    # large isotropic noise is CHANGE, not EMOTION (scale-free E)
    iso = [change_emotion(base + vN + rng.normal(0, 0.06, base.shape), base, tc)["quadrant"]
           for _ in range(80)]
    iso_noise = iso.count("NOISE") / len(iso)
    t3b = iso_noise >= 0.85

    # layer polarization: agreement low, split band high
    agree = np.tile(e1, (L, 1))
    split = np.stack([e1 if l < L // 2 else rax * 2.0 - e1 * 0.0 for l in range(L)])
    split[: L // 2] = e1 * 2.0
    split[L // 2:] = (rax - np.dot(rax, e1) * e1) * 2.0
    P_agree, P_split = layer_polarization(agree, tc), layer_polarization(split, tc)
    t4 = P_split > 3 * max(P_agree, 1e-6)

    # isolation gate: complement invariant, damps stealth, ~idle on routine
    cur = step(base, planted["STEALTH"])
    out = isolation_gate(cur, base, tc, IsolationConfig(trust_ratio=0.5))
    def orth(x):
        c = np.einsum("ldk,ld->lk", tc.Q, x)
        return x - np.einsum("ldk,lk->ld", tc.Q, c)
    drift = float(np.max(np.abs(orth(out["corrected"]) - orth(cur))))
    after = change_emotion(out["corrected"], base, tc)["zE"]
    routine = isolation_gate(step(base, planted["ROUTINE"]), base, tc)
    t5 = drift < 1e-9 and after < out["zE"] and routine["gain"] > 0.99

    tests = [("quadrants recovered on planted steps", t1),
             ("expected (permanent) step is not surprising", t2),
             ("neutral false-quadrant rate <= 2 alpha", t3),
             ("4x isotropic noise read as NOISE, not emotion", t3b),
             ("layer polarization separates agree vs split", t4),
             ("isolation gate: complement invariant, damps STEALTH, idle on ROUTINE", t5)]
    print(f"CRAD Triad v{VERSION} selftest")
    for name, ok in tests:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    print(f"  quadrant hit rates: {hits}")
    print(f"  thresholds zC_hi={tc.zC_hi:.2f} zE_hi={tc.zE_hi:.2f}; neutral false-quadrant={fp_rate:.3f}")
    print(f"  4x isotropic noise filed as NOISE: {iso_noise:.3f}")
    print(f"  P_layers agree={P_agree:.4f} split={P_split:.4f}")
    print(f"  gate on STEALTH: gain={out['gain']:.3f} zE {out['zE']:.2f} -> {after:.2f}; "
          f"complement drift={drift:.1e}; routine gain={routine['gain']:.3f}")
    return 0 if all(ok for _, ok in tests) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--replay", help="npz with H_N,H_L,H_R,D_N,D_L,D_R,traj[,tokens,turns]; "
                   "H_* = stack_states, D_* = within_turn_deltas of v1.3 build_calibration frames")
    p.add_argument("--layer-lo", type=int, default=14)
    p.add_argument("--out", help="write replay JSON here")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    if a.replay:
        z = np.load(a.replay, allow_pickle=False)
        tc = calibrate_triad(z["H_N"], z["H_L"], z["H_R"], z["D_N"], z["D_L"], z["D_R"],
                             layer_lo=a.layer_lo)
        tokens = [str(t) for t in z["tokens"]] if "tokens" in z.files else None
        turns = z["turns"] if "turns" in z.files else None
        res = replay(z["traj"], tc, tokens, turns)
        txt = json.dumps(res, indent=2, default=float)
        if a.out:
            Path(a.out).write_text(txt + "\n")
            print(f"[triad] wrote {a.out}: {res['quadrant_counts']}")
        else:
            print(txt)
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
