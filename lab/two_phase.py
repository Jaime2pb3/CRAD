#!/usr/bin/env python3
"""two_phase v0.1 — is CRAD's L/R plane a separate channel from the decision? Bound to crad_contract.

TwoQuarks Research — Luis Jaime Ledesma Perez

The hypothesis (oil and water)
------------------------------
At the decision row of turn 7, inside CRAD's band, the residual carries two channels that
travel together but do not mix:
  S  CRAD's measurement plane span{uL, uR}: a LOUD channel (moving along it changes the
     output distribution a lot, see FISHER_G4.md) that does NOT carry the TRUTH vs
     SELF_FICTION choice;
  C  where the real decision difference delta = h_TRUTH - h_SF lives (>= 98% of its energy
     is outside S in the A100 pair): it carries the choice.
Neither channel, when pushed, makes the network write into the other one downstream.

Why the old patch_probe result is NOT evidence for this
-------------------------------------------------------
patch_probe moved only the S-part of delta. That part holds 0.1% to 2% of delta's energy,
so "LR does not flip, orth flips" follows from geometry alone. Here every direction is
pushed with the SAME amplitude a = ||delta||, so S gets a fair dose.

Conditions at layer l, decision (anchor) row d of the target, a = ||h_src - h_tgt||:
  full  h_src                                  replication of patch_probe
  C     h_tgt + a * unit(P_perp delta)          the decision channel, full dose
  S+    h_tgt + a * unit(P_S delta)             CRAD's plane, full dose, the S-part of delta's direction
  S2    h_tgt + a * o * e2                     the other direction of the plane: e2 is the unit vector of S
                                                orthogonal to e1 = unit(P_S (median h_TRUTH - median h_SF)),
                                                o = +1 for TRUTH->SF pairs and -1 for SF->TRUTH, so the push
                                                has the same physical sign for every pair. S is a plane: both
                                                of its directions get the full dose
  S-    h_tgt - a * unit(P_S delta)             same as S+, opposite sign (secondary)
  nat   h_tgt + a * r_nat                       natural null: drawn from the covariance of the
                                                calibration states at l, orthogonal to S and delta
  iso   h_tgt + a * r_iso                       isotropic null, orthogonal to S and delta (secondary)

Readouts (model's own logits at row d, and the traced residual downstream at row d):
  toward  sign * delta [log p(secret first token) - log p(fiction token)], sign = +1 if the source is
          TRUTH. The fiction token is the one the SELF_FICTION member of the pair emitted at its decision
          row. The difference cancels the softmax normalizer, so a loud push that only rescales the
          distribution does not count as moving the choice (log p(secret) alone would: kept as toward_lp)
  KL      KL(p_patched || p_base) at row d, whole vocabulary (how LOUD the push is)
  dS, dC  network response downstream: for l < l' <= l + SPAN, resp = (h_patched - h_base)
          - (v - h_tgt), i.e. what the network ADDED beyond carrying the injection forward.
          dS = || its projection on S(l') || in calibration sigmas; dC = |its projection on the
          decision axis of l' with S(l') removed| in decision-row sigmas (leave-pair-out axis).

Subcommands
  selftest                      planted worlds on CPU: the analysis must pass the two-phase world
                                and fail the mixed, quiet and coupled worlds
  plan     captures/<id>        CPU: pairs by rule, geometry of delta against S and the nulls, cost
  run      captures/<id>        GPU: the patching (one model load). --max-pairs, --layers
  analyze  captures/<id>/two_phase/<stamp>   CPU: the preregistered verdict (PREREG_TWOPHASE.md)

Gates (all abort): unpatched run reproduces the captured turn; self-patch leaves the logits
within SELF_TOL; read-back: the traced residual at (d, l) equals the injected vector.
Fake secret on a local model; nothing leaves the machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

VERSION = "0.1"
PRIMARY_LAYERS = (14, 17)          # in CRAD's band: where the ruler is defined
EXPLORATORY_LAYERS = (21,)         # off band: where the real difference dominates R (FISHER_G4)
MODES = ("full", "C", "S+", "S2", "S-", "nat", "iso")
SPAN = 6                           # downstream layers read for cross-talk
N_PCS = 64                         # natural null: top principal components of the calibration states
EXTRA_TOKENS = 24                  # generate up to commit_row + EXTRA_TOKENS (content is complete)
SELF_TOL = 1e-3                    # nats
READBACK_TOL = 2e-3                # relative, fp16
MAX_PAIRS = 40
PAIR_SEED = 20261121
NULL_SEED = 20261122
BOOT_SEED = 20261123
N_BOOT = 10000
MARGIN_R = 0.15                    # H2 equivalence margin (S carries < 15% of C's choice per unit amplitude)
MARGIN_Q = 2.0                     # H3 cross-talk ceiling (at most twice what a natural direction causes)
RATIO_GATE = 0.8                   # G0: the C dose recovers >= 80% of the full patch
KL_FLOOR = 1e-8


# ============================================================ geometry (pure numpy) ==
def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-300)


def orth_basis(vectors):
    if len(vectors) == 0:
        return None
    q, r = np.linalg.qr(np.stack([np.asarray(v, float) for v in vectors], axis=1))
    keep = np.abs(np.diag(r)) > 1e-10 * max(1.0, float(np.abs(r).max()))
    return q[:, keep]


def project_out(v, basis):
    v = np.asarray(v, float)
    if basis is None or basis.shape[1] == 0:
        return v
    return v - basis @ (basis.T @ v)


_SVD = {}


def _top_pcs(X, k):
    key = (id(X), X.shape, k)
    if key not in _SVD:
        X = np.asarray(X, float)
        Xc = X - X.mean(0)
        k = int(min(k, Xc.shape[0] - 1, Xc.shape[1]))
        if k < 2:
            raise ValueError("natural null needs at least 3 calibration states")
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        _SVD[key] = (vt[:k], s[:k] / np.sqrt(max(Xc.shape[0] - 1, 1)))
    return _SVD[key]


def natural_direction(X, exclude, rng, k=N_PCS):
    """Unit direction drawn from the covariance of X (top-k PCs), orthogonal to `exclude`."""
    vt, sd = _top_pcs(X, k)
    g = rng.normal(size=len(sd))
    r = (vt.T * sd) @ g
    r = project_out(r, orth_basis(exclude))
    n = float(np.linalg.norm(r))
    if n < 1e-9:
        raise ValueError("natural null collapsed after removing S and delta")
    return r / n


def iso_direction(d, exclude, rng):
    r = project_out(rng.normal(size=d), orth_basis(exclude))
    return unit(r)


def plane_energy(delta, basis):
    p = basis.T @ delta
    return float(p @ p) / max(float(delta @ delta), 1e-300)


class Degenerate(ValueError):
    """The pair cannot be dosed (identical states, or delta without an S part). Skipped and recorded."""


def plane_frame(uL, uR, h_truth, h_fic):
    """e1 = class-level S-part of the decision, e2 = the other direction of S (unit, orthogonal)."""
    S = orth_basis([uL, uR])
    d = np.median(np.asarray(h_truth, float), 0) - np.median(np.asarray(h_fic, float), 0)
    e1 = unit(S @ (S.T @ d))
    e2 = project_out(S[:, 0], e1[:, None])
    e2 = unit(e2 if np.linalg.norm(e2) > 1e-6 else project_out(S[:, 1], e1[:, None]))
    return e1, e2


def build_doses(h_src, h_tgt, uL, uR, X_cal, rng, e2, orient):
    """All six patch vectors at one (pair, layer), plus their geometry."""
    h_src, h_tgt = np.asarray(h_src, float), np.asarray(h_tgt, float)
    delta = h_src - h_tgt
    a = float(np.linalg.norm(delta))
    if a < 1e-6 * max(float(np.linalg.norm(h_tgt)), 1e-300):
        raise Degenerate("source and target states are identical")
    S = orth_basis([uL, uR])
    pS = S @ (S.T @ delta)
    if float(np.linalg.norm(pS)) < 1e-9 * max(a, 1e-300):
        raise Degenerate("delta has no component in S: the S dose has no direction")
    s_hat, c_hat = unit(pS), unit(delta - pS)
    s2 = float(orient) * unit(e2)
    r_nat = natural_direction(X_cal, [uL, uR, delta], rng)
    r_iso = iso_direction(delta.shape[0], [uL, uR, delta], rng)
    vecs = {"full": h_src, "C": h_tgt + a * c_hat, "S+": h_tgt + a * s_hat, "S2": h_tgt + a * s2,
            "S-": h_tgt - a * s_hat,
            "nat": h_tgt + a * r_nat, "iso": h_tgt + a * r_iso}
    meta = {"a": a, "energy_S": plane_energy(delta, S), "rel_delta": a / max(float(np.linalg.norm(h_tgt)), 1e-300)}
    return vecs, meta


def kl_from_logp(lp_p, lp_q):
    lp_p, lp_q = np.asarray(lp_p, float), np.asarray(lp_q, float)
    p = np.exp(lp_p)
    return float(max(np.sum(p * (lp_p - lp_q)), 0.0))


class Ruler:
    """Downstream readout geometry at one layer l': S(l'), its sigmas, and the decision axis."""

    def __init__(self, uL, uR, X_cal, mu_N, dec_truth, dec_fic):
        self.S = orth_basis([uL, uR])
        self.uL, self.uR = unit(uL), unit(uR)
        Xc = np.asarray(X_cal, float) - np.asarray(mu_N, float)
        self.sig_L = max(float(np.std(Xc @ self.uL)), 1e-12)
        self.sig_R = max(float(np.std(Xc @ self.uR)), 1e-12)
        self.dec_truth, self.dec_fic = dec_truth, dec_fic          # {draw: state}

    def decision_axis(self, drop):
        tr = [v for k, v in self.dec_truth.items() if k not in drop]
        fi = [v for k, v in self.dec_fic.items() if k not in drop]
        if len(tr) < 2 or len(fi) < 2:
            raise ValueError("decision axis needs >= 2 TRUTH and >= 2 SELF_FICTION after leave-pair-out")
        u = unit(project_out(np.median(tr, 0) - np.median(fi, 0), self.S))
        pt, pf = np.array(tr) @ u, np.array(fi) @ u
        sd = np.sqrt((np.sum((pt - pt.mean()) ** 2) + np.sum((pf - pf.mean()) ** 2)) / max(len(pt) + len(pf) - 2, 1))
        return u, max(float(sd), 1e-12)

    def responses(self, resp, drop):
        u, sd = self.decision_axis(drop)
        dS = float(np.hypot((resp @ self.uL) / self.sig_L, (resp @ self.uR) / self.sig_R))
        dC = abs(float(resp @ u)) / sd
        return dS, dC


# ================================================================ statistics ==
def cluster_boot(units, stat, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap over clusters (target draws). units: list of dicts with key 'cluster'."""
    by = defaultdict(list)
    for u in units:
        by[u["cluster"]].append(u)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    est = stat(units)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), size=len(keys))
        sample = [u for i in pick for u in by[keys[i]]]
        v = stat(sample)
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    if len(vals) < n_boot // 2:
        return est, float("nan"), float("nan"), float("nan"), float("nan")
    return est, *(float(np.percentile(vals, q)) for q in (2.5, 5, 95, 97.5))


def sign_flip_p(x, n=N_BOOT, seed=BOOT_SEED):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = x.mean()
    s = rng.choice([-1.0, 1.0], size=(n, len(x)))
    return float((np.sum((s * x).mean(1) >= obs - 1e-12) + 1) / (n + 1))     # one-sided: mean > 0


def st_H1(us):
    return float(np.mean([np.log(max(u["KL"]["S+"], KL_FLOOR) / max(u["KL"]["nat"], KL_FLOOR)) for u in us]))


def st_H2(mode):
    def f(us):
        c = np.mean([u["toward"]["C"] for u in us])
        if not np.isfinite(c) or c <= 1e-9:
            return float("nan")
        return float((np.mean([u["toward"][mode] for u in us]) - np.mean([u["toward"]["nat"] for u in us])) / c)
    return f


def st_Q(src_mode, field):
    def f(us):
        den = np.mean([u[field]["nat"] for u in us])
        return float(np.mean([u[field][src_mode] for u in us]) / den) if den > 1e-12 else float("nan")
    return f


def verdict(units, n_boot=N_BOOT):
    """The preregistered decision (PREREG_TWOPHASE.md). units = (pair, primary layer) records."""
    out = {"n_units": len(units), "n_clusters": len({u["cluster"] for u in units})}
    tw_full = [u["toward"]["full"] for u in units]
    ratios = [u["toward"]["C"] / u["toward"]["full"] for u in units if abs(u["toward"]["full"]) > 0.05]
    out["G0"] = {"mean_toward_full": float(np.mean(tw_full)), "p_full_gt_0": sign_flip_p(tw_full),
                 "median_C_over_full": float(np.median(ratios)) if ratios else float("nan"),
                 "n_ratio": len(ratios)}
    out["G0"]["pass"] = bool(out["G0"]["p_full_gt_0"] < 0.05 and out["G0"]["median_C_over_full"] >= RATIO_GATE)
    e, lo95, lo90, hi90, hi95 = cluster_boot(units, st_H1, n_boot)
    out["H1"] = {"stat": "mean log(KL_S+ / KL_nat)", "est": e, "ci90": [lo90, hi90], "pass": bool(lo90 > 0)}
    parts = {}
    for mode in ("S+", "S2"):
        e, lo95, lo90, hi90, hi95 = cluster_boot(units, st_H2(mode), n_boot)
        parts[mode] = {"est": e, "ci90": [lo90, hi90], "pass": bool(-MARGIN_R < lo90 and hi90 < MARGIN_R),
                       "fail": bool(lo90 >= MARGIN_R or hi90 <= -MARGIN_R)}
    worst = max(parts, key=lambda m: abs(parts[m]["est"]) if np.isfinite(parts[m]["est"]) else np.inf)
    out["H2"] = {"stat": "(mean toward_X - mean toward_nat) / mean toward_C, X = S+ and S2 (both must pass)",
                 "est": parts[worst]["est"], "ci90": parts[worst]["ci90"], "by_direction": parts,
                 "worst_direction": worst, "pass": all(p["pass"] for p in parts.values()),
                 "fail_high": any(p["fail"] for p in parts.values()), "fail_low": False}
    for name, mode, field in (("H3a", "C", "dS"), ("H3b", "S+", "dC")):
        e, lo95, lo90, hi90, hi95 = cluster_boot(units, st_Q(mode, field), n_boot)
        out[name] = {"stat": f"mean {field}[{mode}] / mean {field}[nat]", "est": e, "ci90": [lo90, hi90],
                     "pass": bool(hi90 < MARGIN_Q), "fail": bool(lo90 >= MARGIN_Q)}
    if not out["G0"]["pass"]:
        out["reading"] = "G0 FAILED: the patching does not replicate; no hypothesis is read."
    elif out["H2"]["fail_high"] or out["H2"]["fail_low"]:
        out["reading"] = "REJECTED (mixed): S moves the choice when dosed. The A100 patch result was geometry."
    elif out["H3a"]["fail"] or out["H3b"]["fail"]:
        out["reading"] = "REJECTED (coupled): pushing one channel makes the network write into the other."
    elif out["H2"]["pass"] and out["H3a"]["pass"] and out["H3b"]["pass"] and out["H1"]["pass"]:
        out["reading"] = ("SUPPORTED in this capture: S is louder than a natural direction, does not carry the "
                          "choice, and neither channel writes into the other. One capture, one seed: provisional.")
    elif out["H2"]["pass"] and out["H3a"]["pass"] and out["H3b"]["pass"]:
        out["reading"] = ("WEAK form only: S does not carry the choice and does not mix, but it is not louder "
                          "than a natural direction. The oil is not special.")
    else:
        out["reading"] = "INCONCLUSIVE: some interval straddles its margin. No claim."
    return out


# ================================================================ selftest ==
def _toy_world(kind, rng, d=48, V=64, n_draw=12):
    """Linear-softmax toy: logits = W h_final, h_final = h + M h. Secret = token 0."""
    S = np.linalg.qr(rng.normal(size=(d, 2)))[0]
    C = project_out(rng.normal(size=d), S); C = unit(C)
    rest = np.linalg.qr(np.column_stack([S, C, rng.normal(size=(d, d - 3))]))[0][:, 3:]
    W = rng.normal(size=(V, d)) * 0.15
    W[1:] += 1.2 * rng.normal(size=(V - 1, 2)) @ S.T          # S is loud: many tokens move along S
    w0 = 2.5 * C + 0.15 * rest[:, 0]
    if kind == "mixed":
        w0 = w0 + 2.5 * S[:, 0] + 2.5 * S[:, 1]               # the secret logit reads S too
    if kind == "mixed_S2":
        w0 = w0 + 3.0 * S[:, 1]                               # reads only the S direction delta does not use
    if kind == "quiet":
        W[1:] = project_out(rng.normal(size=(V - 1, d)).T * 0.15, S).T   # nothing reads S: it is quiet
    W[0] = w0
    W[1] = -0.5 * (2.5 * C) + rng.normal(size=d) * 0.05      # the fiction contender: no S
    M = np.zeros((d, d))
    if kind == "coupled":
        M = 1.5 * np.outer(S[:, 0], C)                        # pushing C writes into S downstream
    if kind == "coupled_back":
        M = 1.5 * np.outer(C, S[:, 0])                        # pushing S writes into C downstream
    sd_rest = 0.6
    X_cal = (rng.normal(size=(200, 2)) * 1.5) @ S.T + rng.normal(size=(200, d)) * sd_rest
    base = rng.normal(size=d) * 0.3
    truth = {k: base + 1.0 * C + 0.05 * S[:, 0] + rng.normal(size=d) * 0.05 for k in range(n_draw // 2)}
    fic = {k: base - 1.0 * C - 0.05 * S[:, 0] + rng.normal(size=d) * 0.05 for k in range(n_draw // 2, n_draw)}
    return dict(S=S, W=W, M=M, X_cal=X_cal, truth=truth, fic=fic, uL=S[:, 0], uR=S[:, 1])


def _toy_units(w, rng):
    def logp(h):
        z = w["W"] @ (h + w["M"] @ h)
        z = z - z.max()
        return z - np.log(np.exp(z).sum())

    ruler = Ruler(w["uL"], w["uR"], w["X_cal"], w["X_cal"].mean(0), w["truth"], w["fic"])
    units = []
    pairs = [(s, t) for s in w["truth"] for t in w["fic"]] + [(t, s) for s in w["truth"] for t in w["fic"]]
    allst = {**w["truth"], **w["fic"]}
    _, e2 = plane_frame(w["uL"], w["uR"], list(w["truth"].values()), list(w["fic"].values()))
    for s, t in pairs:
        sgn = 1.0 if s in w["truth"] else -1.0
        hs, ht = allst[s], allst[t]
        vecs, meta = build_doses(hs, ht, w["uL"], w["uR"], w["X_cal"], rng, e2, sgn)
        lb = logp(ht)
        u = {"cluster": t, "pair": [s, t], "layer": 0, "toward": {}, "KL": {}, "dS": {}, "dC": {}}
        for m, v in vecs.items():
            lpm = logp(v)
            u["toward"][m] = sgn * ((lpm[0] - lpm[1]) - (lb[0] - lb[1]))
            u["KL"][m] = kl_from_logp(lpm, lb)
            resp = w["M"] @ (v - ht)                           # what the "network" added downstream
            u["dS"][m], u["dC"][m] = ruler.responses(resp + 1e-6 * rng.normal(size=resp.shape), {s, t})
        units.append(u)
    return units


def selftest() -> int:
    ok = True
    rng = np.random.default_rng(0)
    # geometry
    d = 40
    uL, uR = rng.normal(size=d), rng.normal(size=d)
    X = rng.normal(size=(30, d))
    hs, ht = rng.normal(size=d), rng.normal(size=d)
    S = orth_basis([uL, uR])
    vecs, meta = build_doses(hs, ht, uL, uR, X, rng, project_out(S[:, 0], unit(S[:, 1])[:, None]), 1.0)
    a = np.linalg.norm(hs - ht)
    for m in ("C", "S+", "S2", "S-", "nat", "iso"):
        good = abs(np.linalg.norm(vecs[m] - ht) - a) < 1e-9
        ok &= good
        print(f"[selftest] amplitude {m:4s} = ||delta||: {'OK' if good else 'FAIL'}")
    for m in ("C", "nat", "iso"):
        good = np.linalg.norm(S.T @ (vecs[m] - ht)) < 1e-8 * a
        ok &= good
        print(f"[selftest] {m:4s} orthogonal to S: {'OK' if good else 'FAIL'}")
    for m in ("S+", "S2"):
        good = np.linalg.norm(project_out(vecs[m] - ht, S)) < 1e-8 * a
        ok &= good
        print(f"[selftest] {m} inside S: {'OK' if good else 'FAIL'}")
    for m in ("nat", "iso"):
        good = abs(unit(vecs[m] - ht) @ unit(hs - ht)) < 1e-8
        ok &= good
        print(f"[selftest] {m:4s} orthogonal to delta: {'OK' if good else 'FAIL'}")
    good = np.allclose(vecs["full"], hs)
    ok &= good
    print(f"[selftest] full = h_src: {'OK' if good else 'FAIL'}")
    lp = np.log(np.array([0.5, 0.25, 0.25]))
    good = abs(kl_from_logp(lp, lp)) < 1e-12 and kl_from_logp(lp, np.log(np.array([0.25, 0.5, 0.25]))) > 0
    ok &= good
    print(f"[selftest] KL: {'OK' if good else 'FAIL'}")
    # planted worlds through the SAME verdict code
    expect = {"two_phase": "SUPPORTED", "mixed": "REJECTED (mixed)", "mixed_S2": "REJECTED (mixed)", "quiet": "WEAK",
              "coupled": "REJECTED (coupled)", "coupled_back": "REJECTED"}
    for kind, want in expect.items():
        w = _toy_world(kind, np.random.default_rng(7))
        v = verdict(_toy_units(w, np.random.default_rng(8)), n_boot=2000)
        got = v["reading"]
        good = got.startswith(want)
        ok &= good
        print(f"[selftest] planted world {kind:12s} -> {got.split(':')[0]:28s} "
              f"(H1 {v['H1']['est']:+.2f}, H2 {v['H2']['est']:+.3f}, H3a {v['H3a']['est']:.2f}, "
              f"H3b {v['H3b']['est']:.2f}) {'OK' if good else 'FAIL, wanted ' + want}")
    print("SELFTEST: ALL OK" if ok else "SELFTEST: FAILED")
    return 0 if ok else 1


# ======================================================= capture side (CPU) ==
def _load(capture_dir):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from capture import load_v13
    import run as runmod
    from crad_contract import SharedCalibration
    v13 = load_v13(runmod.SRC)
    return v13, SharedCalibration.from_capture(Path(capture_dir), v13)


def select_pairs(dec, truth, fic, max_pairs, seed=PAIR_SEED):
    """Every TRUTH x SELF_FICTION pair, both directions; deterministic subsample if too many.
    Ties to the rule of ladder.py: equal history length (global_token) is recorded, not required."""
    allp = [(s, t) for s in truth for t in fic] + [(f, s) for s in truth for f in fic]
    allp = sorted(allp)
    if len(allp) > max_pairs:
        rng = np.random.default_rng(seed)
        allp = sorted(allp[i] for i in rng.choice(len(allp), size=max_pairs, replace=False))
    return [{"source": s, "target": t, "dir": "T->F" if dec[s].outcome == "TRUTH" else "F->T",
             "equal_length": dec[s].global_token == dec[t].global_token,
             "same_token": dec[s].token == dec[t].token} for s, t in allp]


def capture_side(sc, T, anchor, layers, max_pairs):
    from patch_probe import captured_state
    z = sc.z
    sup_lo, sup_hi = sc.superset
    n_draws = int(z["draw"][z["kind"] == 1].max()) + 1
    dec = {k: sc.decision(k, T, anchor) for k in range(n_draws)}
    truth = [k for k, d in dec.items() if d.outcome == "TRUTH" and d.row is not None]
    fic = [k for k, d in dec.items() if d.outcome == "SELF_FICTION" and d.row is not None]
    if len(truth) < 3 or len(fic) < 3:
        raise SystemExit(f"need >= 3 TRUTH and >= 3 SELF_FICTION draws (have {len(truth)}, {len(fic)})")
    read_hi = sup_hi - 1                                   # the last captured layer is post-norm: never read
    need = sorted({l2 for l in layers for l2 in range(l, min(l + SPAN, read_hi - 1) + 1)})
    H = {(k, l): captured_state(z, k, T, dec[k].row, l, sup_lo)[0] for k in truth + fic for l in need}
    cal = z["kind"] == 0
    rulers, Xcal, axes = {}, {}, {}
    for l in need:
        ax = sc.axes_at_layer(l)
        X = z["phi"][cal][:, l - sup_lo].astype(np.float64)
        Nmask = z["branch"][cal] == 0
        mu_N = np.median(X[Nmask], 0)
        rulers[l] = Ruler(ax["uL"], ax["uR"], X, mu_N, {k: H[(k, l)] for k in truth}, {k: H[(k, l)] for k in fic})
        Xcal[l], axes[l] = X, ax
        axes[l]["e1"], axes[l]["e2"] = plane_frame(ax["uL"], ax["uR"], [H[(k, l)] for k in truth],
                                                   [H[(k, l)] for k in fic])
    return dec, truth, fic, H, rulers, Xcal, axes, read_hi


def cmd_plan(a) -> int:
    from crad_contract import DEFAULT_ANCHOR, DEFAULT_TARGET_TURN
    v13, sc = _load(a.capture_dir)
    layers = a.layers or list(PRIMARY_LAYERS + EXPLORATORY_LAYERS)
    dec, truth, fic, H, rulers, Xcal, axes, read_hi = capture_side(sc, DEFAULT_TARGET_TURN, DEFAULT_ANCHOR,
                                                                    layers, a.max_pairs)
    pairs = select_pairs(dec, truth, fic, a.max_pairs)
    rng = np.random.default_rng(NULL_SEED)
    print(f"[plan] calibration_id {sc.calibration_id} · TRUTH {truth} · SELF_FICTION {fic}")
    print(f"[plan] {len(pairs)} pairs ({sum(p['equal_length'] for p in pairs)} with equal history length) "
          f"· layers {layers} · readable up to L{read_hi - 1}")
    print("\n| layer | median energy of delta in S | in natural planes | in isotropic planes | S / natural |")
    print("|---|---|---|---|---|")
    geo = {}
    for l in layers:
        eS, eN, eI = [], [], []
        for p in pairs:
            delta = H[(p["source"], l)] - H[(p["target"], l)]
            ax = axes[l]
            eS.append(plane_energy(delta, orth_basis([ax["uL"], ax["uR"]])))
            n1 = natural_direction(Xcal[l], [], rng); n2 = natural_direction(Xcal[l], [n1], rng)
            eN.append(plane_energy(delta, orth_basis([n1, n2])))
            d = delta.shape[0]
            eI.append(plane_energy(delta, orth_basis([rng.normal(size=d), rng.normal(size=d)])))
        geo[l] = {"S": float(np.median(eS)), "nat": float(np.median(eN)), "iso": float(np.median(eI))}
        print(f"| {l} | {geo[l]['S']:.5f} | {geo[l]['nat']:.5f} | {geo[l]['iso']:.6f} | "
              f"{geo[l]['S'] / max(geo[l]['nat'], 1e-12):.2f} |")
    n_targets = len({p["target"] for p in pairs})
    gens = n_targets * (1 + len(layers)) + len(pairs) * len(layers) * len(MODES)
    print(f"\n[plan] generations: {gens} ({n_targets} targets, {len(pairs)} pairs x {len(layers)} layers x "
          f"{len(MODES)} modes, plus baselines and self-patch gates). Each is one turn cut at commit + "
          f"{EXTRA_TOKENS} tokens. Multiply by the seconds per turn of your preflight.")
    out = Path(a.capture_dir) / "two_phase"
    out.mkdir(exist_ok=True)
    (out / "PLAN.json").write_text(json.dumps({"version": VERSION, "calibration_id": sc.calibration_id,
                                               "pairs": pairs, "layers": layers, "geometry": geo,
                                               "generations": gens}, indent=2) + "\n")
    print(f"[plan] wrote {out / 'PLAN.json'}")
    return 0


# ============================================================= run (GPU) ==
def cmd_run(a) -> int:
    from crad_contract import DEFAULT_ANCHOR, DEFAULT_TARGET_TURN, ContractError, classify
    from dose_probe import captured_turn, rng_restore, rng_snapshot
    import crad_base
    v13, sc = _load(a.capture_dir)
    z, args = sc.z, sc.args
    T, anchor = DEFAULT_TARGET_TURN, DEFAULT_ANCHOR
    layers = a.layers or list(PRIMARY_LAYERS + EXPLORATORY_LAYERS)
    dec, truth, fic, H, rulers, Xcal, axes, read_hi = capture_side(sc, T, anchor, layers, a.max_pairs)
    pairs = select_pairs(dec, truth, fic, a.max_pairs)
    lo, hi = sc.band
    sup_lo = sc.superset[0]
    n_tok, focus = int(args["n-tokens"]), int(args["focus-from-turn"])
    print(f"[two_phase] TRUTH {truth} · SELF_FICTION {fic} · {len(pairs)} pairs · layers {layers}", flush=True)

    v13.seed_everything(int(args["seed"]))
    mw = crad_base.RecogModel(args["model"], use_4bit=not args["no4bit"])
    torch = mw.torch
    n_layers = int(mw.lm.config.num_hidden_layers)
    read_hi = min(read_hi, n_layers - 1)
    for l in layers:
        if not (sup_lo <= l < read_hi):
            raise ContractError(f"layer {l} cannot be patched and read (readable [{sup_lo},{read_hi}))")
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
            h[:, -1, :] = HOOK["vec"].to(h.dtype)                # REPLACE at one row, one layer
            return out
        return hook

    handles = [mw.lm.model.layers[l].register_forward_hook(make_hook(l)) for l in layers]

    def release():
        for h in handles:
            h.remove()

    dec_id = {k: captured_turn(z, k, T)[1][dec[k].row] for k in truth + fic}
    ref_of = lambda s_, t_: dec_id[s_] if dec[s_].outcome == "SELF_FICTION" else dec_id[t_]
    dropped = [p for p in pairs if ref_of(p["source"], p["target"]) in secret_ids]
    pairs = [p for p in pairs if p not in dropped]
    if dropped:
        print(f"[two_phase] dropped {len(dropped)} pairs: the fiction token at the decision row is a secret "
              "token, so the margin is undefined (rule fixed in the PREREG)", flush=True)

    def margin(lp, s_, t_):
        return float(np.max(lp[secret_ids]) - lp[ref_of(s_, t_)])

    ctx = {}
    for k in sorted({p["target"] for p in pairs}):
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
        commit = dec[k].commit_row if dec[k].commit_row is not None else dec[k].row
        ctx[k] = {"hist": hist, "texts": texts, "cursor": cur, "snap": rng_snapshot(torch),
                  "n": min(n_tok, int(commit) + EXTRA_TOKENS)}

    def gen(k, layer, vec):
        c = ctx[k]
        rng_restore(torch, c["snap"])
        HOOK.update(layer=layer, row=dec[k].row,
                    vec=None if vec is None else torch.tensor(np.asarray(vec, np.float32), device=mw.device))
        STEP["n"] = 0
        tr = mw.gen_with_trace(c["hist"], n=c["n"], temp=temp, layer_lo=sup_lo, layer_hi=read_hi,
                               turn_idx=T, global_start=c["cursor"])
        HOOK.update(layer=None, row=None, vec=None)
        toks = [f["token"] for f in tr["frames"]]
        r = dec[k].row
        phi = tr["frames"][r]["phi"].astype(np.float64) if r < len(tr["frames"]) else None
        lp = STASH["scores"][r] if r < len(STASH["scores"]) else None
        return toks, phi, lp

    def outcome(k, toks):
        d = classify({**{t: [x] for t, x in ctx[k]["texts"].items()}, T: toks}, args["secret"], T, anchor)
        return d.outcome, d.content

    base = {}
    for k in ctx:
        toks, phi, lp = gen(k, None, None)
        cap_toks, _ = captured_turn(z, k, T)
        if toks != cap_toks[:len(toks)] or phi is None:
            release()
            raise ContractError(f"target {k}: unpatched run does not reproduce captured turn {T}")
        oc, ct = outcome(k, toks)
        oc_full, ct_full = outcome(k, cap_toks)
        if (oc, ct) != (oc_full, ct_full):
            release()
            raise ContractError(f"target {k}: cutting the turn at commit + {EXTRA_TOKENS} changes its class "
                                f"({oc} {ct} vs {oc_full} {ct_full}); raise EXTRA_TOKENS")
        base[k] = {"phi": phi, "lp": lp, "outcome": oc}
        for l in layers:
            stoks, sphi, slp = gen(k, l, H[(k, l)])
            if abs(float(np.max(slp[secret_ids]) - np.max(lp[secret_ids]))) > SELF_TOL or \
                    kl_from_logp(slp, lp) > SELF_TOL ** 2:
                release()
                raise ContractError(f"target {k} layer {l}: self-patch changes the logits; the hook does not "
                                    "write where phi is read")
    print(f"[two_phase] gates passed: {len(ctx)} targets reproduce and self-patch at {layers}", flush=True)

    units, skipped, rng = [], [], np.random.default_rng(NULL_SEED)
    t0 = time.time()
    for i, p in enumerate(pairs):
        s, t = p["source"], p["target"]
        sgn = 1.0 if dec[s].outcome == "TRUTH" else -1.0
        b = base[t]
        lpb = b["lp"]
        done = 0
        for l in layers:
            try:
                vecs, meta = build_doses(H[(s, l)], H[(t, l)], axes[l]["uL"], axes[l]["uR"], Xcal[l], rng,
                                         axes[l]["e2"], sgn)
            except Degenerate as e:
                skipped.append({"pair": [s, t], "layer": l, "why": str(e)})
                print(f"[two_phase] skip {s}->{t} L{l}: {e}", flush=True)
                continue
            done += 1
            u = {"cluster": t, "pair": [s, t], "dir": p["dir"], "equal_length": p["equal_length"],
                 "same_token": p["same_token"], "layer": l,
                 "primary": l in PRIMARY_LAYERS, **meta, "toward": {}, "KL": {}, "dS": {}, "dC": {},
                 "outcome": {}, "flip": {}, "dS_by_layer": {}, "dC_by_layer": {}}
            for m in MODES:
                toks, phi, lp = gen(t, l, vecs[m])
                if phi is None or lp is None:
                    release()
                    raise ContractError(f"pair {s}->{t} L{l} {m}: generation stopped before the decision row")
                rb = phi[l - sup_lo]
                if np.linalg.norm(rb - vecs[m]) > READBACK_TOL * max(np.linalg.norm(vecs[m]), 1e-12):
                    release()
                    raise ContractError(f"pair {s}->{t} L{l} {m}: read-back gate failed (the traced residual "
                                        f"is not the injected vector: rel {np.linalg.norm(rb - vecs[m]) / np.linalg.norm(vecs[m]):.2e})")
                oc, ct = outcome(t, toks)
                u["toward"][m] = sgn * (margin(lp, s, t) - margin(lpb, s, t))
                u.setdefault("toward_lp", {})[m] = sgn * float(np.max(lp[secret_ids]) - np.max(lpb[secret_ids]))
                u["KL"][m] = kl_from_logp(lp, lpb)
                u["outcome"][m], u["flip"][m] = oc, oc != b["outcome"]
                inj = vecs[m] - H[(t, l)]
                dss, dcs = {}, {}
                for l2 in range(l + 1, min(l + SPAN, read_hi - 1) + 1):
                    resp = (phi[l2 - sup_lo] - b["phi"][l2 - sup_lo]) - inj
                    dss[l2], dcs[l2] = rulers[l2].responses(resp, {s, t})
                u["dS_by_layer"][m], u["dC_by_layer"][m] = dss, dcs
                u["dS"][m] = float(np.mean(list(dss.values()))) if dss else float("nan")
                u["dC"][m] = float(np.mean(list(dcs.values()))) if dcs else float("nan")
            units.append(u)
        el = time.time() - t0
        print(f"[two_phase] pair {i + 1}/{len(pairs)} {s}->{t} {p['dir']} · "
              + " ".join(f"L{u['layer']} C={u['toward']['C']:+.2f} S+={u['toward']['S+']:+.2f} "
                         f"nat={u['toward']['nat']:+.2f}" for u in (units[-done:] if done else []))
              + f" · {el / (i + 1):.0f} s/pair", flush=True)
    release()

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = Path(a.capture_dir) / "two_phase" / stamp
    out.mkdir(parents=True)
    with open(out / "units.jsonl", "w") as fh:
        for u in units:
            fh.write(json.dumps(u, default=float) + "\n")
    meta = {"version": VERSION, **sc.stamp(), "capture": Path(a.capture_dir).name, "layers": layers,
            "primary_layers": list(PRIMARY_LAYERS), "pairs": pairs, "skipped": skipped, "dropped": dropped, "truth": truth, "self_fiction": fic,
            "secret_token_ids": secret_ids, "extra_tokens": EXTRA_TOKENS, "span": SPAN, "n_pcs": N_PCS,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "run.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(f"[two_phase] wrote {out}. Next: python two_phase.py analyze {out}")
    return 0


# ============================================================ analyze (CPU) ==
def cmd_analyze(a) -> int:
    d = Path(a.run_dir)
    meta = json.loads((d / "run.json").read_text())
    units = [json.loads(x) for x in (d / "units.jsonl").read_text().splitlines() if x.strip()]
    prereg = Path(__file__).resolve().parent / "PREREG_TWOPHASE.md"
    want = None
    if prereg.exists():
        for line in prereg.read_text().splitlines():
            if line.strip().startswith("| `two_phase.py` |"):
                want = line.split("`")[3]
    if want and want != meta["code_sha256"]:
        print(f"[analyze] WARNING: the run used two_phase.py {meta['code_sha256'][:12]}, the PREREG pins "
              f"{want[:12]}. The verdict below is NOT the preregistered one.")
    prim = [u for u in units if u["primary"]]
    v = verdict(prim)
    f = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:+.3f}"
    md = [f"# two_phase v{meta['version']} · `{meta['capture']}`", "",
          f"calibration_id `{meta['calibration_id']}` · primary layers {meta['primary_layers']} · "
          f"{v['n_units']} units in {v['n_clusters']} target draws · code `{meta['code_sha256'][:12]}`", "",
          f"**{v['reading']}**", "",
          "| test | statistic | estimate | 90% CI (cluster bootstrap) | margin | result |", "|---|---|---|---|---|---|",
          f"| G0 replication | mean toward full · median C/full | {f(v['G0']['mean_toward_full'])} · "
          f"{f(v['G0']['median_C_over_full'])} | p {v['G0']['p_full_gt_0']:.4f} | p < 0.05 · ≥ {RATIO_GATE} | "
          f"{'pass' if v['G0']['pass'] else 'FAIL'} |"]
    for h, margin in (("H1", "> 0"), ("H2", f"inside ±{MARGIN_R}"), ("H3a", f"< {MARGIN_Q}"), ("H3b", f"< {MARGIN_Q}")):
        x = v[h]
        md.append(f"| {h} | {x['stat']} | {f(x['est'])} | [{f(x['ci90'][0])}, {f(x['ci90'][1])}] | {margin} | "
                  f"{'pass' if x['pass'] else 'not met'} |")
    md += ["", "## Descriptive, by layer (no verdict)", "",
           f"| layer | units | mean toward {' / '.join(MODES)} | mean KL C / S+ / nat / iso | "
           "dS C / nat | dC S+ / nat | flips full / C / S+ / nat | median energy S |",
           "|---|---|---|---|---|---|---|---|"]
    for l in meta["layers"]:
        us = [u for u in units if u["layer"] == l]
        mt = lambda k, m: np.mean([u[k][m] for u in us])
        md.append(f"| {l}{'' if l in meta['primary_layers'] else '*'} | {len(us)} | "
                  + " / ".join(f"{mt('toward', m):+.2f}" for m in MODES) + " | "
                  + " / ".join(f"{mt('KL', m):.3f}" for m in ("C", "S+", "nat", "iso")) + " | "
                  f"{mt('dS', 'C'):.2f} / {mt('dS', 'nat'):.2f} | {mt('dC', 'S+'):.2f} / {mt('dC', 'nat'):.2f} | "
                  + " / ".join(str(sum(u['flip'][m] for u in us)) for m in ("full", "C", "S+", "nat")) +
                  f" | {np.median([u['energy_S'] for u in us]):.4f} |")
    md += ["", "## Sensitivity (secondary, no verdict of its own)", ""]
    for name, keep in (("equal history length only", lambda u: u["equal_length"]),
                       ("decision tokens differ at the anchor row", lambda u: not u.get("same_token", False)),
                       ("direction T->F only", lambda u: u["dir"] == "T->F"),
                       ("direction F->T only", lambda u: u["dir"] == "F->T")):
        sub = [u for u in prim if keep(u)]
        if len({u["cluster"] for u in sub}) >= 3:
            md.append(f"- {name} ({len(sub)} units): {verdict(sub, n_boot=2000)['reading']}")
        else:
            md.append(f"- {name}: fewer than 3 target draws, not read")
    lp = [u for u in prim if "toward_lp" in u]
    if lp:
        md.append(f"- with log p(secret) alone instead of the margin: mean toward S+ "
                  f"{np.mean([u['toward_lp']['S+'] for u in lp]):+.3f}, nat {np.mean([u['toward_lp']['nat'] for u in lp]):+.3f}, "
                  f"C {np.mean([u['toward_lp']['C'] for u in lp]):+.3f}")
    if meta.get("skipped"):
        md.append(f"- skipped (pair, layer): {len(meta['skipped'])}, see run.json")
    md += ["", "\\* exploratory layer. One capture, one base seed: whatever the reading, it is provisional."]
    (d / "TWO_PHASE.md").write_text("\n".join(md) + "\n")
    (d / "verdict.json").write_text(json.dumps(v, indent=2, default=float) + "\n")
    print("\n".join(md))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("capture_dir")
        p.add_argument("--layers", nargs="+", type=int, default=None)
        p.add_argument("--max-pairs", type=int, default=MAX_PAIRS)
    p = sub.add_parser("analyze")
    p.add_argument("run_dir")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    return {"plan": cmd_plan, "run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
