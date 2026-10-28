#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRAD rings v0.1 — aros: un mapa de la red que no sabe qué buscar
TwoQuarks Research — Luis Jaime Ledesma Perez

Why this exists
---------------
L/R/N, curvature and the Triad measure what WE decided to measure.  That is
right once something concrete is known, but it smuggles our hypothesis into
every exploration.  Rings measure the shape of the hidden states first and
only afterwards look at which outcome landed where.

The instrument, two levels
--------------------------
1. OUTER RINGS (Mapper; Singh, Memoli & Carlsson 2007).
   - Cloud X: phi at ONE layer for the selected frames (default: every
     analysis frame of every draw).  Reduced to its top `reduce_dim`
     principal coordinates (outcome-blind).
   - Lens f = first principal coordinate.  Its axis is the axis of the rings.
   - Cover: `n_rings` intervals of f with fractional `overlap`.  In the
     original space each interval is a slab: a cylindrical ring around the
     flow.
   - Inside each ring: single-linkage at one global scale
     eps = link_scale * median(k-th nearest-neighbour distance).
     Each cluster is a NODE (the ring's focus).
   - Two nodes in neighbouring rings that share frames are joined: "each ring
     joins its nearest and most affine ring".  Result: a graph.

2. SEVEN CYLINDERS PER GROUP (tomography).
   A node can look homogeneous from outside and still hide structure: in high
   dimension a gap of a few sigma is invisible to nearest-neighbour distances.
   For every node with >= min_group frames:
   - Cross-section: the axis that DEFINED the node (the ring's lens, plus any
     split axis used to reach it) is projected out.  A cylinder looks at the
     disc, not along its own axis.  (Without this, slicing makes every slab
     uniform along the lens and uniform slabs read as false splits.)
   - The section is reduced to its top variance coordinates
     (section_dim, capped at m // 2 so small groups do not overfit their PCA).
   - `n_cylinders` lenses: the section's own top-K principal axes.  Chosen by
     the data, blind to outcomes and to L/R.  (A seeded random rotation of that
     plane is kept as cylinder_mode="rotated"; the selftest showed it dilutes a
     one-axis split: a random 7-D direction needs a1^2 > ~0.56 to see it.)
   - Each cylinder i sees a 1-D projection and either splits it in two
     (exact 1-D 2-means, within/total SS < split_ratio, each side >= min_side)
     or not.  That partition is Phi(r_i).
   - THE TEST: R_min, the within/total SS of the strongest cylinder, against
     the null.  One statistic, no multiplicity.  (v0.1 draft tested the counts
     S and P directly; they are too coarse for small groups and lost power.)
   - DESCRIPTION (Isomeric Polarization, cylinders as views):
         P = mean_{i<j} VI_norm(Phi_i, Phi_j)
     plus S = fraction of cylinders that split.
     P high, S low: a minority of views sees it ("POLARIZED").
     S >= 1/2: the views agree the group is two ("SPLIT").
   - Null (parametric, SigClust-style; Liu et al. 2008): Gaussian groups with
     the SAME covariance spectrum and no clusters, run through the identical
     cylinders.  p_R empirical; flag if p_R < alpha.
     Caveat: real activations are heavy-tailed; a Gaussian null can call tails
     "structure".  Check flagged groups against a resampling null before claims.
   - A flagged group is ZOOMED: consensus split among the splitting cylinders
     gives sub-groups; tomography repeats inside them (depth <= max_depth).

Resolution stability
--------------------
The number of rings is the most arbitrary choice of Mapper.  The map is built
at 5 AND 7 rings; a flagged frame set that survives both is the model's, one
that appears at a single resolution is the instrument's.  Reported as Jaccard.

Contract
--------
- Outcome-blind: map_cloud(X, cfg) receives only X.  Outcomes (crad_contract
  classify) are attached AFTER the map exists, as a readout.
- Same clock as crad_contract: a frame is row t of a turn.  Decision rows
  (anchor / commit) come from crad_contract, never from this file.
- Deterministic and row-order invariant: every random choice is seeded from
  cfg.seed and the SORTED frame ids of the group.
- All constants are declared in RingsConfig and written next to every result.
  Change them BEFORE looking at outcomes, never after.

What this file is NOT
---------------------
- Not evidence about real models.  Selftests are instrument tests on planted data.
- Not causal.  A map says where frames sit, not why the model chose.
- A single capture is a probe, not a test.

CLI
    python crad_rings.py --selftest
    python crad_rings.py map captures/<id> --layer 17 [--turns 7] [--rings 5 7]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

VERSION = "0.1-rings"
TOL = 1e-10


@dataclass(frozen=True)
class RingsConfig:
    n_rings: int = 7
    overlap: float = 0.30          # fraction of a ring shared with its neighbour
    reduce_dim: int = 64           # outer map works on the top principal coordinates
    knn_k: int = 5
    link_scale: float = 1.25       # eps = link_scale * median k-NN distance
    min_node: int = 3              # smaller clusters are kept as nodes but never zoomed
    n_cylinders: int = 7
    cylinder_mode: str = "pcs"     # "pcs": the section's own top-K principal axes (data-chosen,
                                   # outcome-blind). "rotated": seeded random rotation of that
                                   # plane; v0.1 selftest showed it dilutes a 1-axis split
                                   # (a random 7-D direction rarely aligns: needs a1^2 > ~0.56)
    section_dim: int = 48          # cross-section truncated to its top variance coordinates,
                                   # capped at m // 2 (small groups overfit their own PCA)
    split_ratio: float = 0.25      # 1-D 2-means within/total SS; one Gaussian gives ~0.36
    min_side: float = 0.10         # each side of a split holds >= 10% of the group
    min_group: int = 20            # tomography only on groups this large
    n_null: int = 199
    alpha: float = 0.05
    max_depth: int = 2
    seed: int = 20261021


# ------------------------------------------------------------------ helpers --
def _seed_for(cfg: RingsConfig, *parts) -> int:
    h = hashlib.sha256(str(cfg.seed).encode())
    for p in parts:
        h.update(np.ascontiguousarray(p).tobytes() if isinstance(p, np.ndarray) else str(p).encode())
    return int.from_bytes(h.digest()[:8], "little")


def _pca(X: np.ndarray, q: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Top-q principal coordinates. Returns (Z [n,q'], Vt [q',d], S [q'], mu).
    Signs fixed so the largest-|loading| entry of every axis is positive."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    Xc = X - mu
    n, d = Xc.shape
    if n <= d:
        w, V = np.linalg.eigh(Xc @ Xc.T)
        order = np.argsort(w)[::-1]
        w, V = np.maximum(w[order], 0.0), V[:, order]
        S = np.sqrt(w)
        keep = S > TOL * max(S[0], TOL)
        S, V = S[keep], V[:, keep]
        Vt = (V.T @ Xc) / S[:, None]
    else:
        w, V = np.linalg.eigh(Xc.T @ Xc)
        order = np.argsort(w)[::-1]
        w, V = np.maximum(w[order], 0.0), V[:, order]
        S = np.sqrt(w)
        keep = S > TOL * max(S[0], TOL)
        S, Vt = S[keep], V[:, keep].T
    q = min(q, len(S))
    S, Vt = S[:q], Vt[:q]
    sgn = np.sign(Vt[np.arange(q), np.argmax(np.abs(Vt), axis=1)])
    sgn[sgn == 0] = 1.0
    Vt = Vt * sgn[:, None]
    return Xc @ Vt.T, Vt, S, mu


def _pairwise_chunks(A: np.ndarray, B: np.ndarray, chunk: int = 1024):
    bb = np.sum(B * B, axis=1)
    for i in range(0, len(A), chunk):
        a = A[i:i + chunk]
        d2 = np.sum(a * a, axis=1)[:, None] + bb[None, :] - 2.0 * a @ B.T
        yield i, np.sqrt(np.maximum(d2, 0.0))


def knn_scale(Z: np.ndarray, k: int) -> float:
    k = max(1, min(k, len(Z) - 1))
    kth = np.empty(len(Z))
    for i, D in _pairwise_chunks(Z, Z):
        kth[i:i + len(D)] = np.partition(D, k, axis=1)[:, k]   # index 0 is the point itself
    return float(np.median(kth))


def eps_components(Z: np.ndarray, eps: float) -> np.ndarray:
    n = len(Z)
    if n == 1:
        return np.zeros(1, dtype=int)
    rows, cols = [], []
    for i, D in _pairwise_chunks(Z, Z):
        r, c = np.nonzero(D <= eps)
        rows.append(r + i); cols.append(c)
    r, c = np.concatenate(rows), np.concatenate(cols)
    g = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def cover(f: np.ndarray, n: int, overlap: float) -> List[np.ndarray]:
    lo, hi = float(np.min(f)), float(np.max(f))
    if hi - lo < TOL:
        return [np.arange(len(f))]
    L = (hi - lo) / (n - (n - 1) * overlap)
    step = L * (1.0 - overlap)
    rings = []
    for i in range(n):
        a = lo + i * step
        b = hi if i == n - 1 else a + L
        rings.append(np.flatnonzero((f >= a - TOL) & (f <= b + TOL)))
    return rings


# ------------------------------------------------------------ outer rings --
def outer_graph(Z: np.ndarray, cfg: RingsConfig) -> Dict:
    f = Z[:, 0]
    eps = cfg.link_scale * knn_scale(Z, cfg.knn_k)
    nodes = []
    for ri, idx in enumerate(cover(f, cfg.n_rings, cfg.overlap)):
        if len(idx) == 0:
            continue
        lab = eps_components(Z[idx], eps)
        for c in np.unique(lab):
            nodes.append({"id": len(nodes), "ring": ri, "members": np.sort(idx[lab == c])})
    edges = []
    for a in nodes:
        for b in nodes:
            if b["ring"] == a["ring"] + 1 and len(np.intersect1d(a["members"], b["members"], assume_unique=True)):
                edges.append((a["id"], b["id"]))
    if nodes:
        n = len(nodes)
        e = np.array(edges, dtype=int).reshape(-1, 2)
        g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        comp = connected_components(g, directed=False)[1]
        for nd, c in zip(nodes, comp):
            nd["component"] = int(c)
    return {"nodes": nodes, "edges": edges, "eps": eps}


# ---------------------------------------------------------- 1-D cylinders --
def split_1d(x: np.ndarray, ratio: float, min_side: float) -> Tuple[np.ndarray, float]:
    """Exact 1-D 2-means. Returns (labels 0/1 or all 0, within/total SS)."""
    m = len(x)
    order = np.argsort(x, kind="stable")
    xs = x[order]
    tot = float(np.sum((xs - xs.mean()) ** 2))
    if tot < TOL or m < 4:
        return np.zeros(m, dtype=int), 1.0
    k0 = max(1, int(np.ceil(min_side * m)))
    ks = np.arange(k0, m - k0 + 1)
    if len(ks) == 0:
        return np.zeros(m, dtype=int), 1.0
    c1, c2 = np.cumsum(xs), np.cumsum(xs * xs)
    left_ss = c2[ks - 1] - c1[ks - 1] ** 2 / ks
    rn = m - ks
    right_ss = (c2[-1] - c2[ks - 1]) - (c1[-1] - c1[ks - 1]) ** 2 / rn
    within = left_ss + right_ss
    j = int(np.argmin(within))
    R = float(within[j] / tot)
    if R >= ratio:
        return np.zeros(m, dtype=int), R
    lab = np.zeros(m, dtype=int)
    lab[order[ks[j]:]] = 1
    return lab, R


def vi_norm(a: np.ndarray, b: np.ndarray) -> float:
    """Variation of information / log(m): 0 = same partition."""
    m = len(a)
    if m < 2:
        return 0.0
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    joint = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(joint, (ia, ib), 1.0)
    p = joint / m
    pa, pb = p.sum(1), p.sum(0)
    nz = p > 0
    H = lambda v: -float(np.sum(v[v > 0] * np.log(v[v > 0])))
    I = float(np.sum(p[nz] * np.log(p[nz] / (pa[:, None] * pb[None, :])[nz])))
    return (H(pa) + H(pb) - 2.0 * I) / np.log(m)


def cylinders(C: np.ndarray, cfg: RingsConfig, seed: int) -> Dict:
    """C: [m, r] section coordinates. n_cylinders blind lenses in its top-K plane."""
    C = C - C.mean(0)
    _, Vt, S, _ = _pca(C, cfg.n_cylinders)
    K = len(S)
    if K == 0:
        z = np.zeros(len(C), dtype=int)
        return {"labels": [z], "dirs": np.zeros((C.shape[1], 1)), "ratios": [1.0], "P": 0.0, "S": 0.0,
                "R_min": 1.0}
    if cfg.cylinder_mode == "rotated":
        Rot, Rr = np.linalg.qr(np.random.default_rng(seed).normal(size=(K, K)))
        Rot = Rot * np.sign(np.diag(Rr))[None, :]
        W = Vt.T @ Rot                               # [r, K] unit directions
    else:
        W = Vt.T                                     # the section's own principal axes
    proj = C @ W
    labels, ratios = [], []
    for k in range(K):
        lab, R = split_1d(proj[:, k], cfg.split_ratio, cfg.min_side)
        labels.append(lab); ratios.append(R)
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    P = float(np.mean([vi_norm(labels[i], labels[j]) for i, j in pairs])) if pairs else 0.0
    Sfrac = float(np.mean([l.max() > 0 for l in labels]))
    return {"labels": labels, "dirs": W, "ratios": ratios, "P": P, "S": Sfrac,
            "R_min": float(np.min(ratios))}


def _section(Y: np.ndarray, seen_axes: Sequence[np.ndarray], cfg: RingsConfig):
    """Project out the axes that defined the group, keep its top variance coordinates."""
    Yc = Y - Y.mean(0)
    for ax in seen_axes:
        u = ax / max(np.linalg.norm(ax), TOL)
        Yc = Yc - np.outer(Yc @ u, u)
    C, Vt, S, _ = _pca(Yc, max(cfg.n_cylinders, min(cfg.section_dim, len(Y) // 2)))
    return C, Vt, S


def tomography(Y: np.ndarray, ids: np.ndarray, seen_axes: Sequence[np.ndarray],
               cfg: RingsConfig, depth: int = 0) -> Dict:
    """Y: [m, d] the group's frames (any fixed coordinate system). ids: sorted frame ids."""
    m = len(Y)
    C, Vt, S, = _section(Y, seen_axes, cfg)
    seed = _seed_for(cfg, "tomo", depth, ids)
    obs = cylinders(C, cfg, seed)
    rng = np.random.default_rng(seed ^ 0x5EED)
    sd = S / np.sqrt(max(m - 1, 1))
    nP, nS, nR = np.empty(cfg.n_null), np.empty(cfg.n_null), np.empty(cfg.n_null)
    for b in range(cfg.n_null):
        G = rng.normal(size=(m, len(sd))) * sd[None, :]
        nb = cylinders(G, cfg, seed)
        nP[b], nS[b], nR[b] = nb["P"], nb["S"], nb["R_min"]
    pv = lambda null, x: float((1 + np.sum(null >= x - 1e-12)) / (cfg.n_null + 1))
    pP, pS = pv(nP, obs["P"]), pv(nS, obs["S"])
    pR = float((1 + np.sum(nR <= obs["R_min"] + 1e-12)) / (cfg.n_null + 1))   # stronger split = smaller R
    # the TEST is the strongest view (one statistic, no multiplicity); P and S DESCRIBE the structure
    flagged = pR < cfg.alpha and obs["S"] > 0
    out = {"depth": depth, "size": m, "P": obs["P"], "S": obs["S"], "R_min": obs["R_min"],
           "p_R": pR, "p_P": pP, "p_S": pS, "null_R_q05": float(np.quantile(nR, 0.05)),
           "flagged": bool(flagged), "reason": [], "subgroups": []}
    if not flagged:
        return out
    out["reason"].append("SPLIT, views agree" if obs["S"] >= 0.5 else
                         "POLARIZED, a minority of views sees it")
    # consensus among the cylinders that split: co-association >= 1/2
    split_idx = [k for k, l in enumerate(obs["labels"]) if l.max() > 0]
    L = np.stack([obs["labels"][k] for k in split_idx], 0)
    same = np.zeros((m, m))
    for row in L:
        same += (row[:, None] == row[None, :])
    same /= len(split_idx)
    r, c = np.nonzero(same >= 0.5)
    comp = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(m, m)), directed=False)[1]
    split_axes_full = [Vt.T @ obs["dirs"][:, k] for k in split_idx]   # back to Y's coordinates
    for cc in np.unique(comp):
        sel = np.flatnonzero(comp == cc)
        sub = {"ids": ids[sel], "size": int(len(sel))}
        if depth + 1 < cfg.max_depth and len(sel) >= cfg.min_group:
            sub["tomography"] = tomography(Y[sel], ids[sel], list(seen_axes) + split_axes_full, cfg, depth + 1)
        out["subgroups"].append(sub)
    return out


# --------------------------------------------------------------- the map --
def map_cloud(X: np.ndarray, cfg: RingsConfig = RingsConfig(),
              frame_ids: Optional[np.ndarray] = None) -> Dict:
    """Outcome-blind: receives ONLY the frames (and their identities, for seeding).
    Every random choice is seeded by frame IDENTITY, never by row position."""
    fid = np.arange(len(X)) if frame_ids is None else np.asarray(frame_ids, dtype=np.int64)
    Z, Vt, S, mu = _pca(X, cfg.reduce_dim)
    g = outer_graph(Z, cfg)
    lens_axis = np.zeros(Z.shape[1]); lens_axis[0] = 1.0     # lens = first coordinate of Z
    for nd in g["nodes"]:
        ids = nd["members"]
        nd["size"] = int(len(ids))
        if len(ids) >= max(cfg.min_group, cfg.min_node):
            o = np.argsort(fid[ids], kind="stable")          # rows ordered by identity
            t = tomography(Z[ids[o]], fid[ids[o]], [lens_axis], cfg)
            pos = {int(f): int(i) for f, i in zip(fid[ids], ids)}
            _ids_to_rows(t, pos)
            nd["tomography"] = t
    return {"graph": g, "n_frames": int(len(X)), "reduce_dim": int(Z.shape[1]),
            "explained": float(np.sum(S ** 2) / max(np.sum((X - mu) ** 2), TOL))}


def _ids_to_rows(t: Dict, pos: Dict[int, int]) -> None:
    """Tomography works on identities; the map reports row positions."""
    for sg in t["subgroups"]:
        sg["ids"] = np.sort(np.array([pos[int(f)] for f in sg["ids"]], dtype=np.int64))
        if "tomography" in sg:
            _ids_to_rows(sg["tomography"], pos)


def flagged_frames(res: Dict) -> np.ndarray:
    out = []
    def walk(t):
        if t.get("flagged"):
            for s in t["subgroups"]:
                out.append(s["ids"])
                if "tomography" in s:
                    walk(s["tomography"])
    for nd in res["graph"]["nodes"]:
        if "tomography" in nd:
            walk(nd["tomography"])
    return np.unique(np.concatenate(out)) if out else np.array([], dtype=int)


def jaccard(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    u = len(np.union1d(a, b))
    return None if u == 0 else len(np.intersect1d(a, b)) / u


def signature(res: Dict) -> str:
    h = hashlib.sha256()
    for nd in res["graph"]["nodes"]:
        h.update(nd["members"].astype(np.int64).tobytes())
        t = nd.get("tomography")
        if t:
            h.update(f"{t['P']:.12g}|{t['S']:.12g}|{t['flagged']}".encode())
    h.update(flagged_frames(res).astype(np.int64).tobytes())
    return h.hexdigest()[:16]


def multi_resolution(X: np.ndarray, rings: Sequence[int], cfg: RingsConfig) -> Dict:
    maps = {n: map_cloud(X, replace(cfg, n_rings=n)) for n in rings}
    fl = {n: flagged_frames(m) for n, m in maps.items()}
    rs = sorted(rings)
    stab = {f"{a}v{b}": jaccard(fl[a], fl[b]) for i, a in enumerate(rs) for b in rs[i + 1:]}
    stable = fl[rs[0]]
    for n in rs[1:]:
        stable = np.intersect1d(stable, fl[n])
    return {"maps": maps, "flagged": fl, "stability_jaccard": stab, "stable_flagged": stable}


# ---------------------------------------------------------- capture path --
def load_cloud(cd: Path, layer: int, turns: Optional[Sequence[int]], points: str):
    cap = json.loads((cd / "capture.json").read_text())
    if cap.get("problems"):
        raise SystemExit(f"[rings] capture is not OK: {cap['problems']}")
    cfgc = json.loads((cd / "config.json").read_text())
    with np.load(cd / "capture.npz", allow_pickle=False) as zf:
        z = {k: zf[k] for k in zf.files}
    sup_lo, sup_hi = json.loads(str(z["meta"]))["superset"]
    if not (sup_lo <= layer < sup_hi):
        raise SystemExit(f"[rings] layer {layer} outside captured superset [{sup_lo},{sup_hi})")
    mask = (z["kind"] == 1) if points == "analysis" else np.ones(len(z["kind"]), bool)
    if turns:
        mask &= np.isin(z["turn"], list(turns))
    idx = np.flatnonzero(mask)
    X = z["phi"][idx, layer - sup_lo].astype(np.float64)
    return X, idx, z, cap, cfgc


def decisions(z: dict, idx: np.ndarray, secret: str, target_turn: int, anchor: str) -> List[Dict]:
    from crad_contract import classify, turns_from_capture        # the contract owns the clock
    pos = {int(g): i for i, g in enumerate(idx)}
    out = []
    for k in sorted(set(z["draw"][z["kind"] == 1].tolist())):
        toks, glob = turns_from_capture(z, int(k))
        d = classify(toks, secret, target_turn, anchor, glob)
        row = {"draw": int(k), "outcome": d.outcome, "content": d.content,
               "anchor_row": d.row, "commit_row": d.commit_row, "frames": {}}
        for name, r in (("anchor", d.row), ("commit", d.commit_row)):
            if r is None:
                continue
            sel = np.flatnonzero((z["kind"] == 1) & (z["draw"] == k) & (z["turn"] == target_turn)
                                 & (z["token_in_turn"] == r))
            if len(sel) == 1 and int(sel[0]) in pos:
                row["frames"][name] = pos[int(sel[0])]
        out.append(row)
    return out


def locate(res: Dict, frame: int) -> List[Dict]:
    """Every node holding this frame (overlap: up to two) and its sub-group path inside."""
    def walk(t, path, acc):
        for si, s in enumerate(t.get("subgroups", [])):
            if frame in s["ids"]:
                p = path + [si]
                acc.append({"path": p, "flagged_at": t["depth"], "reason": t["reason"]})
                if "tomography" in s:
                    walk(s["tomography"], p, acc)
        return acc
    out = []
    for nd in res["graph"]["nodes"]:
        if frame in nd["members"]:
            t = nd.get("tomography")
            out.append({"node": nd["id"], "ring": nd["ring"], "component": nd.get("component"),
                        "node_flagged": bool(t and t["flagged"]),
                        "inside": walk(t, [], []) if t else []})
    return out


def cmd_map(a) -> int:
    cd = Path(a.capture_dir)
    X, idx, z, cap, cfgc = load_cloud(cd, a.layer, a.turns, a.points)
    cfg = RingsConfig(seed=a.seed)
    print(f"[rings] {len(X)} frames · layer {a.layer} · d={X.shape[1]} · rings {a.rings}", flush=True)
    t0 = time.time()
    mr = multi_resolution(X, a.rings, cfg)
    decs = decisions(z, idx, cfgc["args"]["secret"], a.target_turn, a.anchor)
    primary = max(a.rings)
    for d in decs:
        d["where"] = {n: {al: locate(mr["maps"][n], f) for al, f in d["frames"].items()} for n in a.rings}

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = cd / "rings" / f"{stamp}_L{a.layer}"
    out.mkdir(parents=True)

    def node_rows(res):
        rows = []
        for nd in res["graph"]["nodes"]:
            t = nd.get("tomography")
            rows.append({"node": nd["id"], "ring": nd["ring"], "component": nd.get("component"), "size": nd["size"],
                         **({} if not t else {"P": t["P"], "S": t["S"], "R_min": t["R_min"], "p_R": t["p_R"],
                                              "p_P": t["p_P"], "p_S": t["p_S"],
                                              "flagged": t["flagged"], "reason": t["reason"],
                                              "subgroups": [s["size"] for s in t["subgroups"]]})})
        return rows

    meta = {"rings_version": VERSION, "config": asdict(cfg), "capture": cd.name, "layer": a.layer,
            "turns": a.turns, "points": a.points, "n_frames": len(X),
            "capture_npz": cap.get("facts", {}).get("capture_npz_sha256"),
            "explained_by_reduce_dim": {n: m["explained"] for n, m in mr["maps"].items()},
            "signature": {n: signature(m) for n, m in mr["maps"].items()},
            "stability_jaccard": mr["stability_jaccard"], "n_stable_flagged": int(len(mr["stable_flagged"])),
            "nodes": {n: node_rows(m) for n, m in mr["maps"].items()},
            "decisions": decs, "wall_seconds": time.time() - t0}
    (out / "rings.json").write_text(json.dumps(meta, indent=2, default=lambda o: o.tolist()
                                               if isinstance(o, np.ndarray) else str(o)) + "\n")
    f = lambda x, n=3: "—" if x is None else f"{x:.{n}f}"
    md = [f"# Rings v{VERSION} — `{cd.name}` · layer {a.layer}", "",
          "**Outcome-blind map, outcomes attached afterwards. One capture: a probe, not a test.**", "",
          f"- frames {len(X)} ({a.points}{', turns ' + str(a.turns) if a.turns else ''}) · "
          f"reduce_dim {cfg.reduce_dim} explains {f(mr['maps'][primary]['explained'])} of the variance",
          f"- resolution stability (Jaccard of flagged frames): {mr['stability_jaccard']} · "
          f"stable flagged frames: {len(mr['stable_flagged'])}",
          f"- config: {asdict(cfg)}", ""]
    for n in a.rings:
        g = mr["maps"][n]["graph"]
        md += [f"## {n} rings · {len(g['nodes'])} nodes · {len(g['edges'])} edges · "
               f"{len(set(nd.get('component') for nd in g['nodes']))} components · eps {g['eps']:.3f}", "",
               "| node | ring | comp | size | R_min | p_R | P | S | flagged | sub-groups |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in node_rows(mr["maps"][n]):
            md.append(f"| {r['node']} | {r['ring']} | {r['component']} | {r['size']} | {f(r.get('R_min'))} "
                      f"| {f(r.get('p_R'), 4)} | {f(r.get('P'))} | {f(r.get('S'))} | {'**' + ', '.join(r['reason']) + '**' if r.get('flagged') else '—'} "
                      f"| {r.get('subgroups') or '—'} |")
        md.append("")
    md += [f"## Decision rows (contract) on the {primary}-ring map", "",
           "| draw | outcome | content | row | node | node flagged | inside (sub-group path) |", "|---|---|---|---|---|---|---|"]
    for d in decs:
        for al, hits in d["where"][primary].items():
            for h in hits:
                ins = "; ".join(f"{x['path']}" for x in h["inside"]) or "—"
                md.append(f"| {d['draw']} | {d['outcome']} | {d['content']} | {al} | {h['node']} "
                          f"| {'yes' if h['node_flagged'] else 'no'} | {ins} |")
    md += ["", "R_min = within/total SS of the strongest cylinder split (one Gaussian ~0.36). "
           "p_R = test against Gaussian groups with the same spectrum (the flag). "
           "P = mean VI between the cylinders' partitions (views contradict). S = fraction of cylinders that split."]
    (out / "RINGS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[rings] wrote {out}")
    return 0


# ------------------------------------------------------------ selftest --
def _blob(rng, n, d, center, sd):
    return center[None, :] + rng.normal(size=(n, d)) * sd


def _hidden_split_cloud(rng, n=300, d=128, gap=6.0, long_sd=15.0):
    """Elongated blob (PC1 = e) hiding a two-mode split along u orthogonal to e."""
    e = np.zeros(d); e[0] = 1.0
    u = np.zeros(d); u[1] = 1.0
    X = rng.normal(size=(n, d))
    X[:, 0] *= long_sd
    side = np.arange(n) % 2
    X[:, 1] += (side - 0.5) * gap
    return X, side


def _fake_capture(cd: Path, rng, d=64, layers=(13, 18), n_tok=24):
    """Minimal capture dir with the keys crad_contract and load_cloud read."""
    cols = {k: [] for k in ("kind", "branch", "draw", "turn", "token_in_turn", "global_token", "token", "token_id")}
    phis = []
    plan = {0: "TRUTH", 1: "SELF_FICTION", 2: "TRUTH", 3: "SELF_FICTION"}
    u = np.zeros(d); u[3] = 1.0
    g = 0
    for k, outcome in plan.items():
        for t in range(2, 8):
            if t == 6:
                toks = [" Once", " the", " code", " was", " GREEN", "DRAGON", "."]
            elif t == 7:
                tail = [" BLUE", "F", "AL", "CON", "."] if outcome == "TRUTH" else [" GREEN", "DRAGON", "."]
                toks = ["The", " access", " code", " is", *tail]
            else:
                toks = [" ok"]
            toks = (toks + [" ."] * n_tok)[:n_tok]
            for i, tok in enumerate(toks):
                phi = rng.normal(size=(layers[1] - layers[0], d))
                if t == 7 and i == 4:                               # anchor row carries the planted choice
                    phi += (6.0 if outcome == "TRUTH" else -6.0) * u
                for key, v in (("kind", 1), ("branch", -1), ("draw", k), ("turn", t), ("token_in_turn", i),
                               ("global_token", g), ("token", tok), ("token_id", 1000 + i)):
                    cols[key].append(v)
                phis.append(phi); g += 1
    cd.mkdir(parents=True, exist_ok=True)
    meta = {"superset": list(layers), "crad_band": [14, 17], "phi_dtype": "float16", "fp16_exact": True}
    np.savez(cd / "capture.npz", meta=np.array(json.dumps(meta)), phi=np.asarray(phis, dtype=np.float16),
             kind=np.asarray(cols["kind"], np.int8), branch=np.asarray(cols["branch"], np.int8),
             draw=np.asarray(cols["draw"], np.int32), turn=np.asarray(cols["turn"], np.int32),
             token_in_turn=np.asarray(cols["token_in_turn"], np.int32),
             global_token=np.asarray(cols["global_token"], np.int64), token=np.asarray(cols["token"], dtype=str),
             token_id=np.asarray(cols["token_id"], np.int64))
    (cd / "capture.json").write_text(json.dumps({"problems": [], "facts": {"capture_npz_sha256": "sha256:fake"}}))
    (cd / "config.json").write_text(json.dumps({"name": "fake", "args": {"secret": "BLUEFALCON"}}))
    return plan


def selftest() -> int:
    tests = []
    cfg = RingsConfig(n_null=99)

    # 1. two far blobs -> separated by the outer map (Mapper does its basic job)
    rng = np.random.default_rng(1)
    d = 128
    c = np.zeros(d); c[0] = 40.0
    X = np.vstack([_blob(rng, 150, d, np.zeros(d), 1.0), _blob(rng, 150, d, c, 1.0)])
    side = np.r_[np.zeros(150, int), np.ones(150, int)]
    res = map_cloud(X, cfg)
    pure = all(len(set(side[nd["members"]])) == 1 for nd in res["graph"]["nodes"])
    comps = len(set(nd["component"] for nd in res["graph"]["nodes"]))
    tests.append(("far blobs: outer map separates them (pure nodes, >= 2 components)", pure and comps >= 2))

    # 2. homogeneous anisotropic blobs -> per-group false flag rate near alpha
    fl = tot = 0
    for s in range(12):
        r = np.random.default_rng(100 + s)
        Xh = r.normal(size=(240, d)) * np.linspace(3.0, 0.5, d)[None, :]
        m = map_cloud(Xh, cfg)
        for nd in m["graph"]["nodes"]:
            if "tomography" in nd:
                tot += 1; fl += nd["tomography"]["flagged"]
    fp = fl / max(tot, 1)
    tests.append((f"homogeneous groups: false flag rate {fp:.3f} <= 2 alpha ({tot} groups)", fp <= 2 * cfg.alpha))

    # 3. hidden split: invisible to the outer map, found by the cylinders
    Xs, side_s = _hidden_split_cloud(np.random.default_rng(7), d=d)
    ms = map_cloud(Xs, cfg)
    mixed = [nd for nd in ms["graph"]["nodes"] if "tomography" in nd and len(set(side_s[nd["members"]])) == 2]
    hidden_from_outside = len(mixed) > 0
    det = [nd["tomography"]["flagged"] for nd in mixed]
    purity = []
    for nd in mixed:
        t = nd["tomography"]
        if t["flagged"]:
            for sg in t["subgroups"]:
                if sg["size"] >= 5:
                    purity.append(max(np.mean(side_s[sg["ids"]]), 1 - np.mean(side_s[sg["ids"]])))
    rate = float(np.mean(det)) if det else 0.0
    pur = float(np.mean(purity)) if purity else 0.0
    tests.append((f"hidden split: outer map mixes it, cylinders flag {rate:.2f} of those groups, "
                  f"sub-group purity {pur:.2f}", hidden_from_outside and rate >= 0.8 and pur >= 0.9))

    # 4. resolution stability: the hidden split survives 5 and 7 rings
    mr = multi_resolution(Xs, (5, 7), cfg)
    jac = mr["stability_jaccard"]["5v7"]
    tests.append((f"hidden split stable across 5 and 7 rings (Jaccard {jac if jac is None else round(jac, 2)})",
                  jac is not None and jac >= 0.5))

    # 5. deterministic and row-order invariant
    perm = np.random.default_rng(3).permutation(len(Xs))
    inv = np.argsort(perm)
    ma, mb = map_cloud(Xs, cfg), map_cloud(Xs[perm], cfg, frame_ids=perm)
    fa = flagged_frames(ma); fb = np.sort(perm[flagged_frames(mb)])
    same_twice = signature(ma) == signature(map_cloud(Xs, cfg))
    tests.append((f"deterministic (signature twice: {same_twice}) and row-order invariant given frame ids "
                  f"(flagged {len(fa)} vs {len(fb)})", same_twice and np.array_equal(fa, fb)))
    del inv

    # 6. capture path end to end on a fake capture (needs crad_contract)
    try:
        import tempfile
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import crad_contract  # noqa: F401
        with tempfile.TemporaryDirectory() as td:
            cd = Path(td) / "cap"
            plan = _fake_capture(cd, np.random.default_rng(11))
            Xc, idx, z, _, cfgc = load_cloud(cd, 15, [7], "analysis")
            decs = decisions(z, idx, cfgc["args"]["secret"], 7, r"access code is")
            ok = [dd["outcome"] for dd in decs] == [plan[k] for k in sorted(plan)]
            ok &= all("anchor" in dd["frames"] for dd in decs)
            a = argparse.Namespace(capture_dir=str(cd), layer=15, turns=[7], points="analysis", rings=[5, 7],
                                   target_turn=7, anchor=r"access code is", seed=cfg.seed)
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                cmd_map(a)
            ok &= len(list((cd / "rings").glob("*/RINGS.md"))) == 1
        tests.append(("capture path: contract decisions located, RINGS.md written", bool(ok)))
    except ImportError:
        tests.append(("capture path skipped (crad_contract not importable)", True))

    print(f"CRAD rings v{VERSION} selftest")
    for name, good in tests:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    return 0 if all(g for _, g in tests) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="CRAD rings: outcome-blind map + seven-cylinder tomography")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("map")
    p.add_argument("capture_dir")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--turns", nargs="+", type=int, default=None)
    p.add_argument("--points", choices=["analysis", "all"], default="analysis")
    p.add_argument("--rings", nargs="+", type=int, default=[5, 7])
    p.add_argument("--target-turn", type=int, default=7)
    p.add_argument("--anchor", default=r"access code is")
    p.add_argument("--seed", type=int, default=RingsConfig.seed)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "map":
        return cmd_map(a)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
