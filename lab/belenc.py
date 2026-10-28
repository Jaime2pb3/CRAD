#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BELENC v0.1 — Base Estándar para Localizaciones Espaciales No Conocidas
TwoQuarks Research — Luis Jaime Ledesma Perez

Por qué existe
--------------
Cada instrumento del lab (aros, tubo, proyección L/R, PCA, patch) devuelve números, pero ninguno
tenía una escala contra la cual leerlos. Un negativo ("los aros no ven la decisión") no distingue
"no está" de "está por debajo de lo que el instrumento ve". BELENC es esa escala: perturbaciones
CONOCIDAS plantadas dentro de las activaciones REALES del modelo (método de adición de estándar:
el analito se añade a la muestra, no a agua destilada, para que el efecto matriz cuente), y la
validación de cada instrumento con los parámetros de ICH Q2(R2).

La matriz
---------
phi real de una captura del lab, una capa, un turno, alineado por (draw, fila) con el mismo reloj
del contrato. Nunca un fondo sintético (salvo en --selftest, donde se declara).

Catálogo de analitos (localización, dirección y dosis conocidas)
  shift      fila r*: los draws de S se desplazan a·σ_u a lo largo de u          (un precursor)
  split      fila r*: S en +a/2·σ_u, el resto en -a/2·σ_u                         (dos poblaciones)
  fork       filas > r*: los draws de S se desplazan a·σ_u y se quedan ahí       (bifurcación)
  precursor  fork fijo de 8σ en w desde r*+1, y SOLO en la fila r* los de S se mueven a·σ_u en u
             (anticipación transitoria: aparece y se va)
  precursor_persist  igual, pero el desplazamiento en u empieza en r* y PERSISTE en las filas siguientes
             (lo plausible en un residual que acumula). Los dos quedan en el catálogo: su diferencia de LOD
             mide cuánto le cuesta al tubo que el precursor también viva en la velocidad.
             Nota del 22-oct: la primera corrida dio LOD = None para ambos y lo atribuí a ese efecto. Era
             falso: con 19 barajas el p mínimo es 0.05 y nunca podía bajar de α. Ver la compuerta de
             resolución en run_standard.
  blanco     a = 0 en todos: mide la especificidad sobre la matriz real
S = la mitad de los draws, al azar en cada réplica. Ciego para todos los instrumentos salvo el oráculo.

Familias de dirección (sobre la nube del turno centrada por fila, n ≈ draws × filas)
  high    componentes principales 1-5             (lo que domina la varianza)
  low     componentes en la mitad baja del espectro con varianza todavía confiable
  random  dirección isotrópica en d dimensiones
  real    h_src - h_tgt de un par real en su fila de ancla (opcional, --real-pair 5 0)
σ_u = desviación natural de la matriz a lo largo de u. La dosis está en σ locales: 2 significa lo
mismo en L14 que en L21.

Instrumentos bajo validación
  oracle        TECHO, conoce S: eje diferencia de medias leave-one-out en r*, éxito si AUC >= 0.9
  pca           ciego: primer eje de la rebanada r*, éxito si traslape² con u >= 0.5
  tube_fork     ciego: crad_tube marca bifurcación en la transición r* -> r*+1
  tube_anticip  ciego: crad_tube dice que el estado en r* anticipa el lado (p_enc < 0.05)
  rings         ciego: crad_rings marca un nodo con filas r* y su partición separa S con pureza >= 0.8

Validación (ICH Q2(R2), adaptado)
  especificidad   tasa de éxito con a = 0 (falsa localización), con IC de Wilson al 95%
  LOD95           dosis mínima con éxito >= 95% en ella y en todas las mayores
  LOQ             (oráculo) dosis mínima con error relativo de la dosis estimada <= 20%
  linealidad      pendiente y R² de la dosis estimada contra la plantada, por encima del LOD
  exactitud       traslape² medio con la dirección plantada (pca) y error de fila (tubo)
  precisión       IC de Wilson de cada tasa (réplicas con semillas independientes)

Cota teórica (teoría de matrices aleatorias)
  Modelo con picos: n muestras, dimensión efectiva d_eff = (tr Σ)² / tr(Σ²) (participation ratio),
  γ = d_eff / n, ruido ν = tr Σ / d_eff. Un pico de varianza añadida s² tiene θ = s² / ν.
  BBP (Baik, Ben Arous y Péché 2005): el primer eje solo se separa del bulto si θ > √γ.
  Traslape (Paul 2007): |<v,u>|² = (1 - γ/θ²) / (1 + γ/θ) para θ > √γ, 0 si no.
  BELENC reporta la dosis teórica donde el traslape esperado llega a 0.5 y la compara con el LOD
  empírico del pca. La brecha por familia de dirección ES el efecto matriz: cuánto se aleja la
  geometría real del supuesto isotrópico.

Lo que NO es
  - No es evidencia sobre el modelo. Es la escala con la que se leen los instrumentos.
  - Un LOD vale para ESTA matriz (capa, turno, fila, número de draws). Otra captura, otra ficha.
  - Dosis en σ geométricos. La conversión a nats (efecto conductual) la hace belenc_fisher.py.

CLI
  python belenc.py --selftest
  python belenc.py run captures/<id> --layer 17 [--turn 7] [--row 2] [--reps 20]
                   [--instruments oracle pca tube_fork tube_anticip rings] [--real-pair 5 0]
                   [--fisher-json captures/<id>/belenc_fisher/<stamp>/fisher.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

VERSION = "0.1-belenc"
ANALYTES = ("shift", "split", "fork", "precursor", "precursor_persist")
FAMILIES = ("high", "low", "random", "real")
INSTRUMENTS = ("oracle", "pca", "tube_fork", "tube_anticip", "rings")
APPLIES = {"shift": ("oracle", "pca", "rings"), "split": ("oracle", "pca", "rings"),
           "fork": ("tube_fork",), "precursor": ("tube_anticip", "oracle"),
           "precursor_persist": ("tube_anticip", "oracle")}


@dataclass(frozen=True)
class BelencConfig:
    doses: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    reps: int = 20
    row: int = 2                      # r*: fila plantada (lejos de la bifurcación natural del ancla)
    frac: float = 0.5                 # fracción de draws en S
    min_draws: int = 12               # filas comunes a >= este número de draws (crad_tube.align_turn)
    high_pcs: Tuple[int, int] = (0, 5)
    low_band: Tuple[float, float] = (0.5, 0.9)   # cuantiles del rango de PCs con varianza confiable
    fork_dose: float = 8.0            # bifurcación fija del analito precursor
    auc_ok: float = 0.9
    overlap_ok: float = 0.5
    purity_ok: float = 0.8
    lod_rate: float = 0.95
    loq_rel: float = 0.20
    tube_null: int = 49
    rings_null: int = 49
    seed: int = 20261022


# ================================================================ matriz =====
@dataclass
class Matrix:
    Y: np.ndarray                     # [D, R, d] phi real alineado (draw, fila)
    draws: np.ndarray
    layer: int
    turn: int
    source: str
    real_dir: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)

    @property
    def shape(self):
        return self.Y.shape

    def row_centered_cloud(self) -> np.ndarray:
        """Todas las (draw, fila) con el centro de cada fila restado: quita token y formato."""
        Z = self.Y - self.Y.mean(0, keepdims=True)
        return Z.reshape(-1, Z.shape[-1])


def load_matrix(cd: Path, layer: int, turn: int, min_draws: int, real_pair=None) -> Matrix:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from crad_tube import align_turn
    with np.load(cd / "capture.npz", allow_pickle=False) as zf:
        meta = json.loads(str(zf["meta"]))
        lo, hi = meta["superset"]
        if not (lo <= layer < hi):
            raise SystemExit(f"[belenc] capa {layer} fuera del superset [{lo},{hi})")
        kind, tu = zf["kind"], zf["turn"]
        m = (kind == 1) & (tu == turn)
        idx = np.flatnonzero(m)
        phi = zf["phi"][idx, layer - lo].astype(np.float64)
        draw, row = zf["draw"][idx], zf["token_in_turn"][idx]
    Y, keep, rows = align_turn(phi, draw, row, min_draws)
    mat = Matrix(Y=Y, draws=keep, layer=layer, turn=turn, source=cd.name)
    if real_pair:
        cj = cd / "contract" / "CONTRACT.json"
        if not cj.exists():
            raise SystemExit("[belenc] --real-pair necesita contract/CONTRACT.json (crad_contract.py check)")
        dec = {int(r["draw"]): r for r in json.loads(cj.read_text())["decisions"]}
        s, t = real_pair
        pos = {int(k): i for i, k in enumerate(keep)}
        rs, rt = dec[s]["row"], dec[t]["row"]
        if s not in pos or t not in pos or rs is None or rt is None or max(rs, rt) >= Y.shape[1]:
            raise SystemExit(f"[belenc] el par real {real_pair} no tiene filas de ancla dentro de la matriz")
        dvec = Y[pos[s], rs] - Y[pos[t], rt]
        mat.real_dir = dvec / np.linalg.norm(dvec)
        mat.notes.append(f"real: h[{s}, fila {rs}] - h[{t}, fila {rt}], |delta|/|h_tgt| = "
                         f"{np.linalg.norm(dvec) / np.linalg.norm(Y[pos[t], rt]):.3f}")
    return mat


# ============================================================ direcciones ====
def spectrum(mat: Matrix):
    Z = mat.row_centered_cloud()
    Z = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = S ** 2 / max(len(Z) - 1, 1)
    return Vt, var


def participation_ratio(var: np.ndarray) -> float:
    return float(var.sum() ** 2 / max(np.sum(var ** 2), 1e-300))


def direction(mat: Matrix, family: str, rng, cache: dict, cfg: BelencConfig) -> Optional[np.ndarray]:
    if "spec" not in cache:
        cache["spec"] = spectrum(mat)
    Vt, var = cache["spec"]
    d = mat.shape[2]
    if family == "high":
        a, b = cfg.high_pcs
        return Vt[rng.integers(a, min(b, len(Vt)))]
    if family == "low":
        good = np.flatnonzero(var > 1e-6 * var[0])                # varianza numéricamente confiable
        q0, q1 = cfg.low_band
        lo_i, hi_i = good[int(q0 * (len(good) - 1))], good[int(q1 * (len(good) - 1))]
        return Vt[rng.integers(lo_i, hi_i + 1)]
    if family == "random":
        u = rng.normal(size=d)
        return u / np.linalg.norm(u)
    if family == "real":
        return None if mat.real_dir is None else mat.real_dir.copy()
    raise ValueError(family)


def sigma_along(mat: Matrix, u: np.ndarray) -> float:
    """Desviación natural de la matriz a lo largo de u (nube centrada por fila)."""
    return float((mat.row_centered_cloud() @ u).std(ddof=1))


# ================================================================ plantado ===
def plant(Y0: np.ndarray, analyte: str, u: np.ndarray, sigma: float, dose: float, S: np.ndarray,
          row: int, w: Optional[np.ndarray] = None, fork_dose: float = 8.0, sigma_w: float = 1.0) -> np.ndarray:
    """Devuelve una COPIA de la matriz con el analito plantado. Nada se modifica en sitio."""
    Y = Y0.copy()
    a = dose * sigma
    if analyte == "shift":
        Y[S, row] += a * u
    elif analyte == "split":
        notS = np.setdiff1d(np.arange(len(Y)), S)
        Y[S, row] += 0.5 * a * u
        Y[notS, row] -= 0.5 * a * u
    elif analyte == "fork":
        Y[S, row + 1:] += a * u
    elif analyte in ("precursor", "precursor_persist"):
        if w is None:
            raise ValueError("precursor necesita el eje de la bifurcación w")
        Y[S, row + 1:] += fork_dose * sigma_w * w
        if analyte == "precursor":
            Y[S, row] += a * u
        else:
            Y[S, row:] += a * u
    else:
        raise ValueError(analyte)
    return Y


# =========================================================== instrumentos ===
def auc(y, s) -> float:
    y = np.asarray(y); s = np.asarray(s, float)
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    return float(((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)))


def inst_oracle(Y, S, row, u, sigma, **_) -> Dict:
    """Techo: conoce S. Eje diferencia de medias leave-one-out; AUC y dosis estimada."""
    X = Y[:, row]
    y = np.zeros(len(X), int); y[S] = 1
    sc = np.empty(len(X))
    for i in range(len(X)):
        k = np.arange(len(X)) != i
        dvec = X[k & (y == 1)].mean(0) - X[k & (y == 0)].mean(0)
        dvec /= max(np.linalg.norm(dvec), 1e-12)
        sc[i] = (X[i] - 0.5 * (X[k & (y == 1)].mean(0) + X[k & (y == 0)].mean(0))) @ dvec
    a_ = auc(y, sc)
    a_ = max(a_, 1 - a_)
    return {"ok": bool(a_ >= CFG.auc_ok), "auc": a_, "dose_est": _crossfit_dose(X, y)}


def _crossfit_dose(X: np.ndarray, y: np.ndarray) -> float:
    """Dosis estimada SIN conocer u: la dirección sale de una mitad de los draws y la magnitud
    (separación de medias / sd dentro de grupos, a lo largo de esa dirección) se mide en la otra.
    Con la dirección y la magnitud de la misma muestra el ruido de alta dimensión inflaría la
    estimación (sesgo de |Δmedia| ~ sqrt(d/n))."""
    idx1, idx0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if min(len(idx1), len(idx0)) < 4:
        return float("nan")
    ests = []
    for fold in (0, 1):
        a1, b1 = idx1[fold::2], idx1[1 - fold::2]
        a0, b0 = idx0[fold::2], idx0[1 - fold::2]
        dvec = X[a1].mean(0) - X[a0].mean(0)
        dvec /= max(np.linalg.norm(dvec), 1e-12)
        p1, p0 = X[b1] @ dvec, X[b0] @ dvec
        sd = np.sqrt(0.5 * (p1.var(ddof=1) + p0.var(ddof=1)))
        ests.append(abs(p1.mean() - p0.mean()) / max(sd, 1e-12))
    return float(np.mean(ests))


def inst_pca(Y, S, row, u, **_) -> Dict:
    X = Y[:, row] - Y[:, row].mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    ov = float((Vt[0] @ u) ** 2)
    return {"ok": bool(ov >= CFG.overlap_ok), "overlap2": ov}


def inst_tube_fork(Y, S, row, **_) -> Dict:
    from crad_tube import FORK_Z, tangling, tube_coords
    Q, Z, PE, ACC = tangling(tube_coords(Y), n_null=CFG.tube_null, seed=CFG.seed)
    flagged = np.flatnonzero(Z > FORK_Z).tolist()
    return {"ok": bool(row in flagged), "flagged": flagged, "z_at_row": float(Z[row]),
            "row_error": (int(np.argmax(Z)) - row)}


def inst_tube_anticip(Y, S, row, **_) -> Dict:
    from crad_tube import tangling, tube_coords
    Q, Z, PE, ACC = tangling(tube_coords(Y), n_null=CFG.tube_null, seed=CFG.seed)
    return {"ok": bool(PE[row] < 0.05), "p_enc": float(PE[row]),
            "knn_acc": None if not np.isfinite(ACC[row]) else float(ACC[row])}


def inst_rings(Y, S, row, **_) -> Dict:
    from crad_rings import RingsConfig, map_cloud
    D, R, d = Y.shape
    X = Y.reshape(D * R, d)
    res = map_cloud(X, RingsConfig(n_rings=7, n_null=CFG.rings_null, seed=CFG.seed))
    target = np.array([k * R + row for k in range(D)])
    side = np.zeros(D * R, int); side[[k * R + row for k in S]] = 1
    best = 0.0
    for nd in res["graph"]["nodes"]:
        t = nd.get("tomography")
        if not t or not t["flagged"] or len(t["subgroups"]) < 2:
            continue
        inside = np.intersect1d(nd["members"], target)
        if len(inside) < 6:
            continue
        lab = np.full(D * R, -1)
        for gi, sg in enumerate(t["subgroups"]):
            lab[sg["ids"]] = gi
        li, si = lab[inside], side[inside]
        ok = li >= 0
        if ok.sum() < 6:
            continue
        pur = 0.0
        for g in np.unique(li[ok]):
            m = li[ok] == g
            pur += m.sum() * max(si[ok][m].mean(), 1 - si[ok][m].mean())
        best = max(best, pur / ok.sum())
    return {"ok": bool(best >= CFG.purity_ok), "purity": best}


INST_FN: Dict[str, Callable] = {"oracle": inst_oracle, "pca": inst_pca, "tube_fork": inst_tube_fork,
                                "tube_anticip": inst_tube_anticip, "rings": inst_rings}
CFG = BelencConfig()


# ================================================================ teoría ====
def paul_overlap(theta: float, gamma: float) -> float:
    """Traslape² esperado del primer eje con el pico (Paul 2007; Benaych-Georges y Nadakuditi 2011)."""
    if theta <= np.sqrt(gamma):
        return 0.0
    return float((1 - gamma / theta ** 2) / (1 + gamma / theta))


def theta_for_overlap(target: float, gamma: float) -> float:
    lo, hi = np.sqrt(gamma), max(10.0, 10 * np.sqrt(gamma))
    while paul_overlap(hi, gamma) < target:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if paul_overlap(mid, gamma) < target else (lo, mid)
    return hi


def bbp_dose(analyte: str, sigma: float, nu: float, gamma: float, frac: float, target: float) -> Dict:
    """Dosis (en σ_u) donde la teoría predice traslape² = target y el umbral BBP, para shift/split."""
    th_t, th_b = theta_for_overlap(target, gamma), np.sqrt(gamma)
    var_per_a2 = frac * (1 - frac) if analyte == "shift" else 0.25   # varianza añadida / (a σ)²
    to_dose = lambda th: float(np.sqrt(th * nu / (var_per_a2 * sigma ** 2)))
    return {"dose_overlap": to_dose(th_t), "dose_bbp": to_dose(th_b), "theta_overlap": th_t, "theta_bbp": th_b}


# ============================================================== estadística ==
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def lod_from_rates(doses: Sequence[float], rates: Sequence[float], target: float) -> Optional[float]:
    """Dosis mínima > 0 con tasa >= target en ella y en todas las mayores (monotonía exigida)."""
    ds = [(d, r) for d, r in zip(doses, rates) if d > 0]
    for i, (d, _) in enumerate(ds):
        if all(r >= target for _, r in ds[i:]):
            return d
    return None


def loq_from_estimates(doses, est_by_dose: Dict[float, List[float]], rel: float) -> Optional[float]:
    ds = sorted(d for d in doses if d > 0)
    ok = [np.median(np.abs(np.array(est_by_dose[d]) - d) / d) <= rel for d in ds]
    for i, d in enumerate(ds):
        if all(ok[i:]):
            return d
    return None


def linearity(doses, est_by_dose, above: Optional[float]) -> Optional[Dict]:
    if above is None:
        return None
    x = np.array([d for d in sorted(doses) if d >= above for _ in est_by_dose[d]])
    y = np.array([e for d in sorted(doses) if d >= above for e in est_by_dose[d]])
    if len(np.unique(x)) < 2:
        return None
    A = np.vstack([x, np.ones_like(x)]).T
    (m, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - np.sum((y - (m * x + b)) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-300)
    return {"slope": float(m), "intercept": float(b), "r2": float(r2)}


# ================================================================ corrida ===
def run_standard(mat: Matrix, cfg: BelencConfig, instruments: Sequence[str],
                 analytes: Sequence[str] = ANALYTES, families: Sequence[str] = FAMILIES,
                 log: Callable = print) -> Dict:
    global CFG
    CFG = cfg
    D, R, d = mat.shape
    if not (0 <= cfg.row < R - 1):
        raise SystemExit(f"[belenc] fila plantada {cfg.row} fuera de [0, {R - 1}) en la matriz")
    # resolución de los p-valores de permutación: con n barajas el p mínimo es 1/(n+1). Si no puede quedar
    # debajo de α, el instrumento NUNCA puede detectar y la ficha mediría la configuración, no el instrumento
    # (selftest del 22-oct: con 19 barajas el tubo daba LOD = None con acierto kNN de 1.00).
    for name, n_null in (("tube", cfg.tube_null), ("rings", cfg.rings_null)):
        if 1.0 / (n_null + 1) >= 0.05:
            raise SystemExit(f"[belenc] {name}_null={n_null}: p mínimo {1 / (n_null + 1):.3f} >= 0.05; "
                             "el instrumento no podría detectar nunca. Usa >= 39 barajas.")
    rng0 = np.random.default_rng(cfg.seed)
    cache: dict = {}
    Vt, var = spectrum(mat)
    cache["spec"] = (Vt, var)
    n_slice = D
    d_eff = participation_ratio(var)
    gamma, nu = d_eff / n_slice, float(var.sum() / d_eff)
    out = {"matrix": {"source": mat.source, "layer": mat.layer, "turn": mat.turn, "draws": mat.draws.tolist(),
                      "shape": [D, R, d], "row": cfg.row, "d_eff": d_eff, "gamma_slice": gamma, "nu": nu,
                      "notes": mat.notes},
           "config": asdict(cfg), "cells": []}
    for analyte in analytes:
        insts = [i for i in instruments if i in APPLIES[analyte]]
        if not insts:
            continue
        for fam in families:
            if fam == "real" and mat.real_dir is None:
                continue
            res = {i: {dz: [] for dz in cfg.doses} for i in insts}
            sig_list, theo = [], []
            for rep in range(cfg.reps):
                rng = np.random.default_rng([cfg.seed, ANALYTES.index(analyte), FAMILIES.index(fam), rep])
                u = direction(mat, fam, rng, cache, cfg)
                sigma = sigma_along(mat, u)
                sig_list.append(sigma)
                S = np.sort(rng.choice(D, size=max(2, int(round(cfg.frac * D))), replace=False))
                w = None
                sigma_w = 1.0
                if analyte in ("precursor", "precursor_persist"):
                    w = rng.normal(size=d); w -= (w @ u) * u; w /= np.linalg.norm(w)
                    sigma_w = sigma_along(mat, w)
                for dz in cfg.doses:
                    Y = plant(mat.Y, analyte, u, sigma, dz, S, cfg.row, w, cfg.fork_dose, sigma_w)
                    for i in insts:
                        r = INST_FN[i](Y, S, cfg.row, u=u, sigma=sigma)
                        res[i][dz].append(r)
                if analyte in ("shift", "split"):
                    theo.append(bbp_dose(analyte, sigma, nu, gamma, cfg.frac, cfg.overlap_ok))
            for i in insts:
                rates = [float(np.mean([r["ok"] for r in res[i][dz]])) for dz in cfg.doses]
                ks = [int(np.sum([r["ok"] for r in res[i][dz]])) for dz in cfg.doses]
                lod = lod_from_rates(cfg.doses, rates, cfg.lod_rate)
                cell = {"analyte": analyte, "family": fam, "instrument": i,
                        "rates": dict(zip(map(str, cfg.doses), rates)),
                        "ci": {str(dz): wilson(k, cfg.reps) for dz, k in zip(cfg.doses, ks)},
                        "fpr": rates[cfg.doses.index(0.0)] if 0.0 in cfg.doses else None,
                        "fpr_ci": wilson(ks[cfg.doses.index(0.0)], cfg.reps) if 0.0 in cfg.doses else None,
                        "lod95": lod, "sigma_median": float(np.median(sig_list))}
                if i == "oracle" and analyte in ("shift",):
                    est = {dz: [r["dose_est"] for r in res[i][dz]] for dz in cfg.doses}
                    loq = loq_from_estimates(cfg.doses, est, cfg.loq_rel)
                    cell["loq"] = None if (lod is None or loq is None) else max(loq, lod)   # ICH: LOQ >= LOD
                    cell["linearity"] = linearity(cfg.doses, est, lod)
                    cell["dose_est_median"] = {str(dz): float(np.nanmedian(est[dz])) for dz in cfg.doses}
                if i == "pca":
                    cell["overlap2_mean"] = {str(dz): float(np.mean([r["overlap2"] for r in res[i][dz]]))
                                             for dz in cfg.doses}
                    if theo:
                        cell["theory"] = {k: float(np.median([t[k] for t in theo])) for k in theo[0]}
                if i == "tube_fork":
                    errs = [r["row_error"] for dz in cfg.doses if dz > 0 for r in res[i][dz] if r["ok"]]
                    cell["row_error_abs_mean"] = float(np.mean(np.abs(errs))) if errs else None
                out["cells"].append(cell)
                log(f"[belenc] {analyte:9s} {fam:6s} {i:12s} FPR {cell['fpr']:.2f} LOD95 {lod} "
                    + " ".join(f"{dz:g}:{r:.2f}" for dz, r in zip(cfg.doses, rates)))
    return out


# ================================================================ reporte ===
def report(res: Dict, fisher: Optional[Dict] = None) -> str:
    m, cfg = res["matrix"], res["config"]
    doses = cfg["doses"]
    fmt = lambda x: "—" if x is None else f"{x:g}"
    L = [f"# BELENC v{VERSION} — ficha de validación", "",
         f"**Matriz:** `{m['source']}` · capa {m['layer']} · turno {m['turn']} · fila plantada {m['row']} · "
         f"{m['shape'][0]} draws × {m['shape'][1]} filas × d={m['shape'][2]}",
         f"**Geometría:** d_eff (participation ratio) {m['d_eff']:.1f} · γ = d_eff/n_draws {m['gamma_slice']:.2f} · "
         f"umbral BBP √γ = {np.sqrt(m['gamma_slice']):.2f}", ""]
    L += [f"- {n}" for n in m.get("notes", [])]
    L += ["", f"Réplicas por dosis: {cfg['reps']} · LOD95 = dosis mínima con éxito ≥ {cfg['lod_rate']:.0%} "
          "en ella y en todas las mayores · dosis en σ naturales a lo largo de la dirección plantada", "",
          "**Una ficha vale para esta matriz. Un negativo del instrumento solo es informativo para efectos "
          "por encima de su LOD; un positivo solo si supera su tasa en blancos.**", ""]
    has_f = fisher is not None
    head = "| analito | familia | instrumento | FPR (IC 95%) | " + " | ".join(f"a={fmt(d)}" for d in doses if d > 0) \
           + " | LOD95 (σ) |" + (" LOD95 (nats) |" if has_f else "") + " extra |"
    L += [head, "|" + "---|" * (5 + len([d for d in doses if d > 0]) + (1 if has_f else 0))]
    for c in res["cells"]:
        lo, hi = c["fpr_ci"]
        extra = []
        if c.get("loq") is not None:
            extra.append(f"LOQ {fmt(c['loq'])}")
        if c.get("linearity"):
            extra.append(f"pend {c['linearity']['slope']:.2f} R² {c['linearity']['r2']:.2f}")
        if c.get("theory"):
            extra.append(f"teoría: traslape 0.5 en a≈{c['theory']['dose_overlap']:.2f}, BBP a≈{c['theory']['dose_bbp']:.2f}")
        if c.get("row_error_abs_mean") is not None:
            extra.append(f"|err fila| {c['row_error_abs_mean']:.2f}")
        nats = ""
        if has_f:
            k = (fisher.get("kl_per_sigma2", {}) or {}).get(c["family"])
            nats = f" {k * c['lod95'] ** 2:.3g} |" if (k is not None and c["lod95"] is not None) else " — |"
        L.append(f"| {c['analyte']} | {c['family']} | {c['instrument']} | {c['fpr']:.2f} ({lo:.2f}–{hi:.2f}) | "
                 + " | ".join(f"{c['rates'][str(d)]:.2f}" for d in doses if d > 0)
                 + f" | **{fmt(c['lod95'])}** |{nats} {'; '.join(extra) or '—'} |")
    L += ["", "Lectura de la columna teoría (solo pca): la dosis donde la teoría de matrices aleatorias predice "
          "traslape² = 0.5 bajo un bulto isotrópico de dimensión d_eff. LOD empírico ≈ teoría: la matriz se "
          "comporta como ruido isotrópico en esa familia. LOD empírico ≫ teoría: efecto matriz (estructura real "
          "que compite con la dirección). LOD empírico ≪ teoría: la familia vive en un subespacio más limpio "
          "que el promedio."]
    if has_f:
        L += ["", f"Conversión a nats con `{fisher.get('source', 'fisher.json')}`: KL ≈ ½·a²·σ_u²·uᵀGu, mediana por "
              "familia de dirección (respuesta lineal, belenc_fisher.py)."]
    return "\n".join(L)


# =============================================================== selftest ===
def _synthetic_matrix(rng, D=24, R=8, d=256, tok_sd=6.0, hist_sd=1.5, aniso=True) -> Matrix:
    """Imita la estructura real: token por fila, historia por draw, ruido anisotrópico."""
    T = np.cumsum(rng.normal(size=(R, d)) * tok_sd, 0)
    scale = np.linspace(2.5, 0.3, d) if aniso else np.ones(d)
    Y = np.zeros((D, R, d))
    for k in range(D):
        H = rng.normal(size=d) * hist_sd * scale
        for t in range(R):
            Y[k, t] = T[t] + H + rng.normal(size=d) * scale
    return Matrix(Y=Y, draws=np.arange(D), layer=0, turn=7, source="sintética (selftest)")


def selftest() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    tests = []
    rng = np.random.default_rng(1)
    mat = _synthetic_matrix(rng)
    cfg = BelencConfig(reps=12, doses=(0.0, 1.0, 4.0, 16.0), tube_null=49, rings_null=49)
    global CFG
    CFG = cfg

    # 1. el plantado es exacto y no toca la matriz original
    u = direction(mat, "random", np.random.default_rng(2), {}, cfg)
    sig = sigma_along(mat, u)
    S = np.arange(0, 24, 2)
    Y0 = mat.Y.copy()
    Y = plant(mat.Y, "shift", u, sig, 3.0, S, 2)
    dproj = (Y[S, 2] - Y0[S, 2]) @ u / sig
    tests.append((f"plantado shift exacto: desplazamiento {dproj.mean():.6f} σ (pedido 3), matriz intacta",
                  np.allclose(dproj, 3.0) and np.array_equal(mat.Y, Y0)
                  and np.allclose(Y[np.setdiff1d(np.arange(24), S)], Y0[np.setdiff1d(np.arange(24), S)])))
    Ys = plant(mat.Y, "split", u, sig, 2.0, S, 2)
    sp = (Ys[:, 2] - Y0[:, 2]) @ u / sig
    tests.append(("plantado split: +1σ en S, −1σ fuera de S",
                  np.allclose(sp[S], 1.0) and np.allclose(np.delete(sp, S), -1.0)))
    Yf = plant(mat.Y, "fork", u, sig, 2.0, S, 2)
    tests.append(("plantado fork: solo filas > r*", np.allclose(Yf[:, :3], Y0[:, :3])
                  and np.allclose((Yf[S, 3:] - Y0[S, 3:]) @ u / sig, 2.0)))
    w0 = np.zeros_like(u); w0[0] = 1.0
    Yp = plant(mat.Y, "precursor_persist", u, sig, 2.0, S, 2, w0, 8.0, 1.0)
    Yq = plant(mat.Y, "precursor", u, sig, 2.0, S, 2, w0, 8.0, 1.0)
    tests.append(("precursor persistente sigue en r*+1, el transitorio no",
                  np.allclose((Yp[S, 3] - Y0[S, 3] - 8.0 * w0) @ u / sig, 2.0)
                  and np.allclose((Yq[S, 3] - Y0[S, 3] - 8.0 * w0) @ u / sig, 0.0, atol=1e-9)))

    # 2. familias: high tiene más varianza natural que low, y low sigue teniendo varianza > 0
    c = {}
    sh = np.median([sigma_along(mat, direction(mat, "high", np.random.default_rng(i), c, cfg)) for i in range(5)])
    sl = np.median([sigma_along(mat, direction(mat, "low", np.random.default_rng(i), c, cfg)) for i in range(5)])
    tests.append((f"familias: σ high {sh:.2f} > σ low {sl:.2f} > 0", sh > sl > 0))

    # 3. teoría: traslape de Paul contra simulación isotrópica con pico
    sims, preds = [], []
    for g_n, th in ((400, 2.0), (400, 4.0), (200, 3.0)):
        n, dd = g_n, 800
        gam = dd / n
        uu = np.zeros(dd); uu[0] = 1
        ovs = []
        for s in range(20):
            r = np.random.default_rng(100 + s)
            X = r.normal(size=(n, dd)); X[:, 0] *= np.sqrt(1 + th)
            _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
            ovs.append((Vt[0] @ uu) ** 2)
        sims.append(float(np.mean(ovs))); preds.append(paul_overlap(th, gam))
    tests.append((f"teoría de Paul contra simulación: pred {np.round(preds, 2).tolist()} sim {np.round(sims, 2).tolist()}",
                  all(abs(a - b) < 0.08 for a, b in zip(preds, sims))))

    # 4. lógica de LOD, LOQ y Wilson
    tests.append(("LOD exige monotonía", lod_from_rates([0, 1, 2, 4, 8], [0, .96, .5, .97, 1], .95) == 4
                  and lod_from_rates([0, 1, 2], [0, .5, .9], .95) is None))
    lo_, hi_ = wilson(19, 20)
    tests.append((f"Wilson 19/20 = ({lo_:.3f}, {hi_:.3f})", abs(lo_ - 0.764) < 0.01 and abs(hi_ - 0.991) < 0.01))

    # 5. corrida completa del estándar: blancos bajos y LOD finito para el oráculo en shift
    res = run_standard(mat, cfg, ["oracle", "pca", "tube_fork", "tube_anticip"],
                       analytes=("shift", "fork", "precursor", "precursor_persist"), families=("high", "random"),
                       log=lambda *_: None)
    cells = {(c["analyte"], c["family"], c["instrument"]): c for c in res["cells"]}
    fpr_max = max(c["fpr"] for c in res["cells"])
    o = cells[("shift", "random", "oracle")]
    tf = cells[("fork", "random", "tube_fork")]
    tp = cells[("precursor_persist", "random", "tube_anticip")]
    tq = cells[("precursor", "random", "tube_anticip")]
    tests.append((f"tubo: anticipación persistente LOD95 {tp['lod95']} (a=16: {tp['rates']['16.0']:.2f}), "
                  f"transitoria LOD95 {tq['lod95']} (a=16: {tq['rates']['16.0']:.2f})",
                  tp["rates"]["16.0"] >= 0.9))
    tests.append((f"estándar: FPR máx en blancos {fpr_max:.2f} <= 0.17, oráculo shift LOD95 {o['lod95']}, "
                  f"tubo fork LOD95 {tf['lod95']}", fpr_max <= 0.17 and o["lod95"] is not None and tf["lod95"] is not None))
    lin = o.get("linearity")
    tests.append((f"oráculo: LOQ {o.get('loq')} >= LOD {o['lod95']}, dosis estimada "
                  f"{ {k: round(v, 1) for k, v in o['dose_est_median'].items()} }, linealidad "
                  f"pend {lin['slope'] if lin else None}",
                  o.get("loq") is None or o["loq"] >= o["lod95"]))
    est0 = o["dose_est_median"]["0.0"]
    tests.append((f"dosis estimada sin conocer u: crece con la dosis plantada (a=0: {est0:.2f}, a=16: "
                  f"{o['dose_est_median']['16.0']:.2f})", o["dose_est_median"]["16.0"] > 4 * max(est0, 0.1)))

    # 5b. aros: el adaptador corre sobre la matriz plantada y devuelve una pureza en [0, 1]
    Yr = plant(mat.Y, "split", u, sig, 8.0, S, 2)
    rr = inst_rings(Yr, S, 2)
    tests.append((f"aros: adaptador corre (pureza {rr['purity']:.2f}, éxito {rr['ok']})", 0.0 <= rr["purity"] <= 1.0))

    # 6. determinismo: misma semilla, mismo resultado
    res2 = run_standard(mat, cfg, ["oracle", "pca"], analytes=("shift",), families=("random",), log=lambda *_: None)
    res3 = run_standard(mat, cfg, ["oracle", "pca"], analytes=("shift",), families=("random",), log=lambda *_: None)
    tests.append(("determinista con la misma semilla", json.dumps(res2["cells"]) == json.dumps(res3["cells"])))

    # 6b. la compuerta de resolución detiene una configuración que no puede detectar
    try:
        run_standard(mat, replace(cfg, tube_null=19), ["oracle"], analytes=("shift",), families=("random",),
                     log=lambda *_: None)
        blocked = False
    except SystemExit:
        blocked = True
    tests.append(("compuerta: 19 barajas (p mínimo 0.05) se rechaza antes de correr", blocked))

    # 7. el reporte se arma y declara la matriz
    md = report(res)
    tests.append(("reporte con matriz, teoría y LOD", "ficha de validación" in md and "LOD95" in md and "teoría" in md))

    print(f"BELENC v{VERSION} selftest")
    for name, ok in tests:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in tests) else 1


# ==================================================================== CLI ===
def cmd_run(a) -> int:
    cd = Path(a.capture_dir)
    cfg = BelencConfig(reps=a.reps, row=a.row, min_draws=a.min_draws, seed=a.seed,
                       tube_null=a.tube_null, rings_null=a.rings_null)
    mat = load_matrix(cd, a.layer, a.turn, a.min_draws, a.real_pair)
    print(f"[belenc] matriz {mat.shape} · capa {a.layer} · turno {a.turn} · draws {mat.draws.tolist()}", flush=True)
    t0 = time.time()
    res = run_standard(mat, cfg, a.instruments, analytes=a.analytes, families=a.families)
    res["wall_seconds"] = time.time() - t0
    res["belenc_version"] = VERSION
    fisher = json.loads(Path(a.fisher_json).read_text()) if a.fisher_json else None
    if fisher is not None:
        fisher = fisher.get("layers", {}).get(str(a.layer), fisher)
        fisher["source"] = a.fisher_json
    out = cd / "belenc" / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_L{a.layer}_T{a.turn}_r{a.row}"
    out.mkdir(parents=True)
    (out / "belenc.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    md = report(res, fisher)
    (out / "BELENC.md").write_text(md + "\n")
    print(md)
    print(f"\n[belenc] escribí {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="BELENC: Base Estándar para Localizaciones Espaciales No Conocidas")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("run")
    p.add_argument("capture_dir")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--turn", type=int, default=7)
    p.add_argument("--row", type=int, default=2)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--min-draws", type=int, default=12)
    p.add_argument("--instruments", nargs="+", choices=INSTRUMENTS, default=["oracle", "pca", "tube_fork", "tube_anticip"])
    p.add_argument("--analytes", nargs="+", choices=ANALYTES, default=list(ANALYTES))
    p.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    p.add_argument("--real-pair", nargs=2, type=int, default=None)
    p.add_argument("--fisher-json", default=None)
    p.add_argument("--tube-null", type=int, default=49)
    p.add_argument("--rings-null", type=int, default=49)
    p.add_argument("--seed", type=int, default=20261022)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        return cmd_run(a)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
