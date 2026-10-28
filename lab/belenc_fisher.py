#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BELENC-Fisher v0.1.2 — respuesta lineal y geometría de información en la fila de decisión
TwoQuarks Research — Luis Jaime Ledesma Perez

Qué agrega a BELENC
-------------------
BELENC mide en σ geométricos: cuánto se aparta un estado de la variabilidad natural. Eso no dice
cuánto cambia la CONDUCTA. Este módulo da la unidad conductual y la liga con la geométrica.

Física de base (respuesta lineal + geometría de información)
  Para una perturbación pequeña δ del residual en la capa l, fila de decisión:
      KL( p(·|h) || p(·|h+δ) )  ≈  ½ δᵀ G δ,        G = Jᵀ F J
  F es la métrica de Fisher de la distribución de salida (la única métrica invariante de las
  distribuciones, Čencov) y J el jacobiano de los logits respecto al residual. G es esa métrica
  llevada (pullback) al espacio de activaciones: mide distancia en NATS de conducta, no en unidades
  arbitrarias del residual. Se calcula exacto en el soporte top-K:
      G_K = Σ_{y ∈ top-K} p_y g_y g_yᵀ,   g_y = ∇_h log p_y      (rango <= K, guardado como A = [√p_y g_y])
  y se reporta la masa cubierta Σ_{top-K} p_y.

Cantidades que reporta (por capa, mediana sobre draws)
  tr G               sensibilidad total del output al residual (nats por unidad² de norma)
  τ                  temperatura conductual = ½ tr(G Σ): KL esperada que produce UNA fluctuación natural
                     de la matriz (Σ = covarianza entre draws en la fila de decisión). Es el estándar de
                     variabilidad: dos modelos, checkpoints o capas se comparan en la misma unidad.
  modos              autovalores de Σ^½ G Σ^½: qué fracción de τ lleva cada modo natural
  cobertura c        tr(P_D G) / tr G: fracción de la sensibilidad dentro del subespacio que la propia
                     variabilidad natural explora (P_D = proyector al span de los draws centrados)
  enriquecimiento E  c / (dim span / d): cuántas veces más sensibilidad captura ese span que un
                     subespacio aleatorio de la misma dimensión. E ≈ 1: lo que varía y lo que importa
                     no tienen relación. E ≫ 1: el modelo varía donde la conducta es sensible.
  fragilidad latente φ = 1 - c. Sensibilidad en direcciones que la muestra natural NUNCA visita: ahí un
                     empujón chico (deriva, fine-tuning, inyección) cambia la conducta sin que ninguna
                     medida de variabilidad lo anticipe. Se lee junto con E, porque en alta dimensión φ
                     es casi 1 por geometría (un subespacio de n-1 dimensiones en d=3584).
  χ por familia      KL por σ² natural a lo largo de u: ½ σ_u² uᵀGu, para las familias de BELENC (high,
                     low, random, real) y para los ejes de CRAD (uL, uR). Es la conversión σ -> nats de
                     la ficha de BELENC, y es una prueba física directa de "L/R describe, no ejecuta":
                     χ(L/R) ≈ χ(random) significa que los ejes de CRAD no son direcciones de respuesta.
  linealidad (--pair) parche h_tgt + α(h_src - h_tgt) con α en una rejilla: KL medida contra la predicha
                     ½α²δᵀGδ. El rango lineal (|medida/predicha - 1| <= 20%) es el rango de validez de
                     toda la conversión. Fuera de él, el efecto es no lineal y la unidad no aplica.

Compuertas (abortan)
  - la corrida sin parche reproduce el turno capturado
  - el forward completo (prompt + prefijo) cae en la fila del contrato: su estado está a <= FWD_TOL
    del phi capturado Y a menos de CLOCK_SEP veces la distancia a las filas vecinas (prueba directa de
    alineación, no de magnitud), y sus logits caen en la misma fila (readout_check de patch_how)
  - α = 0 deja los logits intactos (el hook escribe donde se lee)

Corrección 23-oct (v0.1.1)
  La v0.1 comparaba el forward completo contra phi con LIVE_TOL = 1e-3 de patch_how. Esa tolerancia es
  para un estado grabado DURANTE la generación (mismo camino numérico). Aquí el camino es otro: un
  forward de todo el prefijo sin caché contra un decode token a token con caché. En fp16 y nf4 los
  kernels cambian (gemv de 1 token contra dequant + gemm del prefill) y la diferencia es de redondeo.
  Medido en la A100, draw 0, L14: rel 1.15e-3, y la compuerta abortó con un falso "reloj desalineado".
  Una fila corrida da rel >= 0.55 (captura determinista, L14/17/21, turnos 6 y 7): 500 veces más.
  Por la misma causa, la linealidad parte ahora del estado del propio forward (no del phi capturado) y
  δ = phi_src - phi_tgt se suma a ese estado. G se calcula en ese mismo punto, así que todo es coherente.

Corrección 23-oct (v0.1.2)
  Los logits usaban READ_TOL = 0.05 nats. En la G4 el máximo medido fue 0.039, un margen de 1.3x: la
  misma clase de error que el estado. Ahora usa readout_check (mismo argmax, techo absoluto y
  separación contra las filas vecinas), compartido con patch_how attend.

Predicciones, escritas antes de correr en GPU
  F1. χ(uL), χ(uR) ≈ χ(random): el cociente cae en [0.5, 2] en todas las capas.
  F2. χ(real) ≫ χ(random): la diferencia entre un TRUTH y un SELF_FICTION es una dirección de respuesta.
  F3. E > 3 en la fila de ancla: la variabilidad entre draws apunta a donde la conducta es sensible.
  F4. El parche 5 -> 0 sale del rango lineal antes de α = 1 (voltea la respuesta: no es una perturbación pequeña).

CLI
  python belenc_fisher.py --selftest
  python belenc_fisher.py captures/<id> --layers 14 17 21 [--K 64] [--pair 5 0] [--alphas 0.05 0.1 0.25 0.5 1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

VERSION = "0.1.2-fisher"
KL_FLOOR = 1e-5      # nats. Debajo, la KL medida es redondeo (fp16: log p con ~1e-3 de resolución relativa).
                     # Medido en el selftest: en float32 la KL ya da valores NEGATIVOS por debajo de ~1e-7.
SYM_TOL = 0.2        # |KL(+α) - KL(-α)| / (KL(+α) + KL(-α)): a segundo orden deben ser iguales
FWD_TOL = 1e-2       # rel: forward completo sin caché contra phi del decode con caché (fp16 + nf4, otro camino).
                     # Medido 1.15e-3 (A100, L14). Una fila corrida da >= 0.55.
CLOCK_SEP = 0.05     # el estado debe estar 20 veces más cerca de su fila que de las filas vecinas


def clock_check(h_live, h_row, neighbors):
    """¿El forward cae en la fila del contrato? (ok, rel a la fila, rel mínima a las filas vecinas)."""
    def rel(x):
        return float(np.linalg.norm(h_live - x) / max(np.linalg.norm(x), 1e-12))
    r = rel(h_row)
    nb = min((rel(x) for x in neighbors), default=float("inf"))
    return (r <= FWD_TOL and r <= CLOCK_SEP * nb), r, nb


# ========================================================== núcleo físico ==
def forward_ids(mw, prompt_ids, prefix_ids):
    return mw.torch.tensor([list(prompt_ids) + list(prefix_ids)], device=mw.device)


def fisher_at(mw, ids, layer: int, K: int, replace=None):
    """Un forward con el residual de la ÚLTIMA posición, salida del bloque `layer`, como hoja con
    gradiente. Devuelve A [K, d] (filas √p_y ∇log p_y), p top-K, masa, logprobs completos y el estado.
    replace: si no es None, ese vector sustituye al estado antes de derivar (para la linealidad)."""
    torch = mw.torch
    box = {}

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        x = (h[0, -1].detach().clone() if replace is None else replace.to(h.dtype).to(h.device)).requires_grad_(True)
        box["x"], box["h0"] = x, h[0, -1].detach().float().cpu().numpy().copy()
        h2 = h.clone()
        h2[0, -1] = x
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2

    hd = mw.lm.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            out = mw.lm(input_ids=ids, use_cache=False)
            lp = out.logits[0, -1].float().log_softmax(-1)
            p = lp.exp()
            top = torch.topk(p, k=min(K, p.numel()))
            rows = []
            for j, y in enumerate(top.indices.tolist()):
                g, = torch.autograd.grad(lp[y], box["x"], retain_graph=(j < len(top.indices) - 1))
                rows.append((top.values[j].detach().sqrt() * g.float()).cpu().numpy())
    finally:
        hd.remove()
    return {"A": np.stack(rows), "p_top": top.values.detach().cpu().numpy(), "idx": top.indices.cpu().numpy(),
            "mass": float(top.values.detach().sum()), "lp": lp.detach().cpu().numpy(), "h": box["h0"]}


def logprobs_with(mw, ids, layer: int, vec=None):
    """Forward sin gradiente; si vec no es None reemplaza el residual de la última posición en `layer`."""
    torch = mw.torch

    def hook(_mod, _inp, out):
        if vec is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h[0, -1] = torch.as_tensor(vec, device=h.device).to(h.dtype)
        return out

    hd = mw.lm.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = mw.lm(input_ids=ids, use_cache=False)
    finally:
        hd.remove()
    return out.logits[0, -1].float().log_softmax(-1).cpu().numpy()


def kl(lp_p: np.ndarray, lp_q: np.ndarray) -> float:
    p = np.exp(lp_p)
    return float(np.sum(p * (lp_p - lp_q)))


def quad(A: np.ndarray, v: np.ndarray) -> float:
    """vᵀ G v con G = AᵀA."""
    return float(np.sum((A @ v) ** 2))


def temperature(A: np.ndarray, Dc: np.ndarray) -> float:
    """τ = ½ tr(G Σ), Σ = DcᵀDc/(n-1)."""
    return 0.5 * float(np.sum((A @ Dc.T) ** 2)) / max(len(Dc) - 1, 1)


def modes(A: np.ndarray, Dc: np.ndarray) -> np.ndarray:
    """Autovalores de Σ^½ G Σ^½ (en el span de los draws), ordenados, normalizados a suma 1."""
    M = (Dc @ A.T) @ (A @ Dc.T) / max(len(Dc) - 1, 1)
    w = np.clip(np.linalg.eigvalsh(M)[::-1], 0, None)
    return w / max(w.sum(), 1e-300)


def coverage(A: np.ndarray, Dc: np.ndarray) -> dict:
    """Fracción de tr G dentro del span de los draws centrados, y enriquecimiento contra azar."""
    Q, s, _ = np.linalg.svd(Dc.T, full_matrices=False)
    Q = Q[:, s > 1e-8 * max(s[0], 1e-300)]
    tr = float(np.sum(A ** 2))
    c = float(np.sum((A @ Q) ** 2)) / max(tr, 1e-300)
    k, d = Q.shape[1], Q.shape[0]
    return {"coverage": c, "enrichment": c / (k / d), "dim": k, "latent_fragility": 1.0 - c}


# ============================================================== selftest ===
def selftest() -> int:
    import random
    import torch
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here)); sys.path.insert(0, str(here / "src"))
    import crad_base
    from patch_how import Tap, _tiny_model, install_scoring, rebuild_context, run_turn
    tests = []
    mw = _tiny_model(seed=3)
    stash = {}
    install_scoring(mw, stash)
    tap = Tap(mw)
    ramp = crad_base.build_real_ramp("BLUEFALCON", n_turns=8)

    def seed_fn(s):
        random.seed(s); np.random.seed(s % (2 ** 32)); torch.manual_seed(s)

    T, focus, n_tok, temp, lo, hi, row, layer = 7, 2, 10, 0.7, 1, 3, 3, 1
    ctxs = [rebuild_context(mw, ramp, seed_fn, 42, k, T, focus, n_tok, temp, lo, hi) for k in range(6)]
    toks, scores, rec, tr = run_turn(mw, tap, stash, ctxs[0], T, n_tok, temp, lo, hi, record_row=row)
    tap.remove()
    gen_ids = [mw.tok.stoi.get(f["token"], 0) for f in tr["frames"]]
    ids = forward_ids(mw, mw._ids(ctxs[0]["hist"])["input_ids"][0].tolist(), gen_ids[:row])
    rng0 = np.random.default_rng(11)

    # 1. reloj: el forward completo da en la última posición el estado de la fila y los logits de la generación
    F = fisher_at(mw, ids, layer, K=mw.tok.vocab)
    tests.append((f"reloj alineado: estado == phi (|d| {np.abs(F['h'] - tr['frames'][row]['phi'][layer - lo]).max():.1e}), "
                  f"logits == generación (|d| {np.abs(F['lp'] - scores[row]).max():.1e})",
                  np.allclose(F["h"], tr["frames"][row]["phi"][layer - lo], atol=1e-5)
                  and np.abs(F["lp"] - scores[row]).max() < 1e-4))

    # 1b. compuerta del reloj: acepta ruido de redondeo, rechaza una fila corrida
    ph = [f["phi"][layer - lo] for f in tr["frames"]]
    nbr = [ph[row - 1], ph[row + 1]]
    noisy = F["h"] + 1.5e-3 * np.linalg.norm(F["h"]) * rng0.standard_normal(F["h"].shape) / np.sqrt(F["h"].size)
    ok_n, r_n, nb_n = clock_check(noisy, ph[row], nbr)
    ok_s, r_s, _ = clock_check(F["h"], ph[row - 1], [ph[row - 2], ph[row]])
    tests.append((f"reloj: ruido de redondeo 1.5e-3 pasa (rel {r_n:.1e}, vecinas {nb_n:.2f}), "
                  f"fila corrida no pasa (rel {r_s:.2f})", ok_n and not ok_s))

    # 2. G contra diferencias finitas de KL (K = vocabulario completo: exacto)
    rng = np.random.default_rng(0)
    # ε = 3% de |h|: régimen cuadrático y KL muy por encima del redondeo. Con ε = 1% la KL cae a ~1e-7 y en
    # float32 aparecen KL negativas (diagnóstico del 22-oct): eso es el piso numérico, no un error de G.
    ratios = []
    for _ in range(5):
        v = rng.normal(size=F["h"].shape); v /= np.linalg.norm(v)
        eps = 3e-2 * np.linalg.norm(F["h"])
        kp = kl(F["lp"], logprobs_with(mw, ids, layer, F["h"] + eps * v))
        km = kl(F["lp"], logprobs_with(mw, ids, layer, F["h"] - eps * v))
        ratios.append(0.5 * (kp + km) / (0.5 * eps ** 2 * quad(F["A"], v)))
    tests.append((f"KL ≈ ½εᵀGε con diferencias finitas simétricas: cocientes {np.round(ratios, 3).tolist()}",
                  all(abs(r - 1) < 0.05 for r in ratios)))
    eps = 1e-4 * np.linalg.norm(F["h"])
    tiny = kl(F["lp"], logprobs_with(mw, ids, layer, F["h"] + eps * v))
    tests.append((f"piso numérico: con ε=1e-4|h| la KL medida ({tiny:.1e}) queda bajo KL_FLOOR={KL_FLOOR:g} y se marca",
                  tiny < KL_FLOOR))

    # 3. truncar a top-K subestima G (orden de Loewner) y reporta la masa cubierta
    Fk = fisher_at(mw, ids, layer, K=8)
    vs = [rng.normal(size=F["h"].shape) for _ in range(10)]
    tests.append((f"top-8 cubre {Fk['mass']:.3f} de la masa y vᵀG_8v <= vᵀGv",
                  all(quad(Fk["A"], v) <= quad(F["A"], v) * (1 + 1e-6) for v in vs) and 0 < Fk["mass"] <= 1))

    # 4. α = 0 no mueve los logits (el hook escribe donde se lee)
    l0 = logprobs_with(mw, ids, layer, F["h"])
    tests.append((f"α=0 deja los logits intactos (|d| {np.abs(l0 - F['lp']).max():.1e})", np.abs(l0 - F["lp"]).max() < 1e-5))

    # 5. linealidad: α chico sigue la predicción cuadrática, α grande se aparta
    v = rng.normal(size=F["h"].shape); v *= 0.5 * np.linalg.norm(F["h"]) / np.linalg.norm(v)
    r_small = kl(F["lp"], logprobs_with(mw, ids, layer, F["h"] + 0.1 * v)) / (0.5 * 0.1 ** 2 * quad(F["A"], v))
    r_big = kl(F["lp"], logprobs_with(mw, ids, layer, F["h"] + 4.0 * v)) / (0.5 * 4.0 ** 2 * quad(F["A"], v))
    tests.append((f"linealidad: α=0.1 cociente {r_small:.3f}, α=4 cociente {r_big:.3f} (fuera del rango lineal)",
                  abs(r_small - 1) < 0.1 and abs(r_big - 1) > 0.2))

    # 6. τ y modos: la fórmula cerrada == el promedio a mano de ½ δᵀGδ
    D = rng.normal(size=(7, F["h"].shape[0]))
    Dc = D - D.mean(0)
    brute = 0.5 * np.mean([quad(F["A"], x) for x in Dc]) * len(Dc) / (len(Dc) - 1)
    md = modes(F["A"], Dc)
    tests.append((f"τ cerrada {temperature(F['A'], Dc):.4g} == a mano {brute:.4g}, modos suman {md.sum():.3f}",
                  np.isclose(temperature(F["A"], Dc), brute) and np.isclose(md.sum(), 1.0)))

    # 7. cobertura y enriquecimiento: sensibilidad plantada dentro del span -> c=1, E=d/k; al azar -> E≈1
    d = 512
    Qr, _ = np.linalg.qr(rng.normal(size=(d, 6)))
    Dp = (rng.normal(size=(7, 6)) @ Qr.T)
    A_in = rng.normal(size=(4, 6)) @ Qr.T
    A_rand = rng.normal(size=(40, d))
    ci, cr = coverage(A_in, Dp - Dp.mean(0)), coverage(A_rand, Dp - Dp.mean(0))
    tests.append((f"cobertura plantada: dentro c={ci['coverage']:.3f} E={ci['enrichment']:.0f}, "
                  f"al azar E={cr['enrichment']:.2f}",
                  ci["coverage"] > 0.999 and abs(ci["enrichment"] - d / ci["dim"]) < 1 and 0.5 < cr["enrichment"] < 1.6))

    print(f"BELENC-Fisher v{VERSION} selftest")
    for name, ok in tests:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in tests) else 1


# =================================================================== run ===
def cmd_run(a) -> int:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from crad_contract import ContractError
    from dose_probe import captured_turn
    from patch_how import SELF_TOL, Tap, readout_check, _gate_reproduces, _load_model, _setup, \
        install_scoring, rebuild_context
    import belenc as B
    cd = Path(a.capture_dir)
    v13, crad_base, sc = _setup(cd)
    z, args = sc.z, sc.args
    lo, hi = sc.band
    sup_lo, sup_hi = sc.superset
    T, n_tok, focus = a.target_turn, int(args["n-tokens"]), int(args["focus-from-turn"])
    for l in a.layers:
        if not (sup_lo <= l < sup_hi - 1):
            raise ContractError(f"capa {l}: Σ necesita phi capturado y no la última capa ([{sup_lo},{sup_hi - 1}))")
    n_draws = int(z["draw"][z["kind"] == 1].max()) + 1
    dec = {k: sc.decision(k, T, a.anchor) for k in range(n_draws)}
    use = [k for k, d in dec.items() if d.outcome in ("TRUTH", "SELF_FICTION") and d.row is not None]
    from patch_probe import captured_state
    H = {(k, l): captured_state(z, k, T, dec[k].row, l, sup_lo)[0] for k in use for l in a.layers}

    mw, temp, secret_ids = _load_model(v13, crad_base, args)
    ramp = crad_base.build_real_ramp(args["secret"], n_turns=8)
    stash = {}
    install_scoring(mw, stash)
    tap = Tap(mw)
    cfg = B.BelencConfig(seed=a.seed)
    out_layers, per_draw = {}, []
    rng = np.random.default_rng(a.seed)
    fam_dirs = {}
    for l in a.layers:
        mat = B.load_matrix(cd, l, T, cfg.min_draws, tuple(a.pair) if a.pair else None)
        cache = {}
        dirs = {f: [B.direction(mat, f, np.random.default_rng([a.seed, i, l]), cache, cfg) for i in range(a.n_dirs)]
                for f in ("high", "low", "random")}
        if mat.real_dir is not None:
            dirs["real"] = [mat.real_dir]
        ax = sc.axes_at_layer(l)
        dirs["crad_L"], dirs["crad_R"] = [ax["uL"]], [ax["uR"]]
        fam_dirs[l] = {f: [(u, B.sigma_along(mat, u)) for u in us] for f, us in dirs.items()}

    ctx_of, A_of, H_live = {}, {}, {}
    for k in use:
        ctx = rebuild_context(mw, ramp, v13.seed_everything, args["seed"], k, T, focus, n_tok, temp, lo, hi)
        toks, scores, cap_ids = _gate_reproduces(mw, tap, stash, ctx, z, k, T, n_tok, temp, lo, hi)
        r = dec[k].row
        ids = forward_ids(mw, mw._ids(ctx["hist"])["input_ids"][0].tolist(), cap_ids[:r])
        ctx_of[k] = (ctx, ids, scores[r])
        for l in a.layers:
            F = fisher_at(mw, ids, l, a.K)
            nbrs = []
            for rr in (r - 1, r + 1):
                try:
                    nbrs.append(captured_state(z, k, T, rr, l, sup_lo)[0])
                except ContractError:
                    pass
            ok, rel, nb = clock_check(F["h"], H[(k, l)], nbrs)
            if not ok:
                tap.remove()
                raise ContractError(f"draw {k} L{l}: el forward no cae en la fila del contrato (rel {rel:.2e}, "
                                    f"filas vecinas {nb:.2e}; pide <= {FWD_TOL:g} y <= {CLOCK_SEP:g} x vecinas)")
            H_live[(k, l)] = F["h"]
            ok_lp, dd, dd_nb = readout_check(F["lp"], scores, r)
            if not ok_lp:
                tap.remove()
                raise ContractError(f"draw {k} L{l}: los logits del forward no caen en la fila del contrato "
                                    f"(top-5 {dd:.3f} nats, filas vecinas {dd_nb:.3f})")
            A_of[(k, l)] = F["A"]
            per_draw.append({"draw": k, "layer": l, "outcome": dec[k].outcome, "mass_topK": F["mass"],
                             "trG": float(np.sum(F["A"] ** 2)), "fwd_rel": rel, "neighbor_rel": nb, "fwd_vs_gen": dd, "fwd_vs_gen_neighbors": dd_nb})
        print(f"[fisher] draw {k:>2} {dec[k].outcome:12s} " + " ".join(
            f"L{x['layer']} trG {x['trG']:.3g} masa {x['mass_topK']:.3f}" for x in per_draw[-len(a.layers):]), flush=True)

    for l in a.layers:
        Dc = np.stack([H[(k, l)] for k in use]); Dc = Dc - Dc.mean(0)
        taus = [temperature(A_of[(k, l)], Dc) for k in use]
        covs = [coverage(A_of[(k, l)], Dc) for k in use]
        mds = [modes(A_of[(k, l)], Dc)[:3] for k in use]
        chi = {}
        for f, lst in fam_dirs[l].items():
            vals = [0.5 * s ** 2 * quad(A_of[(k, l)], u) for k in use for u, s in lst]
            unit = [quad(A_of[(k, l)], u) for k in use for u, _ in lst]
            chi[f] = {"kl_per_sigma2": float(np.median(vals)), "per_unit_norm2": float(np.median(unit)),
                      "sigma_median": float(np.median([s for _, s in lst]))}
        rnd = chi["random"]["kl_per_sigma2"]
        out_layers[str(l)] = {
            "tau_median": float(np.median(taus)), "tau_by_outcome": {
                o: float(np.median([t for t, k in zip(taus, use) if dec[k].outcome == o])) for o in ("TRUTH", "SELF_FICTION")},
            "coverage_median": float(np.median([c["coverage"] for c in covs])),
            "enrichment_median": float(np.median([c["enrichment"] for c in covs])),
            "latent_fragility_median": float(np.median([c["latent_fragility"] for c in covs])),
            "span_dim": covs[0]["dim"], "top3_modes_median": np.median(np.stack(mds), 0).tolist(),
            "chi": chi, "kl_per_sigma2": {f: v["kl_per_sigma2"] for f, v in chi.items()},
            "chi_ratio_vs_random": {f: v["kl_per_sigma2"] / max(rnd, 1e-300) for f, v in chi.items()}}

    lin = None
    if a.pair:
        s, t = a.pair
        if t not in ctx_of or s not in use:
            raise ContractError(f"el par {a.pair} no tiene filas de decisión utilizables")
        ctx, ids, gen_scores = ctx_of[t]
        lin = {}
        for l in a.layers:
            ht = H_live[(t, l)]                 # estado del forward, el mismo punto donde se calculó G
            delta = H[(s, l)] - H[(t, l)]       # diferencia entre los dos phi capturados
            base = logprobs_with(mw, ids, l, None)
            if np.abs(logprobs_with(mw, ids, l, ht) - base).max() > SELF_TOL:
                tap.remove()
                raise ContractError(f"L{l}: α=0 mueve los logits; el hook no escribe donde se lee")
            pred_unit = 0.5 * quad(A_of[(t, l)], delta)
            rows = []
            for al in a.alphas:
                lq = logprobs_with(mw, ids, l, ht + al * delta)
                lm_ = logprobs_with(mw, ids, l, ht - al * delta)
                meas, meas_m, pred = kl(base, lq), kl(base, lm_), al ** 2 * pred_unit
                sym = abs(meas - meas_m) / max(meas + meas_m, 1e-300)
                rows.append({"alpha": al, "kl_measured": meas, "kl_measured_minus": meas_m, "kl_predicted": pred,
                             "ratio": meas / max(pred, 1e-300), "symmetry": sym, "above_floor": meas > KL_FLOOR,
                             "d_lp_secret": float(max(lq[i] for i in secret_ids) - max(base[i] for i in secret_ids))})
            ok_rows = [r for r in rows if r["above_floor"] and abs(r["ratio"] - 1) <= 0.2 and r["symmetry"] <= SYM_TOL]
            lin_range = max((r["alpha"] for r in ok_rows), default=None)
            lin[str(l)] = {"rows": rows, "linear_range_alpha": lin_range, "delta_fisher_norm": float(np.sqrt(2 * pred_unit))}
            print(f"[fisher] L{l} par {s}->{t}: " + " ".join(f"α{r['alpha']:g} {r['ratio']:.2f}" for r in rows), flush=True)
    tap.remove()

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = cd / "belenc_fisher" / stamp
    out.mkdir(parents=True)
    meta = {"fisher_version": VERSION, **sc.stamp(), "capture": cd.name, "target_turn": T, "row": "anchor",
            "K": a.K, "fwd_tol": FWD_TOL, "clock_sep": CLOCK_SEP, "draws": use, "layers": out_layers, "per_draw": per_draw, "linearity": lin, "pair": a.pair}
    (out / "fisher.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    L = [f"# BELENC-Fisher v{VERSION} — `{cd.name}` · fila de ancla del turno {T}", "",
         f"calibration_id `{sc.calibration_id}` · draws {use} · top-K {a.K} · masa top-K mediana "
         f"{np.median([x['mass_topK'] for x in per_draw]):.3f}", "",
         "| capa | τ (nats) | τ TRUTH | τ FIC | cobertura | enriquecimiento E | fragilidad latente | "
         "χ high | χ low | χ random | χ real | χ L | χ R | L/rand | R/rand |", "|" + "---|" * 15]
    for l in a.layers:
        o = out_layers[str(l)]
        k = o["kl_per_sigma2"]; rr = o["chi_ratio_vs_random"]
        g = lambda f: f"{k[f]:.3g}" if f in k else "—"
        L.append(f"| {l} | {o['tau_median']:.3g} | {o['tau_by_outcome']['TRUTH']:.3g} | {o['tau_by_outcome']['SELF_FICTION']:.3g} "
                 f"| {o['coverage_median']:.4f} | {o['enrichment_median']:.1f} | {o['latent_fragility_median']:.4f} "
                 f"| {g('high')} | {g('low')} | {g('random')} | {g('real')} | {g('crad_L')} | {g('crad_R')} "
                 f"| {rr['crad_L']:.2f} | {rr['crad_R']:.2f} |")
    if lin:
        L += ["", f"## Linealidad del parche {a.pair[0]} → {a.pair[1]} (KL medida / predicha ½α²δᵀGδ)", "",
              "| capa | " + " | ".join(f"α={x:g}" for x in a.alphas) + " | rango lineal α | ‖δ‖_G (√nats) |",
              "|" + "---|" * (len(a.alphas) + 3)]
        for l in a.layers:
            q = lin[str(l)]
            L.append(f"| {l} | " + " | ".join(f"{r['ratio']:.2f}" + ("" if r["above_floor"] else " (bajo piso)")
                                            + ("" if r["symmetry"] <= SYM_TOL else " (asim)") for r in q["rows"])
                     + f" | {q['linear_range_alpha']} | {q['delta_fisher_norm']:.2f} |")
    L += ["", "χ = KL por σ² natural a lo largo de la dirección (½σ²uᵀGu). L/rand y R/rand cerca de 1: los ejes de CRAD "
          "no son direcciones de respuesta. E = cobertura / (dim/d): cuántas veces más sensibilidad cae en lo que el "
          "modelo varía que en un subespacio al azar. Pasa `fisher.json` a belenc.py con --fisher-json para leer los "
          "LOD en nats.", "", "**Una captura, una semilla: una sonda, no una prueba.**"]
    (out / "FISHER.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"[fisher] escribí {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="BELENC-Fisher: respuesta lineal y métrica de Fisher en la fila de decisión")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("capture_dir", nargs="?")
    ap.add_argument("--layers", nargs="+", type=int, default=[14, 17, 21])
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--pair", nargs=2, type=int, default=None)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--n-dirs", type=int, default=8)
    ap.add_argument("--target-turn", type=int, default=7)
    ap.add_argument("--anchor", default=r"access code is")
    ap.add_argument("--seed", type=int, default=20261022)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.capture_dir:
        ap.print_help(); return 0
    return cmd_run(a)


if __name__ == "__main__":
    raise SystemExit(main())
