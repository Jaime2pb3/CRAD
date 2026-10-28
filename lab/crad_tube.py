#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRAD tube v0.1 — el cilindro que sigue la trayectoria
TwoQuarks Research — Luis Jaime Ledesma Perez

Idea (Jaime, 22-oct): las trayectorias no son planas, asi que el aro se vuelve cilindro
y el panal se vuelve malla.

  - Cilindro: vecindad tubular alrededor de la trayectoria media del turno
    (curva principal discreta, Hastie y Stuetzle 1989). En cada fila t se resta el
    centro de la rebanada y se proyecta fuera la tangente local. Eso quita la
    identidad del token y el formato, que es lo que dominaba los mapas de aros.
  - Malla: hecha de los datos, no una rejilla fija. Nodos = (draw, fila).
    Aristas = vecinos dentro de la rebanada + cada draw consigo mismo en t+1.
  - Lectura: tangling (Russo et al. 2018, Neuron). Q(t) es alto cuando estados
    cercanos en la fila t salen en direcciones distintas hacia t+1: una bifurcacion.
    La velocidad es action[t+1] = phi[t+1] - phi[t] del contrato, nunca entre turnos.

Contrato
  - Ciego a outcomes: el mapa usa solo phi, draw, turno y fila. Los outcomes de
    crad_contract se pegan al final como lectura.
  - El umbral sale de un nulo: se baraja que velocidad va con que estado dentro de
    la misma fila (conserva ambas nubes, rompe la pareja estado-velocidad).
  - Una captura es una sonda, no una prueba.

Lo que NO hace
  - No encuentra la DIRECCION de la decision. Encuentra el MOMENTO donde las
    trayectorias se bifurcan. La direccion la da el patch (causal) o muchos draws.
  - Una bifurcacion es cualquier punto donde los draws muestrearon tokens distintos.
    Cual de ellas importa lo dice el contrato, no este archivo.

CLI
  python crad_tube.py --selftest
  python crad_tube.py run captures/<id> --layer 17 --turn 7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

VERSION = "0.1-tube"


# ----------------------------------------------------------------- geometria --
def align_turn(phi: np.ndarray, draw: np.ndarray, row: np.ndarray, min_draws: int):
    """Arma un tensor [draws, filas, d] con las filas presentes en TODOS los draws elegidos.
    Devuelve (Y, draws_ids, filas). Se queda con el prefijo de filas que cubre >= min_draws."""
    ids = np.unique(draw)
    lengths = {int(k): int(row[draw == k].max()) + 1 for k in ids}
    # filas utiles: las que tienen al menos min_draws draws que llegan hasta ahi
    R = 0
    while sum(1 for k in ids if lengths[int(k)] > R) >= min_draws:
        R += 1
    keep = np.array([k for k in ids if lengths[int(k)] >= R], dtype=int)
    d = phi.shape[1]
    Y = np.zeros((len(keep), R, d))
    for i, k in enumerate(keep):
        for t in range(R):
            sel = np.flatnonzero((draw == k) & (row == t))
            if len(sel) != 1:
                raise SystemExit(f"[tube] draw {k} fila {t}: {len(sel)} frames, el reloj no es unico")
            Y[i, t] = phi[sel[0]]
    return Y, keep, np.arange(R)


def tube_coords(X: np.ndarray) -> np.ndarray:
    """X [D, R, d] -> coordenadas del tubo: centro por fila restado, tangente local proyectada fuera."""
    C = X.mean(0)
    Y = X - C[None]
    if X.shape[1] >= 2:
        tang = np.gradient(C, axis=0)
        for t in range(X.shape[1]):
            n = np.linalg.norm(tang[t])
            if n > 1e-12:
                v = tang[t] / n
                Y[:, t] -= np.outer(Y[:, t] @ v, v)
    return Y


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T, 0.0)


def tangling_row(x: np.ndarray, v: np.ndarray, eps_frac: float) -> float:
    """Q de una fila: mediana sobre draws de max_j |v_i - v_j|^2 / (|x_i - x_j|^2 + eps)."""
    dx, dv = _sqdist(x, x), _sqdist(v, v)
    off = dx[~np.eye(len(x), dtype=bool)]
    eps = eps_frac * float(off.mean()) if off.size and off.mean() > 0 else 1e-12
    q = dv / (dx + eps)
    np.fill_diagonal(q, 0.0)
    return float(np.median(q.max(1)))


FORK_Z = 3.5   # regla de atipicos de Iglewicz y Hoaglin (1993) sobre z robusto; fijada antes de ver datos


def tangling(Y: np.ndarray, eps_frac: float = 0.1, n_null: int = 199, seed: int = 20261022, k_enc: int = 3):
    """Dos preguntas distintas por fila t -> t+1.

    DONDE (bifurcacion): z robusto de Q(t) contra las demas filas del mismo turno.
        z = (Q - mediana) / (1.4826 * MAD). Bifurcacion si z > FORK_Z.
    SI EL ESTADO LA ANTICIPA (codificacion): se pregunta sobre la bifurcacion, no sobre la
        velocidad completa. Eje w = eje principal de las velocidades t -> t+1. Lado de cada draw =
        2-means exacto sobre v @ w (ciego). Puntaje = acierto leave-one-out de los k vecinos en t,
        con w proyectado fuera de x[t], al predecir ese lado. Nulo = barajar SOLO el lado
        (etiqueta binaria). p_enc = P(nulo >= acierto).

        Regla de diseno: el lado no puede compartir componentes con el estado en t.
        Tres intentos fallaron por romperla y quedan aqui para no repetirlos:
          1. barajar v: v contiene -x[t] (20% de falsas codificaciones).
          2. barajar x[t+1]: la historia del draw persiste en t y t+1 (100%).
          3. lado desde la rebanada t+1: su eje principal es la historia del draw (20%).
        La historia se cancela en v, y el ruido de t sobre w se quita proyectando w fuera de x[t].
        Costo conocido: si el precursor vive justo sobre el eje de salida w, este test no lo ve.
    """
    D, R, _ = Y.shape
    rng = np.random.default_rng(seed)
    Q, p_enc, acc = np.zeros(R - 1), np.ones(R - 1), np.zeros(R - 1)
    for t in range(R - 1):
        x, v = Y[:, t], Y[:, t + 1] - Y[:, t]
        Q[t] = tangling_row(x, v, eps_frac)
        side, w = fork_side(v)
        if side is None:
            acc[t], p_enc[t] = np.nan, 1.0
            continue
        xs = x - np.outer(x @ w, w)          # v contiene -x[t]; sobre el eje w ese termino es compartido
        nb = knn_index(xs, k_enc)
        acc[t] = loo_knn_acc(nb, side)
        nul = np.array([loo_knn_acc(nb, rng.permutation(side)) for _ in range(n_null)])
        p_enc[t] = (1 + np.sum(nul >= acc[t])) / (n_null + 1)
    med = float(np.median(Q))
    mad = 1.4826 * float(np.median(np.abs(Q - med)))
    z = (Q - med) / max(mad, 1e-12 * max(med, 1.0))
    return Q, z, p_enc, acc


def fork_side(v: np.ndarray, min_side: float = 0.2):
    """Lado de cada draw en la bifurcacion: 2-means exacto sobre el eje principal de las VELOCIDADES.
    La historia de cada draw se cancela en v, asi que el lado no la arrastra.
    Devuelve (lado, eje w) o (None, None) si no hay particion posible."""
    c = v - v.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    w = Vt[0]
    s = c @ w
    o = np.argsort(s, kind="stable"); xs = s[o]; m = len(xs)
    k0 = max(1, int(np.ceil(min_side * m)))
    if m - 2 * k0 < 0:
        return None, None
    ks = np.arange(k0, m - k0 + 1)
    c1, c2 = np.cumsum(xs), np.cumsum(xs * xs)
    within = (c2[ks - 1] - c1[ks - 1] ** 2 / ks) + ((c2[-1] - c2[ks - 1]) - (c1[-1] - c1[ks - 1]) ** 2 / (m - ks))
    j = int(np.argmin(within))
    lab = np.zeros(m, dtype=int); lab[o[ks[j]:]] = 1
    return lab, w


def knn_index(x: np.ndarray, k: int) -> np.ndarray:
    d2 = _sqdist(x, x); np.fill_diagonal(d2, np.inf)
    return np.argsort(d2, 1)[:, :max(1, min(k, len(x) - 1))]


def loo_knn_acc(nb: np.ndarray, lab: np.ndarray) -> float:
    vote = lab[nb].mean(1)
    pred = np.where(vote == 0.5, -1, (vote > 0.5).astype(int))   # empate = fallo
    return float(np.mean(pred == lab))


def mesh_rewire(Y: np.ndarray, k: int = 4) -> np.ndarray:
    """Malla de datos: 1 - Jaccard de los k vecinos de cada draw entre la fila t y t+1."""
    D, R, _ = Y.shape
    k = max(1, min(k, D - 1))
    nb = []
    for t in range(R):
        d2 = _sqdist(Y[:, t], Y[:, t]); np.fill_diagonal(d2, np.inf)
        nb.append([set(r) for r in np.argsort(d2, 1)[:, :k]])
    return np.array([1 - np.mean([len(a & b) / len(a | b) for a, b in zip(nb[t], nb[t + 1])])
                     for t in range(R - 1)])


# ------------------------------------------------------------------ captura --
def load_turn(cd: Path, layer: int, turn: int):
    with np.load(cd / "capture.npz", allow_pickle=False) as zf:
        z = {k: zf[k] for k in zf.files}
    lo, hi = json.loads(str(z["meta"]))["superset"]
    if not (lo <= layer < hi):
        raise SystemExit(f"[tube] capa {layer} fuera del superset [{lo},{hi})")
    m = (z["kind"] == 1) & (z["turn"] == turn)
    idx = np.flatnonzero(m)
    phi = z["phi"][idx, layer - lo].astype(np.float64)
    return phi, z["draw"][idx], z["token_in_turn"][idx], z["token"][idx], z


def cmd_run(a) -> int:
    cd = Path(a.capture_dir)
    phi, draw, row, tok, z = load_turn(cd, a.layer, a.turn)
    X, keep, rows = align_turn(phi, draw, row, a.min_draws)
    print(f"[tube] capa {a.layer} · turno {a.turn} · {len(keep)} draws · {len(rows)} filas comunes · d={X.shape[2]}",
          flush=True)
    Y = tube_coords(X)
    Q, Z, PE, ACC = tangling(Y, a.eps_frac, a.n_null, a.seed)
    M = mesh_rewire(Y, a.k)

    # lectura con el contrato, al final y aparte
    decs = {}
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from crad_contract import classify, turns_from_capture
        cfgc = json.loads((cd / "config.json").read_text())
        for k in keep:
            toks, glob = turns_from_capture(z, int(k))
            d = classify(toks, cfgc["args"]["secret"], a.turn, a.anchor, glob)
            decs[int(k)] = {"outcome": d.outcome, "anchor_row": d.row, "commit_row": d.commit_row}
    except Exception as e:  # el mapa vale sin contrato, solo pierde la lectura
        print(f"[tube] contrato no disponible ({e}); reporto solo el mapa", flush=True)

    rows_out = [{"t": int(t), "Q": float(Q[t]), "z_fork": float(Z[t]), "fork": bool(Z[t] > FORK_Z),
                 "p_enc": float(PE[t]), "encoded": bool(PE[t] < a.alpha), "knn_acc": None if np.isnan(ACC[t]) else float(ACC[t]),
                 "rewire": float(M[t])} for t in range(len(Q))]
    anchors = sorted({v["anchor_row"] for v in decs.values() if v["anchor_row"] is not None})
    meta = {"tube_version": VERSION, "capture": cd.name, "layer": a.layer, "turn": a.turn,
            "draws": keep.tolist(), "n_rows": int(len(rows)), "eps_frac": a.eps_frac, "n_null": a.n_null,
            "alpha": a.alpha, "k": a.k, "seed": a.seed, "rows": rows_out, "decisions": decs,
            "anchor_rows": anchors}
    out = cd / "tube" / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_L{a.layer}_T{a.turn}"
    out.mkdir(parents=True)
    (out / "tube.json").write_text(json.dumps(meta, indent=2) + "\n")
    md = [f"# Tube v{VERSION} — `{cd.name}` · capa {a.layer} · turno {a.turn}", "",
          "**Mapa ciego a outcomes; el contrato se pega al final. Una captura es una sonda, no una prueba.**", "",
          f"- {len(keep)} draws con las {len(rows)} filas comunes · bifurcación si z robusto > {FORK_Z} · "
          f"codificación si p_enc < {a.alpha} ({a.n_null} barajas del lado de la bifurcación entre draws)",
          f"- filas de ancla según el contrato: {anchors or '—'}", "",
          "| t → t+1 | Q | z | bifurcación | acierto kNN | p_enc | el estado la anticipa | recableo malla |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows_out:
        star = " ← ancla" if r["t"] in anchors else ""
        acc_s = "—" if r["knn_acc"] is None else f"{r['knn_acc']:.2f}"
        fork_s = "**sí**" if r["fork"] else "—"
        enc_s = "**sí**" if r["encoded"] else "—"
        md.append(f"| {r['t']} → {r['t'] + 1}{star} | {r['Q']:.2f} | {r['z_fork']:.1f} | {fork_s} | {acc_s} "
                  f"| {r['p_enc']:.3f} | {enc_s} | {r['rewire']:.2f} |")
    md += ["", "Q alto = estados cercanos que salen en direcciones distintas (Russo et al. 2018). "
           "Una bifurcación no dice qué decidió el modelo, dice dónde se separaron los caminos. "
           "p_enc chico = la geometría del estado ya separaba los caminos antes de tomarlos."]
    (out / "TUBE.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[tube] escribí {out}")
    return 0


# ----------------------------------------------------------------- selftest --
def _planted(rng, draws=16, d=256, rows=12, fork=4, pre=0.0):
    """trayectoria curva compartida, historia por draw, bifurcacion despues de la fila `fork`.
    pre > 0 planta en la fila `fork` un desplazamiento hacia el lado que el draw va a tomar."""
    T = np.cumsum(rng.normal(size=(rows, d)) * 6.0, 0)
    o = rng.integers(0, 2, draws); o[0], o[1] = 0, 1
    Tok = rng.normal(size=(2, rows, d)) * 6.0
    u = rng.normal(size=d); u /= np.linalg.norm(u)
    X = np.zeros((draws, rows, d))
    for k in range(draws):
        H = rng.normal(size=d) * 1.5
        for t in range(rows):
            X[k, t] = T[t] + H + rng.normal(size=d) + (Tok[o[k], t] if t > fork else 0.0)
            if t == fork:
                X[k, t] += pre * (2 * o[k] - 1) * u
    return X, o


def selftest() -> int:
    tests = []
    rng = np.random.default_rng(1)

    # 1. el tubo quita la trayectoria compartida: cada rebanada queda centrada y sin tangente
    X, _ = _planted(rng)
    Y = tube_coords(X)
    tests.append(("tubo: rebanadas centradas", float(np.abs(Y.mean(0)).max()) < 1e-8))

    # 2. DONDE: la bifurcacion plantada aparece como atipico, con solo 16 draws
    Q, Z, PE, _ = tangling(Y, n_null=99)
    forks = np.flatnonzero(Z > FORK_Z).tolist()
    tests.append((f"bifurcación plantada en t=4 encontrada sola (marcadas {forks}, z={Z[4]:.1f})", forks == [4]))

    # 3. sin bifurcacion: ninguna fila marcada como bifurcacion
    fl = tot = 0
    for s in range(8):
        Xn, _ = _planted(np.random.default_rng(100 + s), fork=99)
        _, Zn, _, _ = tangling(tube_coords(Xn), n_null=19)
        fl += int(np.sum(Zn > FORK_Z)); tot += len(Zn)
    tests.append((f"sin bifurcación: falsas bifurcaciones {fl}/{tot} = {fl / tot:.3f} <= 0.05", fl / tot <= 0.05))

    # 4. SI LA ANTICIPA: sin precursor el estado no la anticipa, con precursor fuerte si
    no_pre = PE[4]
    Xp, _ = _planted(np.random.default_rng(5), pre=12.0)
    _, Zp, PEp, _ = tangling(tube_coords(Xp), n_null=99)
    tests.append((f"codificación: sin precursor p_enc={no_pre:.2f} (no), con precursor p_enc={PEp[4]:.3f} (sí)",
                  no_pre >= 0.05 and PEp[4] < 0.05))

    # 5. falsas codificaciones en filas sin nada
    fe = tot = 0
    for s in range(6):
        Xn, _ = _planted(np.random.default_rng(200 + s), fork=99)
        _, _, PEn, _ = tangling(tube_coords(Xn), n_null=99)
        fe += int(np.sum(PEn < 0.05)); tot += len(PEn)
    tests.append((f"sin nada: falsas codificaciones {fe}/{tot} = {fe / tot:.3f} <= 0.10", fe / tot <= 0.10))

    # 6. invariante al orden de los draws
    perm = np.random.default_rng(3).permutation(len(X))
    Qp, _, _, _ = tangling(tube_coords(X[perm]), n_null=9)
    tests.append(("invariante al orden de los draws", np.allclose(Q, Qp)))

    # 5. align_turn arma el tensor desde columnas sueltas y respeta el reloj
    D, R, d = 5, 6, 8
    phi = rng.normal(size=(D * R, d)); draw = np.repeat(np.arange(D), R); row = np.tile(np.arange(R), D)
    A, keep, rows = align_turn(phi, draw, row, min_draws=D)
    tests.append(("align_turn reconstruye [draws, filas, d]", A.shape == (D, R, d) and np.allclose(A[2, 3], phi[2 * R + 3])))

    print(f"CRAD tube v{VERSION} selftest")
    for name, ok in tests:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in tests) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="CRAD tube: cilindro que sigue la trayectoria + tangling")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("run")
    p.add_argument("capture_dir")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--turn", type=int, default=7)
    p.add_argument("--min-draws", type=int, default=12)
    p.add_argument("--eps-frac", type=float, default=0.1)
    p.add_argument("--n-null", type=int, default=199)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--anchor", default=r"access code is")
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
