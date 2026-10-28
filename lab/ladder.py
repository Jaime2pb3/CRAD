#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ladder.py — la escalera del lab de CRAD como runner de Python (convertido de crad_lab_colab_g4.ipynb)
TwoQuarks Research — Luis Jaime Ledesma Perez

La escalera no se salta. Cada peldaño es un subcomando. Si uno termina con PELDAÑO ROJO, no sigas al
siguiente: copia el error. Los peldaños que ya tienen resultado en disco no se repiten (reanudable).

  peldaño        dónde   qué
  cpu            CPU     AST de todos los .py, imports y los selftests del lab (0, 1 y 2)
  mock           CPU     el lab completo sobre un modelo falso (captura, contrato, dose, patch)  (3)
  preflight      GPU     carga real, fp16 exacto, reproducibilidad, hook de self-patch            (4)
  cost           —       estimación de costo con los tiempos MEDIDOS por el preflight
  capture-ref    GPU     captura determinista del preprint: replicación entre hardware
  capture-patch  GPU     captura para patching (calibración T=0, 20 draws a T=0.7)
  contract       CPU     contrato sobre la captura y pares elegidos por regla (no a mano)
  probe          GPU     patch_probe sobre el par elegido
  batch          GPU     patch_batch por capa (horas: necesita --run-batch)
  belenc         GPU     BELENC-Fisher (física) y luego las fichas BELENC (estándar, CPU)
  how            GPU     story_copy, patch_how attend y patch_how trace sobre los pares de la regla

  python ladder.py cpu
  python ladder.py mock
  python ladder.py preflight --expect-gpu "RTX PRO 6000"
  python ladder.py capture-ref
  ...
  python ladder.py all --upto contract        (corre en orden y se detiene en el primero rojo)

Diferencias con el notebook: no monta Drive ni sube archivos (los resultados quedan en esta carpeta, en
captures/, preflight/, runs/ y analysis/), y no incluye el forzado de atención (patch_steer), que no se
publica. Todo lo demás llama a los mismos scripts con los mismos argumentos.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
CONFIG_REF = "configs/qwen2.5-7b_det_L14-20.json"            # caso determinista del preprint
CONFIG_PATCH = "configs/qwen2.5-7b_T0.7_cal0_L14-20_s42.json"   # fuente para patching
SUPERSET = (13, 28)            # capas guardadas en capture.npz (la 27 es post-norm: patch_batch la excluye)
PATCH_LAYERS = [14, 17, 19, 21, 23, 25]
PATCH_ALIGN = ["anchor", "commit"]
BELENC_LAYERS = [14, 17, 21]
BELENC_ROW = 2                 # fila plantada: lejos de la bifurcación natural del ancla (fila 4)
SELFTESTS = ("src/CRAD_v1.3.py --selftest", "src/crad_base.py --selftest", "CRAD_v1_5_prismatic.py --selftest",
             "crad_triad.py --selftest", "crad_contract.py selftest", "run.py --selftest",
             "capture_launch.py --selftest", "crad_rings.py --selftest", "crad_tube.py --selftest",
             "story_copy.py --selftest", "patch_how.py --selftest", "belenc_fisher.py --selftest",
             "belenc.py --selftest")
IMPORTS = ("run fingerprint launch_common crad_launch capture_launch capture crad_contract crad_triad "
           "CRAD_v1_5_prismatic dose_probe patch_probe patch_batch preflight_gpu mock_ladder "
           "crad_rings crad_tube story_copy patch_how belenc belenc_fisher").split()


class Red(SystemExit):
    def __init__(self, msg):
        super().__init__(f"\nPELDAÑO ROJO · {msg}\nNo sigas al siguiente peldaño: copia el error de arriba.")


# ================================================================ utilidades ==
def sh(cmd, env=None):
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=LAB, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, env={**os.environ, **(env or {})})
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    if rc != 0:
        raise Red(f"`{cmd}` salió con {rc}")


def latest(pattern):
    hits = sorted(glob.glob(str(LAB / pattern)))
    return hits[-1] if hits else None


def cfg_of(path):
    return json.loads((LAB / path).read_text())


def tag(path):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", cfg_of(path)["name"]).strip("-")


def rel(p):
    return str(Path(p).relative_to(LAB))


def gpu_name():
    r = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, text=True,
                       capture_output=True)
    return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else "CPU"


def need_gpu(a):
    g = gpu_name()
    if g == "CPU":
        raise Red("este peldaño necesita GPU y no hay ninguna")
    if a.expect_gpu and a.expect_gpu not in g:
        raise Red(f"la serie es {a.expect_gpu} y esta máquina tiene {g}")
    return g


def capture_ok(cap):
    return bool(cap) and json.loads(Path(cap, "capture.json").read_text())["problems"] == []


def patch_capture():
    cap = latest(f"captures/*_{tag(CONFIG_PATCH)}_cap{SUPERSET[0]}-{SUPERSET[1]}")
    if not capture_ok(cap):
        raise Red("falta la captura de patching OK: corre `capture-patch`")
    return cap


def same_gpu_or_stop(cap, what):
    """El replay solo reproduce el turno capturado en la misma GPU de la captura."""
    want = (json.loads(Path(cap, "runtime.json").read_text()).get("gpu") or {}).get("name")
    have = gpu_name()
    if want is None or have != want:
        raise Red(f"{what} necesita la GPU de la captura ({want}); esta máquina tiene {have}")
    print(f"GPU {have} == captura ✓")


# ============================================================ pares por regla ==
def fmt(tok):
    return "quote" if '"' in tok else ("space" if tok.strip() == "" else "direct")


def pairs(cap):
    """Pares TRUTH -> SELF_FICTION elegidos por regla desde el contrato: misma longitud de historia
    (mismo global_token) y formato de la fila de ancla. Devuelve (REAL_PAIR, HOW_TRACES, truth, fic)."""
    cj = json.loads(Path(cap, "contract", "CONTRACT.json").read_text())
    truth = [d["draw"] for d in cj["decisions"] if d["outcome"] == "TRUTH" and d["commit_row"] is not None]
    fic = [d["draw"] for d in cj["decisions"] if d["outcome"] == "SELF_FICTION" and d["commit_row"] is not None]
    dec = [d for d in cj["decisions"] if d["outcome"] in ("TRUTH", "SELF_FICTION") and d["row"] is not None]

    def pick(src_fmt, tgt_fmt):
        for a in sorted(dec, key=lambda d: d["draw"]):
            if a["outcome"] != "TRUTH" or fmt(a["token"]) != src_fmt:
                continue
            for b in sorted(dec, key=lambda d: d["draw"]):
                if b["outcome"] == "SELF_FICTION" and fmt(b["token"]) == tgt_fmt and b["global_token"] == a["global_token"]:
                    return a["draw"], b["draw"]
        return None

    anchor_pair = pick("direct", "quote")    # ␣BLUE contra ␣": la fuente del formato
    commit_pair = pick("quote", "quote")     # los dos abren con ␣": difieren en el contenido
    real = anchor_pair or commit_pair
    traces = [(p[0], p[1], al) for p, al in ((anchor_pair, "anchor"), (commit_pair, "commit")) if p]
    return cj, dec, real, traces, truth, fic


# ================================================================== peldaños ==
def step_cpu(a):
    for f in sorted(glob.glob(str(LAB / "*.py")) + glob.glob(str(LAB / "src" / "*.py"))):
        ast.parse(Path(f).read_text(), f)
    print("peldaño 0 · AST OK")
    sh(f'{sys.executable} -c "import {", ".join(IMPORTS)}; print(\'peldaño 1 · import OK\')"')
    for c in SELFTESTS:
        sh(f"{sys.executable} {c}")
    print(f"\npeldaño 2 · los {len(SELFTESTS)} selftests en verde")


def step_mock(a):
    sh(f"{sys.executable} mock_ladder.py", env={"CRAD_MOCK_MODEL": "1"})


def step_preflight(a):
    need_gpu(a)
    sh(f"{sys.executable} preflight_gpu.py {CONFIG_PATCH}")


def step_cost(a):
    pf_path = latest("preflight/*.json")
    if not pf_path:
        raise Red("no hay preflight: corre `preflight`")
    pf = json.loads(Path(pf_path).read_text())
    spt = pf["sec_per_traced_turn"] * max(1.0, pf["n_tokens"] / max(pf["tokens_in_timed_turn"], 1)) * 1.3
    load_s = pf["load_seconds"] + 30

    def capture_turns(path):
        x = cfg_of(path)["args"]
        base = 0.0 if x["temperature-mode"] == "off" else x["temp"]
        ct = base if x["calibration-temp"] is None else x["calibration-temp"]
        at = base if x["analysis-temp"] is None else x["analysis-temp"]
        return (1 if ct == 0 else x["calibration-draws"]) * 3 * 8 + (1 if at == 0 else x["draws"]) * 8

    def batch_turns_one_layer(nt, nf, n_align=len(PATCH_ALIGN)):
        pairs_, targets = 2 * nt * nf, nt + nf
        return targets * 7 + targets + targets * n_align + pairs_ * n_align * 5

    def cost(turns, loads=1):
        h = (turns * spt + loads * load_s) / 3600
        return f"{turns:>6} turnos ≈ {h:5.2f} h" + (f" ≈ {h * a.units_per_hour:6.1f} unidades" if a.units_per_hour else "")

    print(f"GPU {pf['runtime']['gpu'].get('name')} · {pf['tokens_per_second']} tok/s · {spt:.1f} s/turno (techo) · "
          f"carga {load_s:.0f} s\n")
    print("captura referencia :", cost(capture_turns(CONFIG_REF)))
    print("captura patching   :", cost(capture_turns(CONFIG_PATCH)))
    for nt, nf in ((3, 3), (5, 5), (8, 8)):
        print(f"patch_batch {nt}T×{nf}F, {len(PATCH_LAYERS)} capas:",
              cost(batch_turns_one_layer(nt, nf) * len(PATCH_LAYERS), loads=len(PATCH_LAYERS)))


def step_capture_ref(a):
    pat = f"captures/*_{tag(CONFIG_REF)}_cap{SUPERSET[0]}-{SUPERSET[1]}"
    cap = latest(pat)
    if capture_ok(cap):
        print("ya existe una captura OK:", rel(cap))
    else:
        need_gpu(a)
        sh(f"{sys.executable} capture.py {CONFIG_REF} --superset {SUPERSET[0]} {SUPERSET[1]}")
        cap = latest(pat)
        if not capture_ok(cap):
            raise Red("la captura de referencia no quedó OK: revisa CAPTURE.md")
    s = json.loads(Path(cap, "summary.json").read_text())
    g, d0 = s["geometry"], s["draws"][0]
    ov = [f"{c['global_token']} {c['token']!r}" for c in d0["candidates"] if c["classification"] == "OVERRIDE"]
    rows = [("cos(L,R) mediana", "0.1524", f"{g['cos_LR_median']:.4f}"),
            ("coactivación", "38.8%", f"{100 * g['coactivation_rate_all_states']:.1f}%"),
            ("token del evento (fuga completa)", "220", str(d0["event_global_token"])),
            ("OVERRIDE incluye", "216 ' is'", ", ".join(ov[:8]) or "—")]
    print(f"\n{'':34} {'preprint':>10}  esta máquina")
    for r in rows:
        print(f"{r[0]:34} {r[1]:>10}  {r[2]}")
    print("\nSi coincide, el instrumento replica en este hardware. Si no, NO es un fallo del runner: "
          "es un hallazgo de procedencia y se anota tal cual.")


def step_capture_patch(a):
    pat = f"captures/*_{tag(CONFIG_PATCH)}_cap{SUPERSET[0]}-{SUPERSET[1]}"
    cap = latest(pat)
    if capture_ok(cap):
        print("ya existe una captura OK:", rel(cap))
    else:
        need_gpu(a)
        sh(f"{sys.executable} capture.py {CONFIG_PATCH} --superset {SUPERSET[0]} {SUPERSET[1]}")
        cap = latest(pat)
        if not capture_ok(cap):
            raise Red("la captura de patching no quedó OK: revisa CAPTURE.md")
    print(Path(cap, "CAPTURE.md").read_text()[:2500])


def step_contract(a):
    cap = patch_capture()
    sh(f"{sys.executable} crad_contract.py check {rel(cap)}")
    cj, dec, real, traces, truth, fic = pairs(cap)
    print(f"\ncalibration_id {cj['calibration_id']} · outcomes {cj['outcomes']}")
    print(f"TRUTH {truth} · SELF_FICTION {fic}")
    if not truth or not fic:
        raise Red("no hay al menos un TRUTH y un SELF_FICTION: no hay par para parchear. No fuerces nada "
                  "(otra seed u otra config = otra serie)")
    print("\nformato de la fila de ancla:")
    for d in sorted(dec, key=lambda d: d["draw"]):
        print(f"  draw {d['draw']:>2} {d['outcome']:12} {d['token']!r:10} {fmt(d['token']):6} global {d['global_token']}")
    print(f"\nREAL_PAIR = {real} · HOW_TRACES = {traces}")
    if real is None:
        raise Red("no hay ningún par TRUTH/SELF_FICTION con la misma historia. No fuerces un par")
    if len(traces) < 2:
        print("⚠ falta uno de los dos pares: el cómo correrá solo con el que sí existe. Anótalo en la bitácora.")


def step_probe(a):
    cap = patch_capture()
    _, _, real, _, _, _ = pairs(cap)
    if real is None:
        raise Red("no hay par por regla: corre `contract` y revisa la tabla")
    s, t = real
    done = latest(f"{rel(cap)}/patch_probe/*_src{s}_tgt{t}/PATCH.md")
    if done:
        print("ya existe:", rel(done))
    else:
        need_gpu(a)
        sh(f"{sys.executable} patch_probe.py {rel(cap)} --source {s} --target {t} "
           f"--layers {' '.join(map(str, PATCH_LAYERS))}")
        done = latest(f"{rel(cap)}/patch_probe/*_src{s}_tgt{t}/PATCH.md")
    print(Path(done).read_text())


def step_batch(a):
    cap = patch_capture()
    _, _, _, _, truth, fic = pairs(cap)

    def batch_done(layer):
        for p in glob.glob(f"{cap}/patch_batch/*/summary.json"):
            m = json.loads(Path(p).read_text())
            if m["layers"] == [layer] and sorted(m["align"]) == sorted(PATCH_ALIGN):
                return p
        return None

    todo = [l for l in PATCH_LAYERS if not batch_done(l)]
    print(f"TRUTH {len(truth)} · SELF_FICTION {len(fic)} · capas pendientes {todo}")
    if len(truth) < 3 or len(fic) < 3:
        raise Red("patch_batch necesita >= 3 TRUTH y >= 3 SELF_FICTION (leave-one-pair-out)")
    if not a.run_batch:
        print("\nsin --run-batch no corre (tarda horas). Revisa `cost` y vuelve con --run-batch.")
        return
    need_gpu(a)
    for l in todo:
        sh(f"{sys.executable} patch_batch.py {rel(cap)} --layers {l} --align {' '.join(PATCH_ALIGN)}")
    for l in PATCH_LAYERS:
        p = batch_done(l)
        if p:
            print(Path(p).with_name("BATCH.md").read_text()[:6000], "\n")


def step_belenc(a):
    cap = patch_capture()
    _, _, real, _, _, _ = pairs(cap)
    if real is None:
        raise Red("no hay par por regla: corre `contract`")
    same_gpu_or_stop(cap, "BELENC-Fisher")

    def fisher_done():
        for p in sorted(glob.glob(f"{cap}/belenc_fisher/*/fisher.json")):
            m = json.loads(Path(p).read_text())
            if m.get("pair") == list(real) and sorted(m["layers"]) == sorted(map(str, BELENC_LAYERS)):
                return p
        return None

    fj = fisher_done()
    if fj:
        print("ya existe:", rel(fj))
    else:
        sh(f"{sys.executable} belenc_fisher.py {rel(cap)} --layers {' '.join(map(str, BELENC_LAYERS))} "
           f"--pair {real[0]} {real[1]}")
        fj = fisher_done()
    print(Path(fj).with_name("FISHER.md").read_text())
    for L in BELENC_LAYERS:
        if glob.glob(f"{cap}/belenc/*_L{L}_T7_r{BELENC_ROW}/BELENC.md"):
            print(f"ya existe la ficha de L{L}")
            continue
        sh(f"{sys.executable} belenc.py run {rel(cap)} --layer {L} --row {BELENC_ROW} --real-pair {real[0]} {real[1]} "
           f"--instruments oracle pca tube_fork tube_anticip rings --fisher-json {rel(fj)}")
    for L in BELENC_LAYERS:
        p = sorted(glob.glob(f"{cap}/belenc/*_L{L}_T7_r{BELENC_ROW}/BELENC.md"))[-1]
        print(Path(p).read_text()[:8000], "\n")


def step_how(a):
    cap = patch_capture()
    _, _, _, traces, _, _ = pairs(cap)
    sh(f"{sys.executable} story_copy.py {rel(cap)}")
    print(Path(cap, "story_copy", "STORY_COPY.md").read_text())
    same_gpu_or_stop(cap, "patch_how")
    if not glob.glob(f"{cap}/patch_how/*_attend/HOW_ATTEND.md"):
        sh(f"{sys.executable} patch_how.py {rel(cap)} attend")
    for s, t, al in traces:
        if glob.glob(f"{cap}/patch_how/*_trace_src{s}_tgt{t}_{al}/HOW_TRACE.md"):
            print(f"ya existe el trace {s} → {t} ({al})")
            continue
        sh(f"{sys.executable} patch_how.py {rel(cap)} trace --source {s} --target {t} --align {al}")
    for p in sorted(glob.glob(f"{cap}/patch_how/*/HOW_*.md")):
        print(Path(p).read_text()[:6000], "\n")


STEPS = {"cpu": step_cpu, "mock": step_mock, "preflight": step_preflight, "cost": step_cost,
         "capture-ref": step_capture_ref, "capture-patch": step_capture_patch, "contract": step_contract,
         "probe": step_probe, "batch": step_batch, "belenc": step_belenc, "how": step_how}


def main() -> int:
    ap = argparse.ArgumentParser(description="Escalera del lab de CRAD (un peldaño por subcomando)")
    ap.add_argument("step", choices=list(STEPS) + ["all"])
    ap.add_argument("--upto", choices=list(STEPS), default="how", help="con `all`: último peldaño a correr")
    ap.add_argument("--expect-gpu", default=None, help='p. ej. "RTX PRO 6000" o "A100": se detiene si no coincide')
    ap.add_argument("--units-per-hour", type=float, default=None, help="costo por hora de la GPU, para `cost`")
    ap.add_argument("--run-batch", action="store_true", help="enciende patch_batch (tarda horas)")
    a = ap.parse_args()
    os.environ.pop("CRAD_MOCK_MODEL", None)          # el mock solo existe dentro del peldaño `mock`
    if a.step != "all":
        STEPS[a.step](a)
        return 0
    for name, fn in STEPS.items():
        print(f"\n===== peldaño {name} =====", flush=True)
        fn(a)
        if name == a.upto:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
