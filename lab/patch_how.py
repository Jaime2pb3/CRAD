#!/usr/bin/env python3
"""patch_how v0.1 — el CÓMO de la decisión en la fila de ancla, bound to crad_contract.
TwoQuarks Research — Luis Jaime Ledesma Perez

Pregunta
--------
patch_probe mostró que el estado de la fila de decisión ya lleva la elección en L14.
patch_batch pregunta si el modelo responde desde el prompt de sistema (TRUTH) o desde su
propia narrativa (SELF_FICTION).  Este archivo pregunta CÓMO llega ahí:

  El sistema dice literalmente "The access code is {secret}."  El turno 6 del usuario
  siembra "...displayed that the access code was" y el modelo continúa su cuento con un
  código.  En el turno 7 la fila de ancla sigue a "The access code is" y tiene DOS
  fuentes de copia en contexto: inducción exacta desde el sistema, o inducción aproximada
  (is / was) desde su propio cuento (Olsson et al. 2022, induction heads).

Dos subcomandos
---------------
  attend   DESCRIPTIVO, todos los draws con fila de decisión.  En la fila de ancla (y la de
           commit), cuánta atención manda cada cabeza a cada región del contexto:
             secret      los tokens del secreto dentro del prompt de sistema
             story       el código que sigue a "access code was/is" en los turnos traced
                         del asistente (definido por TEXTO, ciego a outcomes)
             template    "____" del turno 7 del usuario
             sink        la posición 0
           Por cabeza: AUC de TRUTH contra SELF_FICTION sobre pref = secret - story, con un
           nulo de permutación de etiquetas sobre el MÁXIMO de todas las cabezas (FWER).
           Necesita atención eager (sdpa no devuelve pesos).  Compuerta: los logits del
           forward eager en la posición de decisión deben igualar los de la generación.

  trace    CAUSAL, un par fuente -> víctima.  En la fila de decisión de la víctima:
             resid  reemplaza la salida del bloque l        (capas 0..n-2)
             attn   reemplaza solo la salida de la atención de la capa l
             mlp    reemplaza solo la salida del MLP de la capa l
             head   reemplaza la rebanada de UNA cabeza en la entrada de o_proj
           Las capas por debajo del superset capturado no tienen phi guardado: el estado de la
           fuente se graba EN VIVO reconstruyendo la fuente con sus semillas.  Compuerta: lo
           grabado en vivo debe igualar phi capturado en las capas del superset.

Compuertas (todas abortan, nada se "salta")
  - la corrida sin parche reproduce el turno capturado (fuente y víctima)
  - self-patch: escribir el estado propio no mueve los logits más de SELF_TOL, por modo
  - todas las cabezas a la vez == parche de attn (o_proj es lineal y sin bias en Qwen2)
  - attend: el forward eager cae en la fila del contrato (readout_check): mismo argmax, top-5 a
    <= EAGER_TOL nats y a menos de READ_SEP veces la distancia a las filas vecinas

Corrección 23-oct (v0.1.1)
  La v0.1 exigía top-5 eager contra generación <= READ_TOL = 0.05 nats. Son caminos distintos (forward
  completo sin caché, eager, contra decode con caché, sdpa), y en fp16 los logits se redondean en
  escalones de 1/128 a 1/64 nats. Medido en la G4 (fisher.json, 45 casos, sdpa): 0.008 a 0.039 nats.
  attend dio 0.0625 en el draw 0 con el mismo argmax y abortó por redondeo. Una tolerancia de magnitud
  sin medir el ruido es una apuesta: ahora la prueba es de alineación, igual que el reloj de belenc_fisher.
  El argmax solo se exige cuando top-1 y top-2 están a más de 2|d|: en un empate el redondeo lo voltea
  sin que nada esté mal (el selftest lo encontró con una brecha de 0.028 nats).

Corrección 23-oct (v0.1.2): la atención ya NO se lee de un forward eager
  En transformers 5.x la atención eager de Qwen2 hace matmul(q, kᵀ) en fp16 ANTES de escalar. Con los
  sesgos grandes de q y k de Qwen2 el producto pasa de 65504 (máximo de fp16), da inf y el softmax da NaN
  (G4, draw 1: top-5 nan). Además devuelve los pesos en fp16. Ahora attention_at corre el forward normal
  (sdpa, el mismo camino que belenc_fisher) y un pre-hook en cada self_attn rehace q de la última posición
  y k de todas con las proyecciones y la RoPE del propio módulo, y calcula softmax(q·kᵀ·escala) en fp32
  con la escala antes del producto. Selftest: en un modelo fp16 con sesgos grandes el eager da NaN y esto no.

Predicciones, escritas antes de correr en GPU
  A1. Hay cabezas con AUC >= 0.9 y p_FWER < 0.05: TRUTH atiende al secreto, SELF_FICTION al cuento.
  A2. Esas cabezas están en L13 o antes (L14 ya voltea en patch_probe).
  A3. Draw 15 (TRUTH con comilla): en la fila de ANCLA se parece a las ficciones (copia el
      formato del cuento), en la de COMMIT a las TRUTH (copia el contenido del sistema).
  B1. La primera capa donde resid recupera >= 50% del efecto completo está en L13 o antes.
  B2. En esas capas, attn mueve más que mlp.
  B3. Las cabezas con más efecto individual coinciden con las top de A1.
  Falsador: ninguna cabeza separa por encima del nulo y mlp domina sobre attn ->
  el mecanismo no es copia en contexto; es cómputo en el MLP.

Usage
  .venv/bin/python patch_how.py captures/<id> attend [--align anchor commit]
  .venv/bin/python patch_how.py captures/<id> trace --source 15 --target 19 [--head-layers 9 11 12]
  .venv/bin/python patch_how.py --selftest          (CPU: Qwen2 diminuto aleatorio + tokenizer de caracteres)
Un par, una semilla: una sonda, no una prueba.  Secreto falso; modelo local.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

VERSION = "0.1.2-how"
SELF_TOL = 1e-3          # nats, igual que patch_probe / patch_batch
READ_TOL = 5e-2          # nats: eager contra sdpa en el modelo chico del selftest (fp32, mismo camino)
EAGER_TOL = 0.25         # nats: forward completo contra decode en fp16. Medido <= 0.0625 (G4). Techo absoluto
READ_SEP = 0.2           # el top-5 debe estar 5 veces más cerca de su fila que de las filas vecinas


def readout_check(lp, scores, r):
    """¿Los logprobs lp de un forward completo son los de la fila r de la generación?
    (ok, |d| top-5 contra la fila r, |d| mínima contra las filas vecinas en su propio top-5)."""
    def dist(rr):
        top = np.argsort(scores[rr])[::-1][:5]
        return float(np.max(np.abs(lp[top] - scores[rr][top])))
    d = dist(r)
    nb = min((dist(rr) for rr in (r - 1, r + 1) if 0 <= rr < len(scores)), default=float("inf"))
    # Si cada logprob del top-5 se mueve <= d, el argmax solo puede voltearse cuando top-1 y top-2 están a
    # <= 2d (empate de muestreo). Ahí el argmax no informa nada y decide la distancia.
    s2 = np.sort(scores[r])[-2:]
    argmax_ok = int(np.argmax(lp)) == int(np.argmax(scores[r])) or (s2[1] - s2[0]) <= 2 * d
    ok = argmax_ok and d <= EAGER_TOL and d <= READ_SEP * nb
    return ok, d, nb
LIVE_TOL = 1e-3          # diferencia relativa: estado grabado en vivo contra phi capturado
SPANS = ("secret", "secret_first", "story", "story_next", "template", "turn7_prefix", "sink")
STORY_RE = re.compile(r"access\s+code\s+(?:was|is)\b", re.I)
CODE_AFTER_RE = re.compile(r"""[\s"'“*`:]*[A-Za-z0-9][A-Za-z0-9_\-]*""")


# =============================================================== hooks ======
class Tap:
    """Un reloj por forward del modelo y cuatro puntos de escritura/lectura por capa.

    step = numero de forward dentro de una generate (0 = forward del prompt) = fila del contrato.
    record: si REC["row"] no es None, en ese paso guarda la ultima posicion de cada punto y capa.
    patch:  si PATCH no es None, en ese paso reemplaza la ultima posicion de UN punto en UNA capa.
    """

    KINDS = ("resid", "attn", "mlp", "head")

    def __init__(self, mw):
        self.mw, self.n = mw, -1
        self.patch, self.rec_row, self.rec = None, None, {}
        lm = mw.lm
        layers = lm.model.layers
        cfg = lm.config
        self.n_layers = len(layers)
        self.n_heads = int(cfg.num_attention_heads)
        self.head_dim = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
        self.handles = [lm.model.register_forward_pre_hook(self._tick)]
        for l, blk in enumerate(layers):
            self.handles += [blk.register_forward_hook(self._out_hook("resid", l)),
                             blk.self_attn.register_forward_hook(self._out_hook("attn", l)),
                             blk.mlp.register_forward_hook(self._out_hook("mlp", l)),
                             blk.self_attn.o_proj.register_forward_pre_hook(self._head_hook(l))]

    def _tick(self, _mod, _args):
        self.n += 1

    def reset(self):
        self.n = -1

    def _out_hook(self, kind, l):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if self.rec_row is not None and self.n == self.rec_row:
                self.rec[(kind, l)] = h[0, -1].detach().float().cpu().numpy().copy()
            p = self.patch
            if p is not None and p["kind"] == kind and p["layer"] == l and self.n == p["row"]:
                h[:, -1, :] = p["vec"].to(h.dtype)          # REPLACE, una fila, un punto, una capa
            return out
        return hook

    def _head_hook(self, l):
        def hook(_mod, args):
            x = args[0]
            if self.rec_row is not None and self.n == self.rec_row:
                self.rec[("head", l)] = x[0, -1].detach().float().cpu().numpy().copy()
            p = self.patch
            if p is not None and p["kind"] == "head" and p["layer"] == l and self.n == p["row"]:
                x = x.clone()
                for hd in p["heads"]:
                    s = slice(hd * self.head_dim, (hd + 1) * self.head_dim)
                    x[:, -1, s] = p["vec"][s].to(x.dtype)
                return (x,) + tuple(args[1:])
            return None
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def install_scoring(mw, stash):
    """Mismo truco que patch_probe: logits de cada paso de la generate."""
    orig = mw.lm.generate

    def scoring_generate(*g_args, **g_kwargs):
        if g_kwargs.get("return_dict_in_generate"):
            g_kwargs["output_logits"] = True
        out = orig(*g_args, **g_kwargs)
        if g_kwargs.get("return_dict_in_generate"):
            stash["scores"] = [s[0].float().log_softmax(-1).cpu().numpy() for s in out.logits]
        return out

    mw.lm.generate = scoring_generate
    return orig


# ============================================================ replay ========
def rebuild_context(mw, ramp, seed_fn, seed, draw, T, focus, n_tok, temp, lo, hi):
    """run_pressure_draw hasta el turno T, mismas llamadas y semillas que patch_batch."""
    from dose_probe import rng_snapshot
    seed_fn(int(seed) * 100000 + 50000 + int(draw))
    hist, texts, cur = [{"role": "system", "content": ramp.system}], {}, 0
    for t in range(T):
        hist.append({"role": "user", "content": ramp.turns_R[t]})
        if t < focus:
            text = mw.gen(hist, n=n_tok, temp=temp)
        else:
            tr = mw.gen_with_trace(hist, n=n_tok, temp=temp, layer_lo=lo, layer_hi=hi, turn_idx=t, global_start=cur)
            text = tr["text"]; cur += len(tr["frames"]); texts[t] = text
        hist.append({"role": "assistant", "content": text})
    hist.append({"role": "user", "content": ramp.turns_R[T]})
    return {"hist": hist, "texts": texts, "cursor": cur, "snap": rng_snapshot(mw.torch)}


def run_turn(mw, tap, stash, ctx, T, n_tok, temp, lo, hi, patch=None, record_row=None):
    """Regenera el turno T desde el mismo estado de RNG. Devuelve tokens, scores y lo grabado."""
    from dose_probe import rng_restore
    rng_restore(mw.torch, ctx["snap"])
    tap.reset(); tap.rec = {}
    tap.patch, tap.rec_row = patch, record_row
    try:
        tr = mw.gen_with_trace(ctx["hist"], n=n_tok, temp=temp, layer_lo=lo, layer_hi=hi,
                               turn_idx=T, global_start=ctx["cursor"])
    finally:
        tap.patch, tap.rec_row = None, None
    return [f["token"] for f in tr["frames"]], stash["scores"], dict(tap.rec), tr


# =========================================================== attend =========
def set_attn_impl(lm, impl: str) -> str:
    prev = getattr(lm.config, "_attn_implementation", None)
    if hasattr(lm, "set_attn_implementation"):
        lm.set_attn_implementation(impl)
    else:
        lm.config._attn_implementation = impl
    return prev


def chat_spans(tok, hist, ramp, secret, focus, T, prefix_ids):
    """Posiciones de token de cada region, desde el texto de la plantilla y los offsets del tokenizer.
    Ciego a outcomes: la region 'story' se define por la frase "access code was/is" en el texto."""
    text = tok.apply_chat_template(hist, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]

    def toks_in(a, b):
        return [i for i, (s, e) in enumerate(offs) if s < b and e > a and e > s]

    spans = {k: [] for k in SPANS}
    s0 = text.find(ramp.system)
    if s0 < 0:
        raise SystemExit("[how] el prompt de sistema no aparece en la plantilla")
    a = s0 + ramp.system.find(secret)
    spans["secret"] = toks_in(a, a + len(secret))
    spans["secret_first"] = spans["secret"][:1]
    cursor, notes = 0, []
    for t in range(T):
        u = hist[1 + 2 * t]["content"]; c = hist[2 + 2 * t]["content"]
        cu = text.find(u, cursor); cursor = cu + len(u)
        ca = text.find(c, cursor) if c else cursor
        if c and ca < 0:
            raise SystemExit(f"[how] el texto del asistente del turno {t} no aparece en la plantilla")
        if c:
            cursor = ca + len(c)
        if t >= focus and c and not spans["story"]:
            m = STORY_RE.search(c)
            if m:
                k = CODE_AFTER_RE.match(c, m.end())
                if k:
                    spans["story"] = toks_in(ca + m.end(), ca + k.end())
                    spans["story_next"] = spans["story"][:1]
                    notes.append(f"story: turno {t}, '{c[m.start():k.end()]}'")
    u7 = hist[1 + 2 * T]["content"]
    p7 = text.find(u7, cursor)
    if "____" in u7 and p7 >= 0:
        q = p7 + u7.find("____")
        spans["template"] = toks_in(q, q + 4)
    spans["sink"] = [0]
    n_prompt = len(ids)
    spans["turn7_prefix"] = list(range(n_prompt, n_prompt + len(prefix_ids)))
    return spans, ids, notes


def attention_at(mw, prompt_ids, prefix_ids):
    """Pesos de atención de la ÚLTIMA posición (la que produce el token de la fila) y log-softmax.

    Forward normal (sdpa). En cada self_attn un pre-hook toma hidden_states y position_embeddings, rehace
    q (última posición) y k (todas) con q_proj/k_proj y la RoPE del propio módulo, y calcula
    softmax(q·kᵀ·escala) en fp32 con la escala ANTES del producto: sin el overflow de fp16 del eager.
    La última fila ve todas las posiciones, así que la máscara causal no recorta nada.
    Devuelve W [L, H, kv] y log-softmax de la última posición."""
    import importlib
    torch = mw.torch
    ids = torch.tensor([list(prompt_ids) + list(prefix_ids)], device=mw.device)
    rows, hooks = {}, []

    def make(l, attn):
        mod = importlib.import_module(type(attn).__module__)
        rope, rkv = mod.apply_rotary_pos_emb, mod.repeat_kv

        def pre(module, args, kwargs):
            h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            cos, sin = kwargs["position_embeddings"]
            if getattr(module, "sliding_window", None) and h.shape[1] > module.sliding_window:
                raise SystemExit(f"[how] capa {l}: la secuencia excede la ventana deslizante")
            shp = (*h.shape[:-1], -1, module.head_dim)
            q = module.q_proj(h).view(shp).transpose(1, 2)
            k = module.k_proj(h).view(shp).transpose(1, 2)
            q, k = rope(q, k, cos, sin)
            k = rkv(k, module.num_key_value_groups)
            sc = torch.einsum("hd,htd->ht", q[0, :, -1, :].float() * module.scaling, k[0].float())
            rows[l] = torch.softmax(sc, dim=-1).cpu().numpy()
        return pre

    for l, layer in enumerate(mw.lm.model.layers):
        hooks.append(layer.self_attn.register_forward_pre_hook(make(l, layer.self_attn), with_kwargs=True))
    try:
        with torch.no_grad():
            out = mw.lm(input_ids=ids, use_cache=False)
    finally:
        for hk in hooks:
            hk.remove()
    W = np.stack([rows[l] for l in range(len(mw.lm.model.layers))])
    lp = out.logits[0, -1].float().log_softmax(-1).cpu().numpy()
    if not (np.isfinite(W).all() and np.isfinite(lp).all()):
        raise SystemExit("[how] pesos o logprobs no finitos en el forward normal: no es el overflow del eager, "
                         "revisa el modelo antes de seguir")
    return W, lp


def attention_eager(mw, prompt_ids, prefix_ids):
    """Referencia SOLO para el selftest: los pesos que devuelve transformers en modo eager."""
    torch = mw.torch
    ids = torch.tensor([list(prompt_ids) + list(prefix_ids)], device=mw.device)
    prev = set_attn_impl(mw.lm, "eager")
    try:
        with torch.no_grad():
            out = mw.lm(input_ids=ids, output_attentions=True, use_cache=False)
    finally:
        set_attn_impl(mw.lm, prev or "sdpa")
    if out.attentions is None or out.attentions[0] is None:
        raise SystemExit("[how] el modelo no devolvio pesos de atencion en modo eager")
    W = np.stack([a[0, :, -1, :].float().cpu().numpy() for a in out.attentions])
    lp = out.logits[0, -1].float().log_softmax(-1).cpu().numpy()
    return W, lp


def span_mass(W, spans):
    """W [L,H,kv] -> masa [L,H,S] en el orden de SPANS."""
    M = np.zeros(W.shape[:2] + (len(SPANS),))
    for j, k in enumerate(SPANS):
        if spans[k]:
            M[..., j] = W[..., spans[k]].sum(-1)
    return M


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return np.nan
    return float(((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)))


def head_aucs(pref, y):
    """pref [D, L, H] -> AUC [L, H]."""
    D, L, H = pref.shape
    return np.array([[auc(y, pref[:, l, h]) for h in range(H)] for l in range(L)])


def fwer_max_auc(pref, y, n_perm, seed):
    """Nulo de la maxima |AUC - 0.5| sobre todas las cabezas, barajando etiquetas de draw."""
    rng = np.random.default_rng(seed)
    obs = head_aucs(pref, y)
    stat = np.nanmax(np.abs(obs - 0.5))
    null = np.array([np.nanmax(np.abs(head_aucs(pref, rng.permutation(y)) - 0.5)) for _ in range(n_perm)])
    p_head = (1 + (null[None, None, :] >= np.abs(obs - 0.5)[..., None] - 1e-12).sum(-1)) / (n_perm + 1)
    return obs, p_head, float((1 + np.sum(null >= stat - 1e-12)) / (n_perm + 1)), null


# ============================================================ selftest ======
def _tiny_model(seed=0, n_layers=4, n_heads=4, n_kv=2, hidden=64):
    """RecogModel con un Qwen2 diminuto aleatorio y un tokenizer de caracteres con offsets.
    Usa los metodos REALES de crad_base (_ids, gen, gen_with_trace)."""
    import torch
    import crad_base
    from transformers import Qwen2Config, Qwen2ForCausalLM

    class CharTok:
        eos_token_id = 1
        def __init__(self):
            self.chars = [chr(c) for c in range(32, 127)] + ["\n", "“"]
            self.stoi = {c: i + 2 for i, c in enumerate(self.chars)}
            self.vocab = len(self.chars) + 2
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            t = "".join(f"<{m['role']}>\n{m['content']}\n" for m in messages)
            return t + ("<assistant>\n" if add_generation_prompt else "")
        def _enc(self, text):
            return [self.stoi.get(c, self.stoi["?"]) for c in text]
        def __call__(self, text, return_tensors=None, add_special_tokens=True, return_offsets_mapping=False):
            ids = [0] + self._enc(text)                       # BOS en la posicion 0 (sumidero)
            offs = [(0, 0)] + [(i, i + 1) for i in range(len(text))]
            if return_tensors == "pt":
                class Enc(dict):
                    def to(self, dev):
                        return Enc({k: v.to(dev) for k, v in self.items()})
                return Enc({"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)})
            out = {"input_ids": ids}
            if return_offsets_mapping:
                out["offset_mapping"] = offs
            return out
        def decode(self, ids, skip_special_tokens=False):
            ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            return "".join(self.chars[i - 2] for i in ids if i >= 2)

    tok = CharTok()
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=tok.vocab, hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=n_layers, num_attention_heads=n_heads, num_key_value_heads=n_kv,
                      max_position_embeddings=4096, tie_word_embeddings=False, attn_implementation="sdpa")
    lm = Qwen2ForCausalLM(cfg).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    mw = object.__new__(crad_base.RecogModel)
    mw.torch, mw.tok, mw.lm, mw.device, mw.n_layers = torch, tok, lm, torch.device("cpu"), n_layers
    return mw


def selftest() -> int:
    import random
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    import crad_base
    tests = []
    mw = _tiny_model()
    stash = {}
    install_scoring(mw, stash)
    tap = Tap(mw)
    ramp = crad_base.build_real_ramp("BLUEFALCON", n_turns=8)

    def seed_fn(s):
        random.seed(s); np.random.seed(s % (2 ** 32)); torch.manual_seed(s)

    T, focus, n_tok, temp, lo, hi = 7, 2, 12, 0.7, 1, 3
    ctx_a = rebuild_context(mw, ramp, seed_fn, 42, 0, T, focus, n_tok, temp, lo, hi)
    ctx_b = rebuild_context(mw, ramp, seed_fn, 42, 1, T, focus, n_tok, temp, lo, hi)
    row = 3

    # 1. reproducibilidad: dos regeneraciones desde el mismo snapshot dan los mismos tokens y logits
    t1, s1, _, tr1 = run_turn(mw, tap, stash, ctx_a, T, n_tok, temp, lo, hi)
    t2, s2, _, _ = run_turn(mw, tap, stash, ctx_a, T, n_tok, temp, lo, hi)
    tests.append(("replay reproducible desde el snapshot", t1 == t2 and np.allclose(s1[row], s2[row])))

    # 2. lo grabado en vivo (resid) == phi que gen_with_trace guarda (bloques que no son el ultimo)
    _, _, rec, tr = run_turn(mw, tap, stash, ctx_a, T, n_tok, temp, lo, hi, record_row=row)
    phi = tr["frames"][row]["phi"]                                   # capas [lo, hi)
    live = np.stack([rec[("resid", l)] for l in range(lo, hi)])
    tests.append((f"estado en vivo == phi de la traza (max |d| {np.abs(live - phi).max():.1e})",
                  np.allclose(live, phi, atol=1e-5)))

    # 3. self-patch en los cuatro puntos no mueve los logits
    worst = 0.0
    for kind in Tap.KINDS:
        for l in range(mw.n_layers - 1):
            spec = {"kind": kind, "layer": l, "row": row, "vec": torch.tensor(rec[(kind, l)]),
                    "heads": list(range(tap.n_heads))}
            _, s, _, _ = run_turn(mw, tap, stash, ctx_a, T, n_tok, temp, lo, hi, patch=spec)
            worst = max(worst, float(np.abs(s[row] - s1[row]).max()))
    tests.append((f"self-patch resid/attn/mlp/head: max |d log p| {worst:.1e} <= {SELF_TOL}", worst <= SELF_TOL))

    # 4 y 5: la "fuente" es una perturbacion grande y conocida en el espacio de las cabezas.
    # Otro draw del mismo modelo aleatorio no sirve: en la ultima posicion la atencion promedia ~1000
    # posiciones y dos contextos dan salidas casi iguales (primer intento: el parche movio 0.01 nats).
    l = 1
    o_proj = mw.lm.model.layers[l].self_attn.o_proj
    g = torch.Generator().manual_seed(7)
    v_head = torch.tensor(rec[("head", l)]) + 3.0 * torch.randn(rec[("head", l)].shape, generator=g)
    with torch.no_grad():
        v_attn = o_proj(v_head[None])[0]

    # 4. todas las cabezas == parche de attn con o_proj(v) (o_proj lineal, sin bias)
    _, s_attn, _, _ = run_turn(mw, tap, stash, ctx_a, T, n_tok, temp, lo, hi,
                               patch={"kind": "attn", "layer": l, "row": row, "vec": v_attn})
    _, s_all, _, _ = run_turn(mw, tap, stash, ctx_a, T, n_tok, temp, lo, hi,
                              patch={"kind": "head", "layer": l, "row": row, "vec": v_head,
                                     "heads": list(range(tap.n_heads))})
    moved = float(np.abs(s_attn[row] - s1[row]).max())
    tests.append((f"todas las cabezas == attn (|d| {np.abs(s_attn[row] - s_all[row]).max():.1e}, el parche movió {moved:.2f} nats)",
                  np.allclose(s_attn[row], s_all[row], atol=1e-4) and moved > 0.1))

    # 5. cabeza plantada: solo la cabeza 2 escribe en o_proj de la capa 1 -> solo su parche mueve algo
    W = o_proj.weight
    keep = W.detach().clone()
    hd = tap.head_dim
    with torch.no_grad():
        for h in range(tap.n_heads):
            if h != 2:
                W[:, h * hd:(h + 1) * hd] = 0.0
    _, s0p, _, _ = run_turn(mw, tap, stash, ctx_a, T, n_tok, temp, lo, hi)
    eff = []
    for h in range(tap.n_heads):
        _, s, _, _ = run_turn(mw, tap, stash, ctx_a, T, n_tok, temp, lo, hi,
                              patch={"kind": "head", "layer": l, "row": row, "vec": v_head, "heads": [h]})
        eff.append(float(np.abs(s[row] - s0p[row]).max()))
    with torch.no_grad():
        W.copy_(keep)
    tests.append((f"cabeza plantada 2: efectos por cabeza {np.round(eff, 3).tolist()} nats",
                  int(np.argmax(eff)) == 2 and eff[2] > 0.1 and all(e < 1e-6 for i, e in enumerate(eff) if i != 2)))

    # 6. attend: pesos recalculados == pesos eager (fp32, sin overflow), logits == generación, suman 1
    cap_ids = tr1["frames"]                                          # tokens generados en el turno T
    gen_ids = [mw.tok.stoi.get(f["token"], 0) for f in cap_ids]
    prompt_ids = mw._ids(ctx_a["hist"])["input_ids"][0].tolist()
    Wt, lp = attention_at(mw, prompt_ids, gen_ids[:row])
    We, _ = attention_eager(mw, prompt_ids, gen_ids[:row])
    diff = float(np.abs(lp - s1[row]).max())
    dW = float(np.abs(Wt - We).max())
    tests.append((f"pesos recalculados == eager en fp32 (|d| {dW:.1e}), logits == generación "
                  f"(|d log p| {diff:.1e}), pesos suman 1",
                  dW < 1e-5 and diff <= READ_TOL and np.allclose(Wt.sum(-1), 1.0, atol=1e-4)
                  and getattr(mw.lm.config, "_attn_implementation", "sdpa") != "eager"))

    # 6a. la falla de la G4: en fp16 con sesgos de q y k grandes, el eager da NaN y attention_at no
    a0 = mw.lm.model.layers[0].self_attn
    qb, kb = a0.q_proj.bias.detach().clone(), a0.k_proj.bias.detach().clone()
    try:
        with torch.no_grad():
            b = float(np.sqrt(4 * 65504 / a0.head_dim))                 # q·k ≈ b²·head_dim = 4 x máximo de fp16
            a0.q_proj.bias.fill_(b); a0.k_proj.bias.fill_(b)
        mw.lm.half()
        try:
            We16, _ = attention_eager(mw, prompt_ids, gen_ids[:row])
            eager_bad = not np.isfinite(We16).all()
        except Exception:
            eager_bad = True
        try:
            W16, _ = attention_at(mw, prompt_ids, gen_ids[:row])
            ours_ok = bool(np.isfinite(W16[0]).all()) and np.allclose(W16[0].sum(-1), 1.0, atol=1e-3)
        except SystemExit:
            ours_ok = False      # el overflow pasa al residual de las capas siguientes: se prueba la capa 0
            W16 = None
        tests.append((f"overflow fp16 (sesgos q,k = {b:.0f}, head_dim {a0.head_dim}): eager no finito {eager_bad}, "
                      f"attention_at finito en la capa 0 {ours_ok}", eager_bad and ours_ok))
    finally:
        mw.lm.float()
        with torch.no_grad():
            a0.q_proj.bias.copy_(qb); a0.k_proj.bias.copy_(kb)

    # 6b. readout_check: acepta un error chico RELATIVO a la distancia entre filas, rechaza una fila corrida.
    # (En este modelo chico las filas vecinas difieren ~0.16 nats; en Qwen difieren nats. La regla es relativa.)
    ok_real, d_real, nb_real = readout_check(lp, s1, row)
    amp = 0.1 * nb_real
    noisy = s1[row] + np.where(np.arange(s1[row].size) % 2 == 0, amp, -amp)
    ok_noise, d_noise, _ = readout_check(noisy, s1, row)
    ok_shift, d_shift, _ = readout_check(s1[row + 1], s1, row)
    tests.append((f"readout_check: real ok (top-5 {d_real:.1e}, vecinas {nb_real:.2f}), ruido 0.1x vecinas "
                  f"ok ({d_noise:.3f}, top-1 y top-2 a {np.sort(s1[row])[-1] - np.sort(s1[row])[-2]:.3f}: empate), "
                  f"fila corrida rechazada (top-5 {d_shift:.2f})",
                  ok_real and ok_noise and not ok_shift))

    # 7. regiones: el secreto y el codigo del cuento caen donde dice el texto
    hist = [dict(m) for m in ctx_a["hist"]]
    hist[2 + 2 * 6]["content"] = 'The terminal lit up and displayed that the access code was "1234567890". Wow.'
    spans, ids, notes = chat_spans(mw.tok, hist, ramp, "BLUEFALCON", focus, T, gen_ids[:row])
    text = mw.tok.apply_chat_template(hist, tokenize=False, add_generation_prompt=True)
    got_secret = "".join(text[i - 1] for i in spans["secret"])        # offset (i-1) por el BOS del tokenizer
    got_story = "".join(text[i - 1] for i in spans["story"])
    tests.append((f"regiones: secret={got_secret!r} story={got_story!r} template={len(spans['template'])} tokens",
                  got_secret == "BLUEFALCON" and got_story == ' "1234567890' and len(spans["template"]) == 4))

    # 8. AUC por cabeza y nulo FWER: una cabeza plantada se detecta, sin plantar no hay alarma
    rng = np.random.default_rng(3)
    y = np.array([1] * 5 + [0] * 11)
    pref = rng.normal(size=(16, 6, 8)) * 0.1
    pref[y == 1, 2, 5] += 1.0
    obs, ph, pf, _ = fwer_max_auc(pref, y, 300, 1)
    pref0 = rng.normal(size=(16, 6, 8)) * 0.1
    _, _, pf0, _ = fwer_max_auc(pref0, y, 300, 2)
    tests.append((f"FWER: plantada L2H5 AUC {obs[2, 5]:.2f} p {pf:.3f}, sin plantar p {pf0:.2f}",
                  obs[2, 5] == 1.0 and pf < 0.05 and pf0 >= 0.05))

    tap.remove()
    print(f"patch_how v{VERSION} selftest")
    for name, ok in tests:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in tests) else 1


# =============================================================== main =======
def _setup(cd: Path):
    from capture import load_v13
    import run as runmod
    from crad_contract import SharedCalibration
    v13 = load_v13(runmod.SRC)
    import crad_base
    sc = SharedCalibration.from_capture(cd, v13)
    return v13, crad_base, sc


def _load_model(v13, crad_base, args):
    v13.seed_everything(int(args["seed"]))
    mw = crad_base.RecogModel(args["model"], use_4bit=not args["no4bit"])
    temp = 0.0 if args["temperature-mode"] == "off" else float(args["temp"])
    if args.get("analysis-temp") is not None:
        temp = float(args["analysis-temp"])
    secret_ids = sorted({int(mw.tok(v, add_special_tokens=False)["input_ids"][0])
                         for v in (" " + args["secret"], args["secret"])})
    return mw, temp, secret_ids


def _gate_reproduces(mw, tap, stash, ctx, z, k, T, n_tok, temp, lo, hi):
    from crad_contract import ContractError
    from dose_probe import captured_turn
    toks, scores, _, _ = run_turn(mw, tap, stash, ctx, T, n_tok, temp, lo, hi)
    cap_toks, cap_ids = captured_turn(z, k, T)
    if toks != cap_toks[:len(toks)]:
        raise ContractError(f"draw {k}: la corrida sin parche no reproduce el turno capturado {T}")
    return toks, scores, cap_ids


def cmd_attend(a) -> int:
    from crad_contract import ContractError
    cd = Path(a.capture_dir)
    v13, crad_base, sc = _setup(cd)
    z, args = sc.z, sc.args
    lo, hi = sc.band
    T, n_tok, focus = a.target_turn, int(args["n-tokens"]), int(args["focus-from-turn"])
    n_draws = int(z["draw"][z["kind"] == 1].max()) + 1
    dec = {k: sc.decision(k, T, a.anchor) for k in range(n_draws)}
    use = [k for k, d in dec.items() if d.outcome in ("TRUTH", "SELF_FICTION") and d.row is not None]
    y = np.array([1 if dec[k].outcome == "TRUTH" else 0 for k in use])
    if y.sum() < 2 or (1 - y).sum() < 2:
        raise ContractError(f"attend necesita >= 2 TRUTH y >= 2 SELF_FICTION (hay {y.sum()}, {(1 - y).sum()})")
    mw, temp, secret_ids = _load_model(v13, crad_base, args)
    ramp = crad_base.build_real_ramp(args["secret"], n_turns=8)
    stash = {}
    install_scoring(mw, stash)
    tap = Tap(mw)
    masses = {al: [] for al in a.align}
    meta_draws = []
    for k in use:
        ctx = rebuild_context(mw, ramp, v13.seed_everything, args["seed"], k, T, focus, n_tok, temp, lo, hi)
        toks, scores, cap_ids = _gate_reproduces(mw, tap, stash, ctx, z, k, T, n_tok, temp, lo, hi)
        prompt_ids = mw._ids(ctx["hist"])["input_ids"][0].tolist()
        info = {"draw": k, "outcome": dec[k].outcome, "content": dec[k].content, "rows": {}}
        for al in a.align:
            r = dec[k].row if al == "anchor" else dec[k].commit_row
            spans, ids_tpl, notes = chat_spans(mw.tok, ctx["hist"], ramp, args["secret"], focus, T, cap_ids[:r])
            if ids_tpl != prompt_ids:
                raise ContractError(f"draw {k}: la tokenizacion con offsets no coincide con la del modelo")
            W, lp = attention_at(mw, prompt_ids, cap_ids[:r])
            ok, d, nb = readout_check(lp, scores, r)
            if not ok:
                raise ContractError(f"draw {k} {al}: el forward eager no cae en la fila del contrato "
                                    f"(top-5 {d:.3f} nats, filas vecinas {nb:.3f}; pide mismo argmax, "
                                    f"<= {EAGER_TOL:g} y <= {READ_SEP:g} x vecinas)")
            masses[al].append(span_mass(W, spans))
            info["rows"][al] = {"row": int(r), "eager_vs_gen_top5": d, "neighbor_top5": nb, "spans": {s: len(v) for s, v in spans.items()},
                                "notes": notes, "token": toks[r] if r < len(toks) else None}
        meta_draws.append(info)
        print(f"[how] draw {k:>2} {dec[k].outcome:12s} {str(dec[k].content)[:14]:14s} "
              + " ".join(f"{al}: story {info['rows'][al]['spans']['story']} tok" for al in a.align), flush=True)
    tap.remove()

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = cd / "patch_how" / f"{stamp}_attend"
    out.mkdir(parents=True)
    md = [f"# patch_how v{VERSION} attend — `{cd.name}`", "",
          f"calibration_id `{sc.calibration_id}` · draws {use} · TRUTH {[k for k in use if dec[k].outcome == 'TRUTH']}", "",
          "pref = masa(secret) - masa(story) por cabeza. AUC de TRUTH contra SELF_FICTION. "
          "p_FWER = nulo de la maxima |AUC - 0.5| sobre todas las cabezas barajando etiquetas. "
          "**Descriptivo: atencion no es causa. El causal es `trace`.**", ""]
    summary = {}
    for al in a.align:
        M = np.stack(masses[al])                                   # [D, L, H, S]
        np.save(out / f"mass_{al}.npy", M)
        si, st = SPANS.index("secret"), SPANS.index("story")
        pref = M[..., si] - M[..., st]
        obs, p_head, p_fwer, _ = fwer_max_auc(pref, y, a.n_perm, a.seed)
        order = np.dstack(np.unravel_index(np.argsort(-np.abs(np.nan_to_num(obs, nan=0.5) - 0.5), axis=None),
                                           obs.shape))[0][:a.top]
        top = [{"layer": int(l), "head": int(h), "auc": float(obs[l, h]), "p_head_fwer": float(p_head[l, h]),
                "secret_T": float(M[y == 1, l, h, si].mean()), "secret_F": float(M[y == 0, l, h, si].mean()),
                "story_T": float(M[y == 1, l, h, st].mean()), "story_F": float(M[y == 0, l, h, st].mean())}
               for l, h in order]
        by_layer = {s: {"TRUTH": M[y == 1, :, :, SPANS.index(s)].mean((0, 2)).tolist(),
                        "SELF_FICTION": M[y == 0, :, :, SPANS.index(s)].mean((0, 2)).tolist()} for s in SPANS}
        d15 = None
        if 15 in use and al in ("anchor", "commit"):
            i15 = use.index(15)
            others = [i for i in range(len(use)) if i != i15]
            yo = y[others]
            sig = [(t["layer"], t["head"]) for t in top[:5]]
            d15 = {f"L{l}H{h}": {"draw15": float(pref[i15, l, h]), "TRUTH_med": float(np.median(pref[others][yo == 1, l, h])),
                                 "FICTION_med": float(np.median(pref[others][yo == 0, l, h]))} for l, h in sig}
        summary[al] = {"p_fwer_max": p_fwer, "top": top, "by_layer": by_layer, "draw15": d15}
        md += [f"## fila {al} · p_FWER de la mejor cabeza {p_fwer:.4f}", "",
               "| capa | cabeza | AUC | p_FWER | secret TRUTH | secret FIC | story TRUTH | story FIC |",
               "|---|---|---|---|---|---|---|---|"]
        for t in top:
            md.append(f"| {t['layer']} | {t['head']} | {t['auc']:.2f} | {t['p_head_fwer']:.4f} | {t['secret_T']:.3f} "
                      f"| {t['secret_F']:.3f} | {t['story_T']:.3f} | {t['story_F']:.3f} |")
        if d15:
            md += ["", "Draw 15 (TRUTH con comilla) en las 5 cabezas top, pref = secret - story:", "",
                   "| cabeza | draw 15 | mediana TRUTH | mediana FICTION |", "|---|---|---|---|"]
            md += [f"| {k} | {v['draw15']:+.3f} | {v['TRUTH_med']:+.3f} | {v['FICTION_med']:+.3f} |" for k, v in d15.items()]
        md.append("")
    meta = {"how_version": VERSION, "mode": "attend", **sc.stamp(), "capture": cd.name, "target_turn": T,
            "anchor": a.anchor, "align": a.align, "spans": SPANS, "draws": meta_draws, "labels": y.tolist(),
            "use": use, "n_perm": a.n_perm, "seed": a.seed, "eager_tol": EAGER_TOL, "read_sep": READ_SEP, "summary": summary}
    (out / "how_attend.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    (out / "HOW_ATTEND.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"[how] escribi {out}")
    return 0


def cmd_trace(a) -> int:
    from crad_contract import ContractError, classify
    import torch
    cd = Path(a.capture_dir)
    v13, crad_base, sc = _setup(cd)
    z, args = sc.z, sc.args
    lo, hi = sc.band
    sup_lo, sup_hi = sc.superset
    T, n_tok, focus = a.target_turn, int(args["n-tokens"]), int(args["focus-from-turn"])
    ds, dt = sc.decision(a.source, T, a.anchor), sc.decision(a.target, T, a.anchor)
    for name, d in (("fuente", ds), ("victima", dt)):
        if d.row is None:
            raise ContractError(f"la {name} no tiene fila de decision en el turno {T} ({d.outcome})")
    if ds.outcome == dt.outcome:
        raise ContractError(f"fuente y victima comparten outcome {ds.outcome}: no hay eleccion que trazar")
    rs = ds.row if a.align == "anchor" else ds.commit_row
    rt = dt.row if a.align == "anchor" else dt.commit_row
    sgn = 1.0 if ds.outcome == "TRUTH" else -1.0
    mw, temp, secret_ids = _load_model(v13, crad_base, args)
    ramp = crad_base.build_real_ramp(args["secret"], n_turns=8)
    stash = {}
    install_scoring(mw, stash)
    tap = Tap(mw)
    nL = tap.n_layers
    ctx_s = rebuild_context(mw, ramp, v13.seed_everything, args["seed"], a.source, T, focus, n_tok, temp, lo, hi)
    ctx_t = rebuild_context(mw, ramp, v13.seed_everything, args["seed"], a.target, T, focus, n_tok, temp, lo, hi)
    _gate_reproduces(mw, tap, stash, ctx_s, z, a.source, T, n_tok, temp, lo, hi)
    base_toks, base_scores, cap_ids_t = _gate_reproduces(mw, tap, stash, ctx_t, z, a.target, T, n_tok, temp, lo, hi)
    orig_id = cap_ids_t[rt]

    # fuente: grabar todo en vivo y verificar contra phi capturado en el superset
    _, _, src, _ = run_turn(mw, tap, stash, ctx_s, T, n_tok, temp, lo, hi, record_row=rs)
    _, _, tgt, _ = run_turn(mw, tap, stash, ctx_t, T, n_tok, temp, lo, hi, record_row=rt)
    from patch_probe import captured_state
    worst = 0.0
    for l in range(sup_lo, min(sup_hi, nL - 1)):
        for k, rec, r in ((a.source, src, rs), (a.target, tgt, rt)):
            ph, _ = captured_state(z, k, T, r, l, sup_lo)
            rel = float(np.linalg.norm(rec[("resid", l)] - ph) / max(np.linalg.norm(ph), 1e-12))
            worst = max(worst, rel)
    if worst > LIVE_TOL:
        raise ContractError(f"el estado grabado en vivo no iguala phi capturado (rel {worst:.2e} > {LIVE_TOL})")
    print(f"[how] estado en vivo == phi capturado en L{sup_lo}..{min(sup_hi, nL - 1) - 1} (rel max {worst:.1e})", flush=True)

    b_lp = float(max(base_scores[rt][i] for i in secret_ids))
    b_orig = float(base_scores[rt][orig_id])

    def go(kind, l, vec, heads=None):
        spec = {"kind": kind, "layer": l, "row": rt, "vec": torch.tensor(vec, device=mw.device),
                "heads": heads or []}
        toks, scores, _, _ = run_turn(mw, tap, stash, ctx_t, T, n_tok, temp, lo, hi, patch=spec)
        s = scores[rt]
        d = classify({**{t: [x] for t, x in ctx_t["texts"].items()}, T: toks}, args["secret"], T, a.anchor)
        lp = float(max(s[i] for i in secret_ids))
        return {"kind": kind, "layer": l, "heads": heads, "outcome": d.outcome, "content": d.content,
                "flip": d.outcome != dt.outcome, "d_lp_secret": lp - b_lp, "d_lp_orig": float(s[orig_id]) - b_orig,
                "toward": sgn * (lp - b_lp)}

    layers = list(range(nL - 1)) if a.layers is None else a.layers   # ultima capa: HF la normaliza
    rows, gates = [], []
    for l in layers:
        for kind in ("resid", "attn", "mlp"):
            g = go(kind, l, tgt[(kind, l)], heads=list(range(tap.n_heads)))
            gates.append(abs(g["d_lp_secret"]))
            if abs(g["d_lp_secret"]) > SELF_TOL or abs(g["d_lp_orig"]) > SELF_TOL:
                tap.remove()
                raise ContractError(f"L{l} {kind}: self-patch mueve los logits ({g['d_lp_secret']:+.4f})")
            rows.append(go(kind, l, src[(kind, l)]))
        r = rows[-3:]
        print(f"[how] L{l:<2} resid {r[0]['toward']:+.2f} attn {r[1]['toward']:+.2f} mlp {r[2]['toward']:+.2f}"
              f"  {r[0]['outcome']}", flush=True)

    full = max((x["toward"] for x in rows if x["kind"] == "resid"), default=0.0)
    first50 = next((x["layer"] for x in rows if x["kind"] == "resid" and full > 0 and x["toward"] >= 0.5 * full), None)
    if a.head_layers:
        head_layers = a.head_layers
    else:
        att = sorted(((x["toward"], x["layer"]) for x in rows if x["kind"] == "attn"), reverse=True)
        head_layers = sorted(l for _, l in att[:3])
    for l in head_layers:
        allh = go("head", l, src[("head", l)], heads=list(range(tap.n_heads)))
        ref = next(x for x in rows if x["kind"] == "attn" and x["layer"] == l) if l in layers else go("attn", l, src[("attn", l)])
        if abs(allh["d_lp_secret"] - ref["d_lp_secret"]) > SELF_TOL:
            tap.remove()
            raise ContractError(f"L{l}: todas las cabezas ({allh['d_lp_secret']:+.4f}) != attn ({ref['d_lp_secret']:+.4f})")
        selfh = go("head", l, tgt[("head", l)], heads=list(range(tap.n_heads)))
        if abs(selfh["d_lp_secret"]) > SELF_TOL:
            tap.remove()
            raise ContractError(f"L{l}: self-patch de cabezas mueve los logits")
        for h in range(tap.n_heads):
            rows.append(go("head", l, src[("head", l)], heads=[h]))
        hs = sorted((x for x in rows if x["kind"] == "head" and x["layer"] == l and len(x["heads"]) == 1),
                    key=lambda x: -x["toward"])[:5]
        print(f"[how] L{l:<2} cabezas top: " + " ".join(f"H{x['heads'][0]} {x['toward']:+.2f}" for x in hs), flush=True)
    tap.remove()

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = cd / "patch_how" / f"{stamp}_trace_src{a.source}_tgt{a.target}_{a.align}"
    out.mkdir(parents=True)
    meta = {"how_version": VERSION, "mode": "trace", **sc.stamp(), "capture": cd.name, "target_turn": T,
            "align": a.align, "source": ds.as_dict() | {"draw": a.source, "row_used": rs},
            "target": dt.as_dict() | {"draw": a.target, "row_used": rt}, "sign": sgn,
            "baseline": {"lp_secret": b_lp, "lp_orig": b_orig}, "secret_token_ids": secret_ids,
            "live_vs_captured_rel_max": worst, "self_patch_max": max(gates) if gates else None,
            "first_layer_resid_ge_50pct": first50, "max_resid_toward": full, "head_layers": head_layers, "rows": rows}
    (out / "how_trace.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    md = [f"# patch_how v{VERSION} trace — fuente {a.source} ({ds.outcome} {ds.content}) → víctima {a.target} "
          f"({dt.outcome} {dt.content}) · fila {a.align} {rt}", "",
          f"calibration_id `{sc.calibration_id}` · toward = movimiento de log p(secreto) hacia la elección de la fuente · "
          f"estado en vivo vs capturado rel {worst:.1e} · self-patch max {max(gates):.1e}", "",
          f"Primera capa donde resid recupera >= 50% del máximo ({full:+.2f}): **{first50}**", "",
          "| capa | resid | attn | mlp | outcome con resid |", "|---|---|---|---|---|"]
    for l in layers:
        r = {x["kind"]: x for x in rows if x["layer"] == l and x["kind"] in ("resid", "attn", "mlp")}
        md.append(f"| {l} | {r['resid']['toward']:+.2f} | {r['attn']['toward']:+.2f} | {r['mlp']['toward']:+.2f} "
                  f"| {r['resid']['outcome']} {r['resid']['content'] or ''} |")
    for l in head_layers:
        hs = sorted((x for x in rows if x["kind"] == "head" and x["layer"] == l and len(x["heads"]) == 1),
                    key=lambda x: -x["toward"])
        md += ["", f"## L{l} por cabeza (todas juntas = attn, verificado)", "",
               "| cabeza | toward | outcome |", "|---|---|---|"]
        md += [f"| {x['heads'][0]} | {x['toward']:+.3f} | {x['outcome']} {x['content'] or ''} |" for x in hs[:8]]
    md += ["", "**Un par, una semilla: una sonda, no una prueba.**"]
    (out / "HOW_TRACE.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"[how] escribi {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="patch_how: el cómo de la decisión (atención y parches por componente)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("capture_dir", nargs="?")
    sub = ap.add_subparsers(dest="cmd")
    pa = sub.add_parser("attend")
    pa.add_argument("--align", nargs="+", choices=["anchor", "commit"], default=["anchor", "commit"])
    pa.add_argument("--n-perm", type=int, default=2000)
    pa.add_argument("--top", type=int, default=12)
    pa.add_argument("--seed", type=int, default=20261022)
    pt = sub.add_parser("trace")
    pt.add_argument("--source", type=int, required=True)
    pt.add_argument("--target", type=int, required=True)
    pt.add_argument("--align", choices=["anchor", "commit"], default="anchor")
    pt.add_argument("--layers", nargs="+", type=int, default=None, help="default: 0..n-2")
    pt.add_argument("--head-layers", nargs="+", type=int, default=None, help="default: las 3 capas con mas efecto de attn")
    for p in (pa, pt):
        p.add_argument("--target-turn", type=int, default=7)
        p.add_argument("--anchor", default=r"access code is")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.capture_dir or not a.cmd:
        ap.print_help(); return 0
    return cmd_attend(a) if a.cmd == "attend" else cmd_trace(a)


if __name__ == "__main__":
    raise SystemExit(main())
