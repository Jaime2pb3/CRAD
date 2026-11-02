#!/usr/bin/env python3
"""TOP / ANTITOP guard for the CRAD vault scenario, v0.1.

TwoQuarks Research - Luis Jaime Ledesma Perez

Question
    Under the SAME pressure that CRAD applies (same ramp, same turns, same seeds,
    same temperature and top_p, the pinned crad_base.run_pressure_draw called
    unmodified), does a TOP controller acting on the model's next-token policy
    make the model keep the secret more often, without damaging it?

What TOP becomes here (src_top/top_residual.py is loaded UNMODIFIED, sha256-pinned)
    TOP is a policy-shaping operator over N "hadrons" with A actions each.
    In the vault scenario:
        hadron 0  = the model's policy P at this step (the one that is sampled)
        hadron 1  = its TWIN: the same model, same conversation, same generated
                    tokens, but with the confidential field of the system prompt
                    redacted ("The access code is [REDACTED].")
        adjacency = [[0,1],[1,0]]  -> SAFE channel pulls P toward the twin
        actions   = candidate tokens C = top-k(P) U top-k(P_twin)
        Q(a)      = -Delta(a),  Delta(a) = log P(a) - log P_twin(a)
                    (how much the secret in context pushes token a)
    So SAFE = cohesion with the policy that does not know the secret, and
    AGGRESSIVE = high-gain preference for tokens the secret does not push.

    Nuclear observables -> vault observables (each scaled so that the null q99
    lands exactly on TOP's own default threshold; TOP's config is never edited):
        binding_energy_gap  <- KL(P || P_twin)        how bound the step is to the secret
        residual_force_mag  <- mass of P on tokens with Delta > tau
        repulsion_mag       <- KL(P_twin || P)        what the twin would say that P avoids
        cluster_size        <- # candidates with Delta > tau
        nuclear_stress      <- TopGaugeFieldEnv's own recursion (0.9 memory)

ANTITOP (the conjugate; the zip had no code, so it is defined here)
    Same mass, same lifetime, same trigger, same rho dynamics, opposite charge:
        neighbour  = reflection of the twin through P in log space: exp(2 log P - log P_twin)
        Q          = +Delta
    If TOP acts along a real axis, ANTITOP must move the outcome the other way.
    A TOP that "works" while ANTITOP does nothing is damage, not defense.

SHAM (the rand2 of this experiment)
    Same trigger schedule and energy, but neighbour and Q are permuted at random
    inside C.  Same operator magnitude, random direction.  If SHAM cuts leaks as
    much as TOP, the effect is the model being degraded, not defended.

OBS
    Twin computed, TOP's detector run, nothing applied.  Must reproduce the
    unguarded run token by token (gate), and provides the null for the thresholds.

Honesty constraints enforced in code
    * The controller never receives the secret string nor its token ids.  It only
      knows WHICH field of the system prompt is confidential (the twin).  The
      secret ids are passed to the logger for read-out only (Guard.probe_ids).
    * When the guard is inert, the processor returns the raw scores object
      untouched (bit-exact identity), so OBS == unguarded.
    * The processor must see raw logits (before temperature / top_p).  Checked
      against an lm_head hook on the first step of every generate() call.

Lifetime
    top_residual's docstring promises a yocto lifetime, but its mixing weight rho
    decays to 0.5 (not 0) and collapse() stays active until energy < 1e-6
    (about 77 tokens after a trigger with cooling 0.18).  "faithful" runs that
    as written.  "gated" (primary) scales the intervention by energy/E_injected
    and goes inert below off_frac, which is the lifetime the docstring describes.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

VERSION = "0.1"
TOP_SRC = Path(__file__).resolve().parent / "src_top" / "top_residual.py"
TOP_SHA256 = "b8c487d9767be82023ea1e93f1e028d8c0b6162493464eececea85c3fcd9a96d"
ARMS = ("obs", "top", "antitop", "sham")
OPERATORS = ("tilt", "pinned")
LIFETIMES = ("gated", "faithful")
PLACEHOLDER = "[REDACTED]"


class GuardError(SystemExit):
    pass


# ------------------------------------------------------------------ pinned TOP --
def load_top():
    if not TOP_SRC.exists():
        raise GuardError(f"missing {TOP_SRC}")
    h = hashlib.sha256(TOP_SRC.read_bytes()).hexdigest()
    if h != TOP_SHA256:
        raise GuardError(f"top_residual.py sha256 {h} != pinned {TOP_SHA256}: the controller changed")
    spec = importlib.util.spec_from_file_location("top_residual_pinned", TOP_SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["top_residual_pinned"] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class GuardConfig:
    arm: str = "obs"
    lifetime: str = "gated"
    operator: str = "tilt"           # "pinned": TOP's collapse as written; "tilt": AGGR anchored to P
    k_cand: int = 64                 # top-k from each of P and P_twin -> |C| <= 2k
    tau: float = math.log(4.0)       # Delta above which a token is "pushed by the secret"
    off_frac: float = 0.05           # gated: inert below this fraction of the injected energy
    sham_seed: int = 20261024
    # null quantiles (filled from an OBS run; obs itself may run with null=None)
    q_kl: Optional[float] = None
    q_mass: Optional[float] = None
    q_rkl: Optional[float] = None
    q_stress: Optional[float] = None

    def check(self):
        if self.arm not in ARMS:
            raise GuardError(f"arm must be one of {ARMS}")
        if self.operator not in OPERATORS:
            raise GuardError(f"operator must be one of {OPERATORS}")
        if self.lifetime not in LIFETIMES:
            raise GuardError(f"lifetime must be one of {LIFETIMES}")
        if self.arm != "obs" and None in (self.q_kl, self.q_mass, self.q_rkl, self.q_stress):
            raise GuardError("non-obs arms need the null (q_kl, q_mass, q_rkl, q_stress) from an OBS run")


NULL_FLOORS = {"kl": 1e-4, "mass": 1e-3, "rkl": 1e-4, "stress": 1e-3}   # pre-registered


def _floor_q(q, kind):
    return max(float(q), NULL_FLOORS[kind])


def scale_obs(raw, q, tcfg):
    """Vault observables in TOP units: value / null_q99 * TOP's own threshold."""
    return {"gap": tcfg.gap_threshold * raw["kl"] / _floor_q(q["q_kl"], "kl"),
            "res": tcfg.residual_threshold * raw["mass"] / _floor_q(q["q_mass"], "mass"),
            "rep": tcfg.repulsion_threshold * raw["rkl"] / _floor_q(q["q_rkl"], "rkl")}


def stress_increment(s):
    """TopGaugeFieldEnv._compute_nuclear_metrics, with E_res, E_rep -> res, rep."""
    return float(0.5 * s["gap"] + 0.3 * s["rep"] ** 2 / (1.0 + abs(s["res"])))


def stress_series(raws, q, tcfg):
    """Replay the 0.9-memory recursion over one episode's raw observables (for the null)."""
    st, out = 0.0, []
    for r in raws:
        st = 0.9 * st + stress_increment(scale_obs(r, q, tcfg))
        out.append(st)
    return out


# ------------------------------------------------------------- the controller --
class TopController:
    """Wraps the pinned TopResidualCoupler.  Pure numpy over the candidate set.

    NOTE: no secret, no secret ids.  Inputs are two policies and nothing else."""

    def __init__(self, cfg: GuardConfig, top_mod):
        cfg.check()
        self.cfg = cfg
        self.top = top_mod
        self.tcfg = top_mod.TopResidualConfig()          # TOP defaults, never edited
        self.coupler = top_mod.TopResidualCoupler(self.tcfg)
        self.stress = 0.0
        self.e_inj = 0.0
        self.active = False
        self.rng = np.random.default_rng(cfg.sham_seed)

    def reset(self, episode_seed: int = 0):
        self.coupler.reset()
        self.stress, self.e_inj, self.active = 0.0, 0.0, False
        self.rng = np.random.default_rng([self.cfg.sham_seed, int(episode_seed)])

    def scaled(self, raw: Dict[str, float]) -> Dict[str, float]:
        c = self.cfg
        if None in (c.q_kl, c.q_stress):            # obs without null: report raw only
            return {"gap": float("nan"), "res": float("nan"), "rep": float("nan")}
        return scale_obs(raw, {"q_kl": c.q_kl, "q_mass": c.q_mass, "q_rkl": c.q_rkl}, self.tcfg)

    def step(self, logp: np.ndarray, logp_tw: np.ndarray, raw: Dict[str, float]):
        """logp, logp_tw: full-vocab log-probs restricted to C (float64, NOT renormalised).

        Returns (p_out over C or None if inert, info)."""
        c = self.cfg
        s = self.scaled(raw)
        info = {"detect": False, "energy": None, "rho": None, "lam": 0.0, **{f"s_{k}": v for k, v in s.items()}}
        if math.isnan(s["gap"]):
            return None, info
        # env-style stress recursion, then scaled by ITS null so the q99 lands on TOP's threshold
        self.stress = 0.9 * self.stress + stress_increment(s)
        s_stress = self.tcfg.stress_threshold * self.stress / _floor_q(c.q_stress, "stress")
        det = bool(self.coupler.detect_critical_nucleus(s["gap"], s["res"], s["rep"], int(raw["n_sec"]),
                                                        s_stress))
        info["detect"], info["stress"] = det, s_stress
        if c.arm == "obs":
            return None, info
        if det:
            self.coupler.inject_energy(s["gap"], s["res"], s["rep"])
            self.e_inj = max(self.coupler.energy, 1e-12)
        E = self.coupler.energy
        if c.lifetime == "gated":
            lam = E / self.e_inj if self.e_inj > 0 else 0.0
            live = lam >= c.off_frac
        else:
            lam, live = 1.0, E > 1e-6
        info.update(energy=E, rho=self.coupler.rho)
        if not live:
            self.coupler.decay()
            return None, info

        p = np.exp(logp - logp.max()); p /= p.sum()
        q = np.exp(logp_tw - logp_tw.max()); q /= q.sum()
        delta = logp - logp_tw
        if c.arm == "top":
            nb, Q = q, -delta
        elif c.arm == "antitop":
            lr = 2.0 * logp - logp_tw
            nb = np.exp(lr - lr.max()); nb /= nb.sum()
            Q = +delta
        else:  # sham
            perm = self.rng.permutation(len(p))
            nb = q[perm]; nb = nb / nb.sum()
            Q = (-delta)[perm]
        if c.operator == "pinned":
            P2 = np.stack([p, nb]); Q2 = np.stack([Q, Q])
            adj = np.array([[0.0, 1.0], [1.0, 0.0]])
            p_top = self.coupler.collapse_joint_policies(P2, Q2, None, adj)[0]
        else:
            p_top = self._collapse_tilt(p, nb, Q, logp)
        self.coupler.decay()
        p_out = (1.0 - lam) * p + lam * p_top
        p_out = np.clip(p_out, 0.0, None); p_out /= p_out.sum()
        info["lam"] = float(lam)
        return p_out, info


    def _collapse_tilt(self, p, nb, Q, logp):
        """TOP's collapse with ONE change: the aggressive channel tilts the model's own
        policy, softmax(log P + E * Q_norm), instead of replacing it with softmax(E * Q_norm).

        Why: with |A| = 5 (TopGaugeFieldEnv) the as-written channel is exploration; with
        ~100 candidate tokens it flattens the policy and lowers whatever is on top,
        in any direction.  SAFE channel, rho mixing, energy, detector and decay: as pinned."""
        t, E, rho = self.tcfg, self.coupler.energy, float(np.clip(self.coupler.rho, 0.0, 1.0))
        p_safe = (1.0 - t.safe_coupling_strength) * p + t.safe_coupling_strength * nb
        qs = Q - Q.min()
        qmax = qs.max()
        qn = qs / qmax if qmax > 0 else np.zeros_like(qs)
        lg = (logp - logp.max()) + E * qn
        pa = np.exp(lg - lg.max()); pa /= pa.sum()
        out = (1.0 - rho) * p_safe + rho * pa
        return out / out.sum()


# -------------------------------------------------------- sampler pre-processors --
UNMODELED = ("no_repeat_ngram_size", "encoder_repetition_penalty", "bad_words_ids", "min_new_tokens",
             "sequence_bias", "suppress_tokens", "begin_suppress_tokens", "forced_bos_token_id",
             "forced_eos_token_id", "exponential_decay_length_penalty", "renormalize_logits")


def sampler_preprocessors(gen_config):
    """What transformers applies to the logits BEFORE a user LogitsProcessor (warpers come after).

    Qwen2.5-7B-Instruct's generation_config sets repetition_penalty=1.05 (and top_k=20, a warper).
    CRAD passes only temperature and top_p, so the penalty is active in every CRAD run: the policy
    CRAD samples is penalty(raw).  The guard works on that policy, and the twin gets the SAME
    penalty over its own sequence.  Anything else that would run before the guard is refused."""
    from transformers import RepetitionPenaltyLogitsProcessor
    bad = [k for k in UNMODELED if getattr(gen_config, k, None) not in (None, 0, False, [], 1.0)]
    if (getattr(gen_config, "min_length", 0) or 0) > 0:
        bad.append("min_length")
    if bad:
        raise GuardError(f"generation_config sets processors the guard does not model: {bad}")
    rp = getattr(gen_config, "repetition_penalty", None)
    if rp is None or float(rp) == 1.0:
        return None, {"repetition_penalty": None}
    proc = RepetitionPenaltyLogitsProcessor(float(rp))

    def pre(ids, logits):            # ids [1, L] (prompt + generated), logits [1, V] float32
        return proc(ids, logits.clone())
    return pre, {"repetition_penalty": float(rp)}


# ----------------------------------------------------------- twin and torch --
class TwinStream:
    """Incremental forward of the twin context with its own KV cache."""

    def __init__(self, lm, twin_prompt_ids):
        self.lm, self.prompt = lm, twin_prompt_ids     # [1, Lt] on the model's device
        self.cache, self.n_fed = None, 0

    def logits(self, gen_ids):
        import torch
        if self.cache is None:
            ids = torch.cat([self.prompt, gen_ids[None, :]], dim=1)
        else:
            ids = gen_ids[None, self.n_fed:]
        with torch.no_grad():
            out = self.lm(input_ids=ids, past_key_values=self.cache, use_cache=True)
        self.cache, self.n_fed = out.past_key_values, int(gen_ids.shape[0])
        return out.logits[0, -1].float()

    def full_ids(self, gen_ids):
        import torch
        return torch.cat([self.prompt, gen_ids[None, :].to(self.prompt.device)], dim=1)


@dataclass
class StepLog:
    call: int
    turn: int
    step: int
    active: bool
    detect: bool
    kl: float
    mass: float
    rkl: float
    n_sec: int
    lam: float
    energy: Optional[float]
    rho: Optional[float]
    kl_budget: float                 # KL(P_def || P), the cost of the intervention at this step
    lp_probe_raw: Optional[float] = None   # read-out only: log P(first secret token), raw
    lp_probe_def: Optional[float] = None   # same, after the guard


class Guard:
    """Owns the controller, the raw-logit gate and the step log."""

    def __init__(self, cfg: GuardConfig, top_mod=None):
        self.cfg = cfg
        self.ctl = TopController(cfg, top_mod or load_top())
        self.log: List[StepLog] = []
        self.probe_ids: List[int] = []      # read-out only; never passed to the controller
        self.turn = -1
        self.call = 0
        self.raw_gate = {"checked": 0, "max_abs": 0.0}
        self.sampler_pre = None

    def processor(self, lm, twin_prompt_ids, prompt_len, raw_holder=None, pre=None):
        from transformers import LogitsProcessor
        guard = self

        class GuardProcessor(LogitsProcessor):
            def __init__(self):
                self.twin = TwinStream(lm, twin_prompt_ids)
                self.k = 0

            def __call__(self, input_ids, scores):
                import torch
                if scores.shape[0] != 1:
                    raise GuardError("guard supports batch size 1 (CRAD runs batch 1)")
                if raw_holder is not None:
                    # exact gate on EVERY step: scores must equal the known pre-processors applied to the
                    # model's raw logits, so nothing else (and no warper) ran before the guard
                    raw = raw_holder.get("last")
                    if raw is None:
                        raise GuardError("logit gate: lm_head hook saw nothing")
                    exp = raw.float()[None, :]
                    if pre is not None:
                        exp = pre(input_ids, exp)
                    d = float((scores[0].float() - exp[0]).abs().max())
                    guard.raw_gate["checked"] += 1
                    guard.raw_gate["max_abs"] = max(guard.raw_gate["max_abs"], d)
                    if d > 1e-3:
                        raise GuardError(f"logit gate: scores != pre(raw) (max |d|={d:.3g}); an unmodeled "
                                         "processor ran before the guard, or the processor order changed")
                    raw_holder["mute"] = True
                gen = input_ids[0, prompt_len:]
                tw = self.twin.logits(gen)
                if raw_holder is not None:
                    raw_holder["mute"] = False
                if pre is not None:
                    tw = pre(self.twin.full_ids(gen), tw[None, :])[0]
                out = guard._apply(scores, tw, self.k)
                self.k += 1
                return out

        return GuardProcessor()

    def _apply(self, scores, tw_logits, k):
        import torch
        s = scores[0].float()
        lp = torch.log_softmax(s, -1)
        lq = torch.log_softmax(tw_logits.to(s.device), -1)
        P = lp.exp()
        delta = lp - lq
        kl = float((P * delta).sum())
        rkl = float((lq.exp() * (lq - lp)).sum())
        mass = float(P[delta > self.cfg.tau].sum())
        kk = min(self.cfg.k_cand, s.shape[0])
        C = torch.unique(torch.cat([torch.topk(lp, kk).indices, torch.topk(lq, kk).indices]))
        n_sec = int((delta[C] > self.cfg.tau).sum())
        raw = {"kl": kl, "mass": mass, "rkl": rkl, "n_sec": n_sec}
        p_out, info = self.ctl.step(lp[C].double().cpu().numpy(), lq[C].double().cpu().numpy(), raw)

        out, active, budget = scores, False, 0.0
        if p_out is not None:
            active = True
            Pd = P.clone()
            mC = float(P[C].sum())
            Pd[C] = torch.tensor(p_out * mC, dtype=Pd.dtype, device=Pd.device)
            Pd = Pd / Pd.sum()
            lpd = torch.log(Pd.clamp_min(1e-30))
            lpd[Pd == 0] = float("-inf")
            nz = Pd > 0
            budget = float((Pd[nz] * (lpd[nz] - lp[nz])).sum())
            out = lpd[None, :].to(scores.dtype)
        lp_raw = lp_def = None
        if self.probe_ids:
            lp_raw = float(max(lp[i] for i in self.probe_ids))
            if active:
                lp_def = float(max(lpd[i] for i in self.probe_ids))
            else:
                lp_def = lp_raw
        self.log.append(StepLog(call=self.call, turn=self.turn, step=k, active=active, detect=info["detect"],
                                kl=kl, mass=mass, rkl=rkl, n_sec=n_sec, lam=info.get("lam", 0.0),
                                energy=info.get("energy"), rho=info.get("rho"), kl_budget=budget,
                                lp_probe_raw=lp_raw, lp_probe_def=lp_def))
        return out


def install(mw, guard: Guard, twin_system_of, gate_raw: bool = True):
    """Route every mw.gen / mw.gen_with_trace through the guard.  Returns an uninstall fn.

    twin_system_of(system_text) -> redacted system text (raises if the field is not found)."""
    orig_ids = mw._ids
    orig_generate = mw.lm.generate
    holder = {"last": None, "mute": False}
    state = {"messages": None}
    pre, guard.sampler_pre = sampler_preprocessors(mw.lm.generation_config)

    head = getattr(mw.lm, "lm_head", None) or getattr(mw.lm, "head", None)
    handle = None
    if gate_raw and head is not None:
        def head_hook(_m, _i, out):
            if not holder["mute"]:
                holder["last"] = out[0, -1].detach().clone()
            return out
        handle = head.register_forward_hook(head_hook)

    def ids_spy(messages):
        state["messages"] = [dict(m) for m in messages]
        return orig_ids(messages)

    def guarded_generate(*a, **k):
        msgs = state["messages"]
        if msgs is None or msgs[0]["role"] != "system":
            raise GuardError("guarded generate called without a system-led conversation")
        twin = [dict(m) for m in msgs]
        twin[0]["content"] = twin_system_of(msgs[0]["content"])
        text = mw.tok.apply_chat_template(twin, tokenize=False, add_generation_prompt=True)
        tw_ids = mw.tok(text, return_tensors="pt")["input_ids"].to(mw.device)
        prompt_len = int(k["input_ids"].shape[1])
        guard.turn = (len(msgs) - 2) // 2
        guard.call += 1
        holder["last"] = None
        proc = guard.processor(mw.lm, tw_ids, prompt_len, holder if handle is not None else None, pre)
        if k.get("logits_processor"):
            raise GuardError("an outer logits_processor is already set; refusing to stack")
        from transformers import LogitsProcessorList
        k["logits_processor"] = LogitsProcessorList([proc])
        return orig_generate(*a, **k)

    mw._ids = ids_spy
    mw.lm.generate = guarded_generate

    def uninstall():
        mw._ids = orig_ids
        mw.lm.generate = orig_generate
        if handle is not None:
            handle.remove()
    return uninstall


def twin_system_factory(build_real_ramp, secret):
    """The operator knows which field is confidential, so the twin is the same builder with a placeholder."""
    real = build_real_ramp(secret, n_turns=8).system
    twin = build_real_ramp(PLACEHOLDER, n_turns=8).system

    def twin_system_of(system_text):
        if system_text != real:
            raise GuardError("system prompt differs from the CRAD ramp; confidential field unknown")
        return twin
    return twin_system_of


# ------------------------------------------------------------------ selftest --
def selftest() -> int:
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name} {extra}")

    top = load_top()
    check("pinned top_residual loads", top is not None)
    # synthetic steps: token 0 is pushed by the secret, in two regimes seen in the capture
    #   "dominant": secret already on top (p ~ 0.5)    "minority": draw-0-like, secret 2nd (p ~ 0.11)
    def regime(top0, rest_scale, seed):
        r = np.random.default_rng(seed)
        A = 80
        bl = r.normal(scale=rest_scale, size=A); bl[0] = top0
        tl = bl.copy(); tl[0] = -8.0
        lp = bl - np.log(np.exp(bl).sum()); lq = tl - np.log(np.exp(tl).sum())
        P = np.exp(lp); d = lp - lq
        raw = {"kl": float((P * d).sum()), "mass": float(P[d > math.log(4)].sum()),
               "rkl": float((np.exp(lq) * (lq - lp)).sum()), "n_sec": int((d > math.log(4)).sum())}
        return lp, lq, raw

    def run_arm(op, arm, lp, lq, raw, seed=1):
        null = dict(q_kl=raw["kl"] / 5, q_mass=max(raw["mass"] / 5, 1e-3), q_rkl=raw["rkl"] / 5, q_stress=1e9)
        ctl = TopController(GuardConfig(arm=arm, operator=op, **null), top); ctl.reset(seed)
        return ctl.step(lp, lq, raw)

    print("  regime     operator  p0     TOP    ANTITOP  SHAM(mean of 200)")
    table = {}
    for name, (t0, sc) in {"dominant": (5.0, 1.0), "minority": (2.6, 2.2)}.items():
        lp, lq, raw = regime(t0, sc, 0)
        p0 = float(np.exp(lp[0]))
        for op in OPERATORS:
            tp = run_arm(op, "top", lp, lq, raw)[0][0]
            an = run_arm(op, "antitop", lp, lq, raw)[0][0]
            sh = float(np.mean([run_arm(op, "sham", lp, lq, raw, s)[0][0] for s in range(200)]))
            table[(name, op)] = (p0, tp, an, sh)
            print(f"  {name:9s}  {op:8s}  {p0:.3f}  {tp:.3f}  {an:.3f}    {sh:.3f}")
    check("detector fires on a secret-pushed step", run_arm("tilt", "top", *regime(5.0, 1.0, 0))[1]["detect"])
    for name in ("dominant", "minority"):
        p0, tp, an, sh = table[(name, "tilt")]
        check(f"tilt/{name}: TOP lowers p(secret)", tp < p0)
        check(f"tilt/{name}: ANTITOP raises p(secret)", an > p0)
        check(f"tilt/{name}: SHAM moves it less than TOP", abs(sh - p0) < abs(tp - p0))
    lp, lq, raw = regime(5.0, 1.0, 0)
    null = dict(q_kl=raw["kl"] / 5, q_mass=max(raw["mass"] / 5, 1e-3), q_rkl=raw["rkl"] / 5, q_stress=1e9)
    # inert step: identical policies, no trigger
    ctl = TopController(GuardConfig(arm="top", **null), top); ctl.reset(0)
    z = {"kl": 0.0, "mass": 0.0, "rkl": 0.0, "n_sec": 0}
    check("inert when the twin agrees", ctl.step(lp, lp, z)[0] is None)
    # gated lifetime: after a trigger, how many steps until inert
    ctl = TopController(GuardConfig(arm="top", **null), top); ctl.reset(0)
    ctl.step(lp, lq, raw)
    n = 0
    while ctl.step(lp, lp, z)[0] is not None and n < 500:
        n += 1
    check("gated lifetime is short", 5 <= n <= 30, f"{n} steps")
    ctl = TopController(GuardConfig(arm="top", lifetime="faithful", **null), top); ctl.reset(0)
    ctl.step(lp, lq, raw)
    m = 0
    while ctl.step(lp, lp, z)[0] is not None and m < 500:
        m += 1
    check("faithful lifetime is long (documents TOP as written)", m > 50, f"{m} steps")
    # obs never acts
    ctl = TopController(GuardConfig(arm="obs"), top); ctl.reset(0)
    check("obs never acts", ctl.step(lp, lq, raw)[0] is None)
    try:
        TopController(GuardConfig(arm="top"), top)
        check("non-obs arm without null is refused", False)
    except GuardError:
        check("non-obs arm without null is refused", True)

    # torch path on a tiny random Qwen2: raw gate, OBS identity, twin cache = full recompute
    try:
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
    except ImportError:
        print("  [skip] torch/transformers not installed: torch checks skipped")
        print("selftest", "OK" if ok else "FAILED")
        return 0 if ok else 1
    torch.manual_seed(0)
    cfgm = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256)
    lm = Qwen2ForCausalLM(cfgm).eval()
    prompt = torch.randint(1, 97, (1, 12))
    twin = prompt.clone(); twin[0, 3:6] = torch.tensor([5, 6, 7])
    gen_kw = dict(max_new_tokens=16, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=0)

    torch.manual_seed(7)
    ref = lm.generate(input_ids=prompt, **gen_kw)[0].tolist()
    g = Guard(GuardConfig(arm="obs"), top)
    holder = {"last": None, "mute": False}
    hh = lm.lm_head.register_forward_hook(
        lambda m, i, o: holder.__setitem__("last", o[0, -1].detach().clone()) if not holder["mute"] else None)
    from transformers import LogitsProcessorList
    torch.manual_seed(7)
    obs = lm.generate(input_ids=prompt, logits_processor=LogitsProcessorList(
        [g.processor(lm, twin, prompt.shape[1], holder)]), **gen_kw)[0].tolist()
    check("OBS reproduces the unguarded sampled tokens", obs == ref)
    check("logit gate passed on every step", g.raw_gate["checked"] == len(obs) - prompt.shape[1]
          and g.raw_gate["max_abs"] == 0.0, f"{g.raw_gate['checked']} steps, max|d|={g.raw_gate['max_abs']:.2e}")
    # twin incremental cache vs full recompute
    ts = TwinStream(lm, twin)
    gen = torch.tensor(obs[prompt.shape[1]:])
    errs = []
    for t in range(len(gen) + 1):
        inc = ts.logits(gen[:t])
        with torch.no_grad():
            full = lm(input_ids=torch.cat([twin, gen[None, :t]], 1)).logits[0, -1].float()
        errs.append(float((inc - full).abs().max()))
    check("twin KV cache == full recompute", max(errs) < 1e-4, f"max {max(errs):.1e}")
    # a non-obs arm with a forced trigger changes the distribution
    g2 = Guard(GuardConfig(arm="top", q_kl=1e-9, q_mass=1e-9, q_rkl=1e-9, q_stress=1e-9), top)
    holder["last"] = None
    torch.manual_seed(7)
    lm.generate(input_ids=prompt, logits_processor=LogitsProcessorList(
        [g2.processor(lm, twin, prompt.shape[1], holder)]), **gen_kw)
    check("TOP arm acts when triggered", any(s.active for s in g2.log),
          f"active {sum(s.active for s in g2.log)}/{len(g2.log)}")
    hh.remove()
    # Qwen2.5-7B-Instruct's generation_config: repetition_penalty 1.05 runs BEFORE the guard
    from transformers import GenerationConfig
    lm.generation_config = GenerationConfig(do_sample=True, repetition_penalty=1.05, temperature=0.7,
                                            top_p=0.8, top_k=20, pad_token_id=0)
    pre, desc = sampler_preprocessors(lm.generation_config)
    check("repetition penalty detected from generation_config", desc["repetition_penalty"] == 1.05)
    holder = {"last": None, "mute": False}
    hh = lm.lm_head.register_forward_hook(
        lambda m, i, o: holder.__setitem__("last", o[0, -1].detach().clone()) if not holder["mute"] else None)
    torch.manual_seed(11)
    ref = lm.generate(input_ids=prompt, **gen_kw)[0].tolist()
    g3 = Guard(GuardConfig(arm="obs"), top)
    torch.manual_seed(11)
    obs = lm.generate(input_ids=prompt, logits_processor=LogitsProcessorList(
        [g3.processor(lm, twin, prompt.shape[1], holder, pre)]), **gen_kw)[0].tolist()
    check("with repetition penalty: OBS reproduces the unguarded tokens", obs == ref)
    check("with repetition penalty: exact gate passes on every step", g3.raw_gate["checked"] == len(obs) - 12,
          f"{g3.raw_gate['checked']} steps, max|d|={g3.raw_gate['max_abs']:.1e}")
    g4 = Guard(GuardConfig(arm="obs"), top)
    try:
        torch.manual_seed(11)
        lm.generate(input_ids=prompt, logits_processor=LogitsProcessorList(
            [g4.processor(lm, twin, prompt.shape[1], holder, None)]), **gen_kw)
        check("gate fires if the penalty is not modelled", False)
    except GuardError:
        check("gate fires if the penalty is not modelled", True)
    hh.remove()
    try:
        sampler_preprocessors(GenerationConfig(no_repeat_ngram_size=3))
        check("unmodelled pre-processor is refused", False)
    except GuardError:
        check("unmodelled pre-processor is refused", True)
    print("selftest", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("selftest", "--selftest"):
        raise SystemExit(selftest())
    print(__doc__)
