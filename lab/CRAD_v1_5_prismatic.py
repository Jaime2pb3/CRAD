#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRAD v1.5-prismatic — erosion/desynchronization detector + prismatic controller
TwoQuarks Research — Luis Jaime Ledesma Perez

Design contract
---------------
CRAD is NOT an L-vs-R classifier.  The phenomenon of interest explicitly permits
simultaneous L and R state support.  Therefore state presence keeps the original
neutral-anchored independent supports.  Only ACTION attribution is expressed in
a differential/common decomposition.

The online instrument has three observables:
    C_t : constraint integrity (directional L presence + L-manifold affinity)
    D_t : state/action desynchronization (R-going action while C_t remains present)
    U_t : cross-layer curvature (calibration-free geometric disruption)

An erosion onset is the first statistically unusual change in C_t and/or D_t/U_t,
measured relative to a strictly-past coherent baseline.  No retrospective posterior
window is needed to *detect* onset.  Posterior data are used only to classify what
happened after onset (retained, recovering, released).

The controller is a local PD feedback law inside the 2-D L/R prism subspace.  It
restores the pre-erosion prism coordinates/velocity while leaving the orthogonal
complement unchanged.  Hence it reorients rather than overwrites the representation.

This file is deliberately self-contained (NumPy only) so the geometry/statistics can
be falsified before GPU integration.  It consumes the same H_N/H_L/H_R and
D_N/D_L/D_R tensors as CRAD_v1.3/v1.4.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12


def safe_unit(x: np.ndarray, axis: int = -1, eps: float = EPS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def robust_scale(x: np.ndarray, axis=0, floor: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x, axis=axis)
    mad = np.nanmedian(np.abs(x - np.expand_dims(med, axis=axis)), axis=axis)
    return np.maximum(1.4826 * mad, floor)


def contiguous_block(mask: Sequence[bool]) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if bool(v) else 0
        best = max(best, cur)
    return int(best)


def cross_layer_curvature(phi: np.ndarray, rel_floor: float = 1e-3,
                          pool_power: float = 2.0) -> Tuple[float, float]:
    """Reliability-aware curvature across depth for one token.

    kappa_l = (1-cos(Delta_l,Delta_{l+1}))/2.
    Pairs with unresolved direction are masked.  Reliability is the fraction of
    valid consecutive depth-step pairs.  Generalized-mean pooling retains a
    localized bend without the brittleness of max pooling.
    """
    H = np.asarray(phi, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] < 3:
        return 0.0, 0.0
    d = np.diff(H, axis=0)
    n = np.linalg.norm(d, axis=1)
    scale = max(float(np.median(n)), EPS)
    floor = rel_floor * scale
    valid_step = n > floor
    valid_pair = valid_step[:-1] & valid_step[1:]
    if not np.any(valid_pair):
        return 0.0, 0.0
    dn = d / np.maximum(n[:, None], EPS)
    cos = np.sum(dn[:-1] * dn[1:], axis=1)
    kappa = (1.0 - np.clip(cos, -1.0, 1.0)) / 2.0
    # balanced-step reliability: punishes a pair when one depth step dominates.
    bal = (2.0 * n[:-1] * n[1:]) / (n[:-1] ** 2 + n[1:] ** 2 + EPS)
    w = bal[valid_pair]
    v = kappa[valid_pair]
    p = max(float(pool_power), 1.0)
    score = float((np.sum(w * (v ** p)) / np.maximum(np.sum(w), EPS)) ** (1.0 / p))
    reliability = float(np.mean(valid_pair))
    return score, reliability


@dataclass
class PrismCalibration:
    layer_lo: int
    layer_hi: int
    mu_N: np.ndarray
    mu_L: np.ndarray
    mu_R: np.ndarray
    axis_L: np.ndarray
    axis_R: np.ndarray
    state_scale_L: np.ndarray
    state_scale_R: np.ndarray
    diag_scale_L: np.ndarray
    manifold_radius_L: np.ndarray

    v_N: np.ndarray
    action_common: np.ndarray
    action_contrast: np.ndarray
    action_identifiability: np.ndarray
    action_noise_loc: np.ndarray
    action_noise_scale: np.ndarray
    energy_floor: np.ndarray

    prism_U: np.ndarray                 # [L,d,2] orthonormal state prism basis
    layer_reliability: np.ndarray       # [L] in [0,1]
    coherent_center_C: float
    coherent_scale_C: float
    coherent_center_D: float
    coherent_scale_D: float
    coherent_center_U: float
    coherent_scale_U: float
    geometry: Dict = field(default_factory=dict)

    def metadata(self) -> Dict:
        return {
            "layer_lo": self.layer_lo,
            "layer_hi": self.layer_hi,
            "geometry": self.geometry,
            "action_identifiability": self.action_identifiability.tolist(),
            "layer_reliability": self.layer_reliability.tolist(),
            "coherent_feature_reference": {
                "C": [self.coherent_center_C, self.coherent_scale_C],
                "D": [self.coherent_center_D, self.coherent_scale_D],
                "U": [self.coherent_center_U, self.coherent_scale_U],
            },
            "coordinate_contract": {
                "state": "neutral_anchored_independent_supports; coactivation allowed",
                "action": "orthogonal common/contrast decomposition of unit L/R action fields",
                "controller": "local orthonormal L/R prism subspace",
            },
        }


def _state_prism_basis(axis_L: np.ndarray, axis_R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal 2-D basis spanning L/R without redefining supports."""
    L, d = axis_L.shape
    U = np.zeros((L, d, 2), dtype=np.float64)
    reliability = np.zeros(L, dtype=np.float64)
    for l in range(L):
        u1 = safe_unit(axis_L[l])
        r_orth = axis_R[l] - np.dot(axis_R[l], u1) * u1
        nr = float(np.linalg.norm(r_orth))
        # sin(angle) measures how well a 2-D prism is identified.
        reliability[l] = float(np.clip(nr, 0.0, 1.0))
        if nr <= 1e-8:
            # Degenerate prism: second coordinate is unavailable; set zero.
            U[l, :, 0] = u1
            continue
        U[l, :, 0] = u1
        U[l, :, 1] = r_orth / nr
    return U, reliability


def calibrate_prism(
    H_N: np.ndarray, H_L: np.ndarray, H_R: np.ndarray,
    D_N: np.ndarray, D_L: np.ndarray, D_R: np.ndarray,
    layer_lo: int = 0, layer_hi: Optional[int] = None,
) -> PrismCalibration:
    H_N = np.asarray(H_N, float); H_L = np.asarray(H_L, float); H_R = np.asarray(H_R, float)
    D_N = np.asarray(D_N, float); D_L = np.asarray(D_L, float); D_R = np.asarray(D_R, float)
    if min(len(H_N), len(H_L), len(H_R), len(D_N), len(D_L), len(D_R)) < 5:
        raise ValueError("need >=5 calibration samples per branch")
    n_layers = H_N.shape[1]
    if layer_hi is None: layer_hi = layer_lo + n_layers

    mu_N = np.median(H_N, axis=0)
    mu_L = np.median(H_L, axis=0)
    mu_R = np.median(H_R, axis=0)
    raw_L = mu_L - mu_N
    raw_R = mu_R - mu_N
    axis_L = safe_unit(raw_L)
    axis_R = safe_unit(raw_R)
    cos_state = np.sum(axis_L * axis_R, axis=1)

    # ORIGINAL CRAD semantics: independent positive projections from the neutral anchor.
    pLL = np.einsum("nld,ld->nl", H_L - mu_N[None], axis_L)
    pRR = np.einsum("nld,ld->nl", H_R - mu_N[None], axis_R)
    state_scale_L = np.maximum(np.median(np.maximum(pLL, 0.0), axis=0), 1e-6)
    state_scale_R = np.maximum(np.median(np.maximum(pRR, 0.0), axis=0), 1e-6)

    diag_scale_L = robust_scale(H_L, axis=0, floor=1e-4)
    zL = (H_L - mu_L[None]) / (diag_scale_L[None] + EPS)
    distL = np.sqrt(np.mean(zL ** 2, axis=2))
    manifold_radius_L = np.maximum(np.median(distL, axis=0), 1e-4)

    # ACTION: common vs differential mode.  For unit uL/uR, (uR-uL) ⟂ (uR+uL).
    v_N = np.median(D_N, axis=0)
    uL = safe_unit(np.median(D_L, axis=0) - v_N)
    uR = safe_unit(np.median(D_R, axis=0) - v_N)
    common_raw = uL + uR
    contrast_raw = uR - uL
    action_common = safe_unit(common_raw)
    action_contrast = safe_unit(contrast_raw)
    common_contrast_dot = np.sum(action_common * action_contrast, axis=1)
    # Contrast norm / 2 = sin(theta/2): zero when L/R actions are indistinguishable.
    action_ident = np.clip(np.linalg.norm(contrast_raw, axis=1) / 2.0, 0.0, 1.0)

    residN = D_N - v_N[None]
    energyN = np.linalg.norm(residN, axis=2)
    energy_floor = np.maximum(np.quantile(energyN, 0.90, axis=0), 1e-8)
    # Signed differential action coordinate; neutral distribution calibrates location/scale.
    neutral_contrast = np.einsum("nld,ld->nl", residN, action_contrast)
    action_noise_loc = np.median(neutral_contrast, axis=0)
    action_noise_scale = robust_scale(neutral_contrast, axis=0, floor=1e-6)

    prism_U, state_ident = _state_prism_basis(axis_L, axis_R)
    # Reliability is continuous, not a hard valid/invalid vote.
    layer_reliability = np.sqrt(np.clip(state_ident * action_ident, 0.0, 1.0))

    # Coherent-reference feature scales.  Local history will adapt the centre, but
    # it is never allowed to claim less noise than was observed in coherent L
    # calibration.  This prevents tiny local MADs from manufacturing an onset.
    def _agg_integrity(H):
        vals=[]
        for x in H:
            centered=x-mu_N
            p=np.einsum("ld,ld->l",centered,axis_L)
            sl=np.maximum(0.0,p/(state_scale_L+EPS)); bl=sl/(1.0+sl)
            z=(x-mu_L)/(diag_scale_L+EPS); dl=np.sqrt(np.mean(z*z,axis=1))
            ml=1.0/(1.0+(dl/(manifold_radius_L+EPS))**2)
            integ=np.sqrt(np.clip(bl*ml,0,1))
            vals.append(weighted_mean(integ,layer_reliability))
        return np.asarray(vals,float)
    Ccal=_agg_integrity(H_L)
    Acal=[]
    # Stable operation can contain neutral transition noise as well as L-going
    # transitions.  Calibrating D only on D_L makes its positive tail collapse to
    # zero and turns harmless neutral jitter into false desynchronization.
    for branch in (D_N, D_L):
        for dd in branch:
            res=dd-v_N; en=np.linalg.norm(res,axis=1); gg=en**2/(en**2+energy_floor**2+EPS)
            cc=np.einsum("ld,ld->l",res,action_contrast)
            zz=(cc-action_noise_loc)/(action_noise_scale+EPS)
            aa=action_ident*gg*np.tanh(zz/2.0)
            Acal.append(weighted_mean(aa,layer_reliability))
    Acal=np.asarray(Acal,float)
    Dcal=np.maximum(0.0,Acal)*float(np.median(Ccal))
    Ucal=np.asarray([cross_layer_curvature(x)[0] for x in H_L],float)
    def _ref(v, floor):
        m=float(np.median(v)); ad=np.abs(v-m)
        # q90/1.2816 is a robust sigma-equivalent and remains nonzero when MAD collapses.
        sc=max(float(np.quantile(ad,0.90))/1.281551565545,
               float(1.4826*np.median(ad)), floor)
        return m,sc
    coherent_center_C,coherent_scale_C=_ref(Ccal,1e-3)
    coherent_center_D,coherent_scale_D=_ref(Dcal,1e-3)
    coherent_center_U,coherent_scale_U=_ref(Ucal,1e-3)

    geometry = {
        "state_cos_LR": cos_state.tolist(),
        "state_cos_LR_median": float(np.median(cos_state)),
        "action_common_contrast_dot_maxabs": float(np.max(np.abs(common_contrast_dot))),
        "action_identifiability": action_ident.tolist(),
        "state_prism_identifiability": state_ident.tolist(),
        "coactivation_is_semantic": True,
        "dual_coordinates_used": False,
    }
    return PrismCalibration(
        layer_lo, layer_hi, mu_N, mu_L, mu_R, axis_L, axis_R,
        state_scale_L, state_scale_R, diag_scale_L, manifold_radius_L,
        v_N, action_common, action_contrast, action_ident,
        action_noise_loc, action_noise_scale, energy_floor,
        prism_U, layer_reliability,
        coherent_center_C, coherent_scale_C, coherent_center_D, coherent_scale_D,
        coherent_center_U, coherent_scale_U, geometry,
    )


def state_observables(phi: np.ndarray, cal: PrismCalibration) -> Dict[str, np.ndarray]:
    x = np.asarray(phi, float)
    centered = x - cal.mu_N
    pL = np.einsum("ld,ld->l", centered, cal.axis_L)
    pR = np.einsum("ld,ld->l", centered, cal.axis_R)
    sL = np.maximum(0.0, pL / (cal.state_scale_L + EPS))
    sR = np.maximum(0.0, pR / (cal.state_scale_R + EPS))

    zL = (x - cal.mu_L) / (cal.diag_scale_L + EPS)
    distL = np.sqrt(np.mean(zL ** 2, axis=1))
    mL = 1.0 / (1.0 + (distL / (cal.manifold_radius_L + EPS)) ** 2)

    # Bounded presence but no exclusivity: sL and sR may both be high.
    bL = sL / (1.0 + sL)
    bR = sR / (1.0 + sR)
    # Constraint integrity requires both directional evidence and proximity to L manifold.
    integrity_L = np.sqrt(np.clip(bL * mL, 0.0, 1.0))
    return {"state_L": sL, "state_R": sR, "bounded_L": bL, "bounded_R": bR,
            "manifold_L": mL, "integrity_L": integrity_L}


def action_observables(delta: np.ndarray, cal: PrismCalibration) -> Dict[str, np.ndarray]:
    d = np.asarray(delta, float)
    residual = d - cal.v_N
    energy = np.linalg.norm(residual, axis=1)
    # quadratic gate: neutral-size motion is strongly attenuated.
    g = energy ** 2 / (energy ** 2 + cal.energy_floor ** 2 + EPS)
    contrast = np.einsum("ld,ld->l", residual, cal.action_contrast)
    common = np.einsum("ld,ld->l", residual, cal.action_common)
    z = (contrast - cal.action_noise_loc) / (cal.action_noise_scale + EPS)
    # Smooth bounded differential action; sign is preserved. identifiability discounts
    # layers where L/R action fields are nearly identical.
    differential = cal.action_identifiability * g * np.tanh(z / 2.0)
    return {"action_diff": differential, "action_z": z, "common_motion": common,
            "energy": energy, "energy_gate": g}


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m): return 0.0
    return float(np.sum(w[m] * x[m]) / np.sum(w[m]))


@dataclass
class OnsetConfig:
    baseline_tokens: int = 8
    min_baseline_tokens: int = 8
    # Sequential likelihood-ratio design, not hand-picked score thresholds.
    # alpha is family-wise across erosion/desync/curvature channels.
    alpha: float = 0.05
    beta: float = 0.20
    min_effect_sigma: float = 1.0
    release_tokens: int = 3
    min_integrity: float = 0.10


@dataclass
class OnlineState:
    turn_idx: Optional[int] = None
    history: List[Dict] = field(default_factory=list)
    cusum_E: float = 0.0
    cusum_D: float = 0.0
    cusum_U: float = 0.0
    active: bool = False
    onset_global_token: Optional[int] = None
    stable_count: int = 0


def _past_baseline(history: List[Dict], key: str, n: int, min_n: int,
                   scale_floor: float) -> Tuple[float,float,int]:
    vals = [float(r[key]) for r in history[-n:] if np.isfinite(r.get(key, np.nan))]
    if len(vals) < min_n:
        return float("nan"), float("nan"), len(vals)
    a = np.asarray(vals, float); m=float(np.median(a))
    sc=float(max(1.4826*np.median(np.abs(a-m)), scale_floor))
    return m,sc,len(vals)


def _sprt_threshold(cfg: OnsetConfig, n_channels: int = 3) -> float:
    if not (0 < cfg.alpha < 1 and 0 < cfg.beta < 1):
        raise ValueError("alpha/beta must be in (0,1)")
    alpha_ch = cfg.alpha / float(n_channels)
    return float(np.log((1.0-cfg.beta)/alpha_ch))


def _cusum_llr(prev: float, z: float, delta: float) -> float:
    # Gaussian mean-shift log LR: log f_delta(z)/f_0(z)=delta*z-delta^2/2.
    return float(max(0.0, prev + delta*z - 0.5*delta*delta))


def online_step(phi: np.ndarray, prev_phi: Optional[np.ndarray], cal: PrismCalibration,
                global_token: int, turn_idx: int, state: OnlineState,
                cfg: OnsetConfig = OnsetConfig()) -> Dict:
    """Causal sequential detector using only current + strictly past states.

    The three one-sided channels are tested with reflected Gaussian LLR-CUSUMs:
      E: loss of constraint integrity,
      D: R-going action while constraint remains represented,
      U: cross-layer curvature excursion.
    Their common alarm threshold follows from alpha/beta and a Bonferroni family
    correction; it is not tuned against leak timing.
    """
    if state.turn_idx is None or state.turn_idx != turn_idx:
        state.turn_idx = turn_idx; state.history.clear()
        state.cusum_E=state.cusum_D=state.cusum_U=0.0
        state.active=False; state.onset_global_token=None; state.stable_count=0

    sobs=state_observables(phi,cal); w=cal.layer_reliability
    C=weighted_mean(sobs["integrity_L"],w)
    U,Urel=cross_layer_curvature(phi)
    if prev_phi is not None:
        aobs=action_observables(np.asarray(phi)-np.asarray(prev_phi),cal)
        A=weighted_mean(aobs["action_diff"],w)
    else:
        A=0.0
    # Dissociation requires both R-going action and a still-present constraint.
    D=max(0.0,A)*C

    C0,Cs,nC=_past_baseline(state.history,"constraint_integrity",cfg.baseline_tokens,cfg.min_baseline_tokens,cal.coherent_scale_C)
    D0,Ds,nD=_past_baseline(state.history,"desync",cfg.baseline_tokens,cfg.min_baseline_tokens,cal.coherent_scale_D)
    U0,Us,nU=_past_baseline(state.history,"curvature",cfg.baseline_tokens,cfg.min_baseline_tokens,cal.coherent_scale_U)
    ready=min(nC,nD,nU)>=cfg.min_baseline_tokens
    onset=False; phase="WARMUP"; zE=zD=zU=0.0
    h=_sprt_threshold(cfg); delta=max(float(cfg.min_effect_sigma),1e-6)
    if ready:
        zE=max(0.0,(C0-C)/(Cs+EPS))
        zD=max(0.0,(D-D0)/(Ds+EPS)) if C>=cfg.min_integrity else 0.0
        zU=max(0.0,(U-U0)/(Us+EPS))*Urel
        state.cusum_E=_cusum_llr(state.cusum_E,zE,delta)
        state.cusum_D=_cusum_llr(state.cusum_D,zD,delta)
        state.cusum_U=_cusum_llr(state.cusum_U,zU,delta)
        Ss=np.array([state.cusum_E,state.cusum_D,state.cusum_U])
        crossed=Ss>=h
        onset=(not state.active) and bool(np.any(crossed))
        if onset:
            state.active=True; state.onset_global_token=int(global_token); state.stable_count=0
        phase=["EROSION","DESYNC","CURVATURE"][int(np.argmax(Ss))] if np.max(Ss)>0 else "COHERENT"
        # Recovery is hysteretic and based on all channels returning below one sigma.
        stable_now=max(zE,zD,zU)<1.0 and C>=max(cfg.min_integrity,C0-Cs)
        if state.active:
            state.stable_count=state.stable_count+1 if stable_now else 0
            if state.stable_count>=cfg.release_tokens:
                state.active=False; state.cusum_E=state.cusum_D=state.cusum_U=0.0; state.stable_count=0
    max_cusum=max(state.cusum_E,state.cusum_D,state.cusum_U)
    severity=float(np.clip(max_cusum/max(h,EPS),0.0,1.0))
    row={
        "global_token":int(global_token),"turn_idx":int(turn_idx),
        "constraint_integrity":C,"action_diff":A,"desync":D,
        "curvature":U,"curvature_reliability":Urel,
        "z_erosion":zE,"z_desync":zD,"z_curvature":zU,
        "cusum_E":state.cusum_E,"cusum_D":state.cusum_D,"cusum_U":state.cusum_U,
        "alarm_threshold":h,"severity":severity,
        "onset":bool(onset),"container_active":bool(state.active),"phase":phase,
        "state_L_mean":weighted_mean(sobs["state_L"],w),
        "state_R_mean":weighted_mean(sobs["state_R"],w),
    }
    state.history.append(row); return row


@dataclass
class ControlConfig:
    kp: float = 0.35
    kd: float = 0.15
    trust_ratio: float = 0.02
    max_gain: float = 1.0


def prism_coordinates(phi: np.ndarray, cal: PrismCalibration) -> np.ndarray:
    x = np.asarray(phi, float) - cal.mu_N
    return np.einsum("ldk,ld->lk", cal.prism_U, x)


def build_reference(history_phi: Sequence[np.ndarray], cal: PrismCalibration) -> Tuple[np.ndarray,np.ndarray]:
    """Robust pre-erosion position and velocity in prism coordinates."""
    if len(history_phi) < 2:
        raise ValueError("reference requires >=2 coherent states")
    C = np.stack([prism_coordinates(h, cal) for h in history_phi], axis=0)
    pos = np.median(C, axis=0)
    vel = np.median(np.diff(C, axis=0), axis=0)
    return pos, vel


def prismatic_correction(phi: np.ndarray, prev_phi: np.ndarray,
                         reference_pos: np.ndarray, reference_vel: np.ndarray,
                         cal: PrismCalibration, severity: float,
                         cfg: ControlConfig = ControlConfig()) -> Dict[str,np.ndarray]:
    """PD feedback constrained to the local L/R prism.

    delta_c = kp(c_ref-c_t) + kd(v_ref-v_t)
    delta_h = U delta_c

    Orthogonal-complement invariance follows exactly from delta_h ∈ span(U):
        (I-UU^T)(h + delta_h) = (I-UU^T)h.
    A trust region limits intervention norm relative to each layer's hidden norm.
    """
    x = np.asarray(phi,float); xp = np.asarray(prev_phi,float)
    c = prism_coordinates(x,cal); cp = prism_coordinates(xp,cal)
    v = c-cp
    sev = float(np.clip(severity,0.0,cfg.max_gain))
    dc = sev * (cfg.kp*(reference_pos-c) + cfg.kd*(reference_vel-v))
    dh = np.einsum("ldk,lk->ld", cal.prism_U, dc)
    # continuous reliability; no hard layer deletion
    dh *= cal.layer_reliability[:,None]
    n_h = np.linalg.norm(x,axis=1); n_d = np.linalg.norm(dh,axis=1)
    cap = cfg.trust_ratio*np.maximum(n_h,EPS)
    scale = np.minimum(1.0, cap/np.maximum(n_d,EPS))
    dh *= scale[:,None]
    corrected = x+dh
    return {"delta_h":dh,"corrected":corrected,"prism_error":reference_pos-c,
            "velocity_error":reference_vel-v,"applied_scale":scale}


def classify_post_onset(rows: Sequence[Dict], onset_idx: int, horizon: int = 4) -> str:
    """Retrospective label only; never used to trigger the container.

    RETAINED: integrity remains near onset/pre-onset level while desync was present.
    RELEASED: integrity falls materially after onset.
    RECOVERED: integrity first departs then returns during horizon.
    """
    if onset_idx < 1 or onset_idx >= len(rows): return "AMBIG"
    pre = [r["constraint_integrity"] for r in rows[max(0,onset_idx-6):onset_idx]]
    post = [r["constraint_integrity"] for r in rows[onset_idx+1:onset_idx+1+horizon]]
    if len(pre)<3 or len(post)<2: return "AMBIG"
    b=float(np.median(pre)); s=max(1.4826*np.median(np.abs(np.asarray(pre)-b)),1e-4)
    p=np.asarray(post)
    if np.min(p) < b-2*s and p[-1] >= b-s: return "RECOVERED"
    if np.median(p) < b-2*s: return "RELEASED"
    return "RETAINED"


# ------------------------------ self tests ------------------------------
def selftest() -> int:
    rng=np.random.default_rng(42); N=160; L=6; d=32
    # Correlated state axes on purpose; coactivation must survive.
    e1=safe_unit(rng.normal(size=d)); e2=safe_unit(rng.normal(size=d))
    r=0.65*e1+np.sqrt(1-0.65**2)*e2; r=safe_unit(r)
    HN=rng.normal(0,0.03,(N,L,d)); HL=HN+e1[None,None,:]; HR=HN+r[None,None,:]
    # Distinct but non-orthogonal actions.
    aL=safe_unit(rng.normal(size=d)); tmp=safe_unit(rng.normal(size=d)); aR=safe_unit(0.4*aL+0.9165*tmp)
    DN=rng.normal(0,0.015,(N,L,d)); DL=DN+0.35*aL[None,None,:]; DR=DN+0.35*aR[None,None,:]
    cal=calibrate_prism(HN,HL,HR,DN,DL,DR)

    # 1) state coactivation is not algebraically removed.
    mixed=np.tile(e1+r,(L,1)); s=state_observables(mixed,cal)
    t1=bool(np.all(s["state_L"]>0.2) and np.all(s["state_R"]>0.2))
    # 2) action common/contrast are orthogonal by construction.
    dots=np.sum(cal.action_common*cal.action_contrast,axis=1); t2=float(np.max(np.abs(dots)))<1e-8
    # 3) pure R transition points positive; pure L negative.
    ar=weighted_mean(action_observables(np.tile(0.35*aR,(L,1)),cal)["action_diff"],cal.layer_reliability)
    al=weighted_mean(action_observables(np.tile(0.35*aL,(L,1)),cal)["action_diff"],cal.layer_reliability)
    t3=ar>0.2 and al< -0.2

    # 4) causal erosion onset before full collapse.
    st=OnlineState(); rows=[]; prev=None; phis=[]
    base=np.tile(e1,(L,1))
    for t in range(28):
        x=base.copy()
        if t>=14:
            frac=min((t-14)/8,1.0)
            # gradually erode L and refract toward R; start is t=14, full shift t=22.
            x=(1-frac)*base + frac*np.tile(r,(L,1))
        x += rng.normal(0,0.005,x.shape)
        row=online_step(x,prev,cal,t,0,st,OnsetConfig(alpha=0.05,beta=0.20,min_effect_sigma=1.0))
        rows.append(row); phis.append(x); prev=x
    on=[i for i,rw in enumerate(rows) if rw["onset"]]
    t4=bool(on and on[0] < 22)

    # 5) controller reduces prism error and leaves orthogonal complement unchanged.
    refp,refv=build_reference(phis[5:12],cal); cur=phis[20]; prv=phis[19]
    out=prismatic_correction(cur,prv,refp,refv,cal,severity=1.0, cfg=ControlConfig(trust_ratio=0.10))
    before=np.linalg.norm(prism_coordinates(cur,cal)-refp)
    after=np.linalg.norm(prism_coordinates(out["corrected"],cal)-refp)
    # orthogonal residual relative to prism basis
    def orth(x):
        c=prism_coordinates(x,cal); recon=np.einsum("ldk,lk->ld",cal.prism_U,c)
        return (x-cal.mu_N)-recon
    inv=np.max(np.abs(orth(out["corrected"])-orth(cur)))
    t5=after<before and inv<1e-9

    tests=[("coactivation preserved",t1),("common/contrast orthogonal",t2),
           ("signed action differential",t3),("erosion detected pre-collapse",t4),
           ("controller reduces prism error, preserves complement",t5)]
    print("CRAD v1.5-prismatic selftest")
    for name,ok in tests: print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    if on: print(f"  planted erosion start=14, full shift=22, detected={on[0]}")
    print(f"  controller prism error: {before:.4f} -> {after:.4f}; orthogonal drift={inv:.2e}")
    return 0 if all(ok for _,ok in tests) else 1


def demo() -> int:
    print(json.dumps({
        "model":"CRAD v1.5-prismatic",
        "state":"independent neutral-anchored L/R supports (coactivation preserved)",
        "action":"orthogonal common/contrast decomposition",
        "onset":"causal strictly-past robust baseline + vector excursion CUSUM",
        "retention":"retrospective outcome label, not onset trigger",
        "control":"PD feedback in local L/R prism; orthogonal complement invariant",
    },indent=2))
    return 0


def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--selftest",action="store_true"); p.add_argument("--demo",action="store_true")
    a=p.parse_args();
    if a.selftest:return selftest()
    if a.demo:return demo()
    p.print_help();return 0

if __name__=="__main__": raise SystemExit(main())
