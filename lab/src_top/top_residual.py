#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TOP Residual Coupler (top_residual.py)
======================================

Inter-hadron isomeric controller inspired by nuclear residual forces.

Role:
-----
The module shapes the joint policies of multiple hadrons (agents) through
two competing channels:

SAFE  (L) → nuclear cohesion via residual coupling between hadrons.
AGGR  (R) → high-gain collapse driven by Q-values under extreme instability.

The isomeric polarization ρ ∈ [0,1] interpolates between both modes.

ρ ≈ 0 → SAFE-dominant (cohesion & stabilization)
ρ ≈ 1 → AGGRESSIVE-dominant (rupture / high-gain reconfiguration)

ρ is dynamically computed from nuclear observables:
    - binding_energy_gap
    - residual_force_magnitude
    - repulsion_magnitude

Lifetime:
---------
TOP intervention is ultrashort: both energy and ρ decay rapidly, modeling
the yoctosecond lifetime of the Top quark (no hadronization).

Integration:
------------
Designed to directly consume metrics produced by TopGaugeFieldEnv:

    - adjacency matrix (residual couplings)
    - binding_energy_gap
    - residual_force_mag
    - repulsion_mag
    - cluster_size
    - nuclear_stress
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


# =====================================================================
# 1. Configuration
# =====================================================================

@dataclass
class TopResidualConfig:
    """
    Hyperparameters for the TOP residual controller.
    """

    # Thresholds defining nuclear critical regimes
    gap_threshold: float = 0.6
    residual_threshold: float = 0.8
    repulsion_threshold: float = 0.7
    stress_threshold: float = 1.0

    # Isomeric gains
    max_gain_safe: float = 2.0
    max_gain_aggr: float = 4.0

    # Safe-channel neighbor coupling
    safe_coupling_strength: float = 0.45

    # Yocto-decay
    cooling_rate: float = 0.18

    # Polarization dynamics
    rho_lr: float = 0.25
    rho_k: float = 1.0


# =====================================================================
# 2. TOP Residual Coupler
# =====================================================================

class TopResidualCoupler:
    """
    Multi-hadron residual force controller.

    State variables:
        self.energy   -> instantaneous mass gain scaling collapse.
        self.rho      -> isomeric polarization [SAFE..AGGRESSIVE].

    Usage per step:

        if coupler.detect_critical_nucleus(...):
            coupler.inject_energy(...)

        policies_top = coupler.collapse_joint_policies(...)
        coupler.decay()
    """

    def __init__(self, cfg: Optional[TopResidualConfig] = None):
        self.cfg = cfg or TopResidualConfig()
        self.energy: float = 0.0
        self.rho: float = 0.5   # neutral isomeric state

    # -----------------------------------------------------------------
    # Episode reset
    # -----------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset nuclear TOP state for a new episode.

        Used by outer environments (e.g. Fermi Paradox adapter) to ensure
        each episode starts from a neutral nucleus with no accumulated
        energy or polarization bias.
        """
        self.energy = 0.0
        self.rho = 0.5

    # -----------------------------------------------------------------
    # Detection of nuclear critical regime
    # -----------------------------------------------------------------

    def detect_critical_nucleus(
        self,
        binding_energy_gap: float,
        residual_force_mag: float,
        repulsion_mag: float,
        cluster_size: int,
        nuclear_stress: float,
    ) -> bool:
        """
        Determines if the nucleus is in a critical regime requiring
        intervention by the TOP controller.
        """
        c = self.cfg

        # Borderline stability window
        near_edge = (
            binding_energy_gap >= 0.4
            and residual_force_mag >= c.residual_threshold
        )

        # Overload zone (repulsion + gap)
        overload = (
            repulsion_mag >= c.repulsion_threshold
            and binding_energy_gap >= c.gap_threshold
        )

        # Accumulated instability
        stressed = nuclear_stress >= c.stress_threshold

        # Large cluster under moderate instability
        cluster_fragile = (
            cluster_size >= 4
            and binding_energy_gap >= 0.45
        )

        return near_edge or overload or stressed or cluster_fragile

    # -----------------------------------------------------------------
    # Isomeric polarization dynamics
    # -----------------------------------------------------------------

    def _update_isomeric_state(
        self,
        binding_energy_gap: float,
        residual_force_mag: float,
        repulsion_mag: float,
    ):
        """
        Computes target ρ from nuclear conditions and moves current ρ
        toward it using an exponential smoother.
        """
        c = self.cfg

        # normalize signals
        s_gap = binding_energy_gap / (c.gap_threshold + 1e-8)
        s_res = residual_force_mag / (c.residual_threshold + 1e-8)
        s_rep = repulsion_mag / (c.repulsion_threshold + 1e-8)

        s_gap = min(1.5, s_gap)
        s_res = min(1.5, s_res)
        s_rep = min(1.5, s_rep)

        # cohesion vs instability
        cohesion_signal    = s_res - 0.5 * s_rep
        instability_signal = s_gap + 0.5 * s_rep

        nuclear_stress_signal = instability_signal - 0.5 * cohesion_signal

        # squash around 1 → neutral
        rho_target = 0.5 * (1.0 + np.tanh(c.rho_k * (nuclear_stress_signal - 1.0)))

        self.rho = (1 - c.rho_lr) * self.rho + c.rho_lr * rho_target
        self.rho = float(np.clip(self.rho, 0.0, 1.0))

    # -----------------------------------------------------------------
    # Energy injection (Top "mass")
    # -----------------------------------------------------------------

    def inject_energy(
        self,
        binding_energy_gap: float,
        residual_force_mag: float,
        repulsion_mag: float,
    ):
        """
        Updates polarization and effective TOP energy.
        """
        c = self.cfg

        self._update_isomeric_state(
            binding_energy_gap,
            residual_force_mag,
            repulsion_mag,
        )

        self.energy = (
            (1.0 - self.rho) * c.max_gain_safe
            + self.rho * c.max_gain_aggr
        )

    # -----------------------------------------------------------------
    # Residual collapse of joint policies
    # -----------------------------------------------------------------

    def collapse_joint_policies(
        self,
        policies: np.ndarray,
        q_mean: np.ndarray,
        q_var: Optional[np.ndarray],
        adjacency: np.ndarray,
    ) -> np.ndarray:
        """
        Applies SAFE + AGGRESSIVE isomeric mixture over multiple hadrons.

        Args:
            policies  : (N,A) baseline distributions.
            q_mean    : (N,A) expected Q-values.
            q_var     : (N,A) uncertainty (optional).
            adjacency : (N,N) residual coupling matrix.
        """

        # ---------- helpers anti-NaN / anti-colapso ----------
        def _sanitize_probs(P: np.ndarray) -> np.ndarray:
            """
            Garantiza:
            - sin NaN / Inf
            - ninguna fila totalmente cero
            - filas normalizadas a 1
            """
            P = np.asarray(P, dtype=float)
            P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

            row_sum = P.sum(axis=1, keepdims=True)          # (N,1)
            bad_rows = (row_sum <= 0.0).squeeze(-1)         # (N,)

            if np.any(bad_rows):
                n_actions = P.shape[1]
                P[bad_rows, :] = 1.0 / float(n_actions)
                row_sum = P.sum(axis=1, keepdims=True)

            row_sum = np.clip(row_sum, 1e-12, None)
            return P / row_sum

        def _sanitize_matrix(M: np.ndarray) -> np.ndarray:
            return np.nan_to_num(np.asarray(M, dtype=float),
                                 nan=0.0, posinf=0.0, neginf=0.0)

        # ---------- copias base ----------
        P = _sanitize_probs(policies)
        Q = _sanitize_matrix(q_mean)
        A = _sanitize_matrix(adjacency)

        N, A_dim = P.shape  # A_dim no se usa, pero documenta estructura

        # Caso trivial: TOP sin energía → devolvemos políticas base
        if self.energy <= 1e-6:
            return P

        # ------------------------------------------------------------------
        # SAFE channel: acoplamiento residual entre hadrones
        # ------------------------------------------------------------------
        W = np.clip(A, 0.0, None)
        W = _sanitize_matrix(W)

        row_w = W.sum(axis=1, keepdims=True)
        row_w[row_w <= 0.0] = 1.0
        W /= row_w  # filas estocásticas

        P_neigh = W @ P
        alpha = self.cfg.safe_coupling_strength
        P_safe = (1.0 - alpha) * P + alpha * P_neigh

        # penalizar acciones inciertas si q_var está disponible
        if q_var is not None:
            V = _sanitize_matrix(q_var)

            V = V - V.min(axis=1, keepdims=True)          # (N,A)
            vmax = V.max(axis=1, keepdims=True)           # (N,1)

            denom = np.where(vmax > 0.0, vmax, 1.0)
            V_norm = np.where(vmax > 0.0, V / denom, 0.0)

            # acotar a [0,1] para no invertir signo
            V_norm = np.clip(V_norm, 0.0, 1.0)

            # más var → menos probabilidad
            P_safe *= (1.0 - 0.6 * V_norm)

        P_safe = _sanitize_probs(P_safe)

        # ------------------------------------------------------------------
        # AGGRESSIVE channel: colapso por Q-values (alta ganancia)
        # ------------------------------------------------------------------
        Q_shift = Q - Q.min(axis=1, keepdims=True)
        qmax = Q_shift.max(axis=1, keepdims=True)
        denom_q = np.where(qmax > 0.0, qmax, 1.0)
        Q_norm = np.where(qmax > 0.0, Q_shift / denom_q, 0.0)

        logits = self.energy * Q_norm
        logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        logits -= logits.max(axis=1, keepdims=True)

        expq = np.exp(logits)
        P_aggr = expq / (expq.sum(axis=1, keepdims=True) + 1e-9)
        P_aggr = _sanitize_probs(P_aggr)

        # ------------------------------------------------------------------
        # Isomeric mixing SAFE–AGGR controlado por ρ
        # ------------------------------------------------------------------
        rho = float(self.rho)
        P_top = (1.0 - rho) * P_safe + rho * P_aggr
        P_top = _sanitize_probs(P_top)

        return P_top

    # -----------------------------------------------------------------
    # Decay (yoctosecond lifetime)
    # -----------------------------------------------------------------

    def decay(self):
        """
        Natural decay of both energy and polarization toward neutrality.
        """
        c = 1.0 - self.cfg.cooling_rate

        self.energy *= c
        self.rho = 0.5 + (self.rho - 0.5) * c

        self.energy = float(max(self.energy, 0.0))
        self.rho = float(np.clip(self.rho, 0.0, 1.0))

