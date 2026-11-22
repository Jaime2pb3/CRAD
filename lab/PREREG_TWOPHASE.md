# PREREG · Oil and water: is the CRAD L/R plane a separate channel from the decision?

TwoQuarks Research · Luis Jaime Ledesma Pérez · written on November 21, 2026, **before any GPU run of `two_phase.py`**.
Code: `two_phase.py` v0.1 (hashes at the end). `python ladder.py twophase` refuses to run if the hash does not match.

## Where this hypothesis comes from (and why it is post hoc)

This hypothesis was written **after** seeing three results, so those results do not count as evidence for it. They only serve to state it.

1. **The September preregistered test failed** (`PREREG.md`, `PREREG_RESULT.md`). `n_pre_override` does not separate COMPLY from REFUSE, and its direction changes between seeds. The rejection replicated on the G4. The hypothesis of OVERRIDE as a leak precursor is **dead**. It is not rescued or reinterpreted here.
2. **The October TOP guard showed no defense** (`PREREG_TOP.md`). But the conjugate did pass: ANTITOP added 12 leaks, while TOP removed 5 and the sham removed 3. Pushing toward the secret is much easier than pulling the model out of it.
3. **Patching of one pair** (`results/patch_probe_A100_src5_tgt0.md`): copying only the L/R plane never flipped the decision, and copying everything except L/R flipped it as well as the full state.

**Correction before writing this.** Result 3 does **not** show that the channels are separate. The real difference δ = h_TRUTH − h_SF has between 0.1% and 2% of its energy inside L/R (column `energy LR` of the report). The L/R only patch moved the state very little, and its effect on log p(secret) was roughly proportional to that share (L14: 9% of the amplitude and 7% of the effect). Given that, the result that L/R does not flip and the rest does follows from geometry alone.

What is real and not trivial, and motivates this test:

- **Fisher** (`results/FISHER_G4.md`): R is a very loud direction for the output (R/rand 548x at L14, 350x at L17), louder even than the real decision direction at L14 and L17.
- **δ barely touches L/R**, even though L/R was built from the same activations.

In other words, the decision travels outside the loudest channel. That is the oil and water picture. This test puts both on equal footing: **every direction gets the same dose**.

## The hypothesis

At the decision row of turn 7, inside the CRAD band, the residual carries two channels that travel together but do not mix:

- **S** = span{uL, uR}, the CRAD measurement plane: a **loud** channel that does **not carry** the TRUTH vs SELF_FICTION choice.
- **C** = where δ lives: it **carries** the choice.
- Pushing either one does not make the network write into the other one downstream.

## Data

The most recent patching capture that passes the contract (`ladder.py patch_capture`, configuration `qwen2.5-7b_T0.7_cal0_L14-20_s42`, superset 13 to 28), **on the same GPU as the capture** (the replay requires token by token reproduction). The `calibration_id` is written to `run.json`.

The states of that capture were already used in Fisher and BELEN with one pair. **None of the readouts of this test exist yet.**

## Pairs, layers and rows

- **Pairs:** every TRUTH × SELF_FICTION pair with an anchor row, in both directions. If there are more than 40, 40 are drawn at random with seed 20261121 (deterministic). Equal history length is **recorded**, not required.
- **Fixed exclusion rule:** a pair is dropped if the fiction token at its decision row is a secret token (the margin is undefined). A (pair, layer) is dropped if both states are identical or if δ has no part in S. Every drop is written to `run.json`.
- **Row:** the anchor row of the contract (`crad_contract`, single clock).
- **Primary layers:** 14 and 17 (inside the band, where the CRAD ruler is defined). **Exploratory:** 21.
- **Generation:** turn 7 is cut at `commit_row + 24` tokens. A gate checks that the cut does not change the class or the content of the captured turn.

## Conditions

At layer l, row d of the target, with a = ‖h_src − h_tgt‖ (the natural amplitude of the decision):

| mode | injected vector (REPLACES the residual) | purpose |
|---|---|---|
| full | h_src | replication of patch_probe |
| C | h_tgt + a · unit(P⊥ δ) | the decision channel, full dose |
| S+ | h_tgt + a · unit(P_S δ) | the CRAD plane, full dose, along the part of δ that falls in S |
| S2 | h_tgt + a · o · e2 | the other direction of the plane. e2 is the unit vector of S orthogonal to e1 = unit(P_S (median h_TRUTH − median h_SF)), o = +1 for TRUTH→SF and −1 for SF→TRUTH |
| S− | h_tgt − a · unit(P_S δ) | secondary |
| nat | h_tgt + a · r_nat | **natural null**: a direction drawn from the covariance of the calibration states at l (64 principal components), orthogonal to S and to δ |
| iso | h_tgt + a · r_iso | isotropic null, orthogonal to S and to δ (secondary) |

The null directions are drawn once per (pair, layer) with seed 20261122.

## Readouts

- **toward** = sign · Δ[log p(first secret token) − log p(fiction token)] at row d, sign +1 if the source is TRUTH. The fiction token is the one the SELF_FICTION member of the pair emitted at its decision row. **The difference cancels the softmax normalizer**: a loud push that only rescales the distribution does not count as moving the choice. log p(secret) alone is kept as `toward_lp` (secondary).
- **KL** = KL(p_patched ‖ p_base) at row d, over the whole vocabulary. It measures how **loud** the push is.
- **dS, dC** = the network response downstream. For l < l′ ≤ l + 6, resp = (h_patched − h_base) − (v − h_tgt), that is, what the network **added** on top of carrying the injection through the residual. dS = the norm of its projection on S(l′), in calibration sigmas. dC = the absolute value of its projection on the decision axis of l′ with S(l′) removed, in sigmas of the decision rows (axis built without the two draws of the pair). Averaged over l′.

## Gates (all abort)

1. The unpatched run reproduces the captured turn token by token.
2. Self-patch: writing the target state itself does not move log p(secret) by more than 1e-3 nats, and the KL against the base is ≤ 1e-6.
3. Read-back: the traced residual at (d, l) is the injected vector (relative error ≤ 2e-3, fp16).
4. Cutting the turn at `commit_row + 24` does not change its class or its content.

## Hypotheses and tests

Unit = (pair, primary layer). Layers 14 and 17 are pooled. Intervals are a **cluster bootstrap** over target draws (10,000 resamples, seed 20261123), at 90% (equivalent to one-sided tests at 5% and to TOST at 5% for equivalence).

| | statistic | passes if |
|---|---|---|
| **G0 · replication** (reading gate) | mean toward(full) > 0 (one-sided sign flip), and median toward(C)/toward(full) over pairs with \|toward(full)\| > 0.05 | p < 0.05 **and** median ≥ 0.8 |
| **H1 · S is loud** | mean log(KL_S+ / KL_nat) | lower bound of the 90% CI > 0 |
| **H2 · S does not carry the choice** | R_X = (mean toward_X − mean toward_nat) / mean toward_C, for X = S+ **and** X = S2 | **both** 90% CIs inside (−0.15, +0.15) |
| **H3a · C does not write into S** | mean dS[C] / mean dS[nat] | upper bound of the 90% CI < 2 |
| **H3b · S does not write into C** | mean dC[S+] / mean dC[nat] | upper bound of the 90% CI < 2 |

Why these margins, fixed now: 0.15 means that, per unit of amplitude, S moves the choice less than 15% of what C moves it. In the A100 pair, the S part had 9% of the amplitude of δ. 2 means that one channel can cause in the other at most twice what any natural direction causes.

## How it is read (in this order)

| result | reading |
|---|---|
| G0 fails | The patching does not replicate in this capture. No hypothesis is read. |
| any R_X with its whole 90% CI outside ±0.15 | **REJECTED (mixed):** with the full dose, S does move the choice. The A100 result was geometry. |
| H3a or H3b with the lower bound of the 90% CI ≥ 2 | **REJECTED (coupled):** pushing one channel makes the network write into the other. |
| H1, H2, H3a and H3b pass | **SUPPORTED in this capture:** S is louder than a natural direction, does not carry the choice, and neither channel writes into the other. One capture and one base seed: provisional. |
| H2, H3a and H3b pass, H1 does not | **Weak form only:** S does not carry the choice and does not mix, but it is not louder than a natural direction. The oil is not special. |
| any other combination | **INCONCLUSIVE:** some interval straddles its margin. No claim. |

The rule is written in code (`two_phase.verdict`) and was tested on planted worlds before this date (`python two_phase.py selftest`): the two-phase world reads SUPPORTED, the mixed worlds (through S+ or only through S2) and the backward-coupled world read REJECTED (mixed), the forward-coupled world reads REJECTED (coupled), and the world without a loud S reads weak form. Twelve seeds per world, no errors.

## Secondary and exploratory (no verdict of their own)

- Layer 21 and the layer profile of R and Q. If there is an **emulsion point**, my prediction is that it appears after the band, near L21, where the real difference dominates R. It is reported, not decided.
- S− (is the response antisymmetric?), the isotropic null, `toward_lp`, class flips, and the geometry of δ against S and against natural planes (`two_phase.py plan`).
- Sensitivity: equal history length only, only pairs with a different token at the anchor row, each direction separately.
- **Outside this test:** the TOP asymmetry (P4). It needs the v0.2 guard on the residual and its own preregistration.

## Declared limits

- One capture, one base seed, 4-bit nf4, one model, one secret.
- The bootstrap resamples targets. Sources repeat across targets, so the intervals may be somewhat optimistic.
- Rejecting requires the whole 90% CI to cross the margin in one of four tests. Supporting requires all of them to pass. The rule is stricter with the positive claim than with the negative one, on purpose.
- The dose a = ‖δ‖ may leave the linear range (in Fisher, linearity holds up to α ≈ 0.25). Every mode uses the same amplitude, so the comparison is fair, but the absolute numbers are not extrapolated.
- Carrying the choice is measured at the anchor row. In pairs where both draws open with the same token (`␣"`), the choice closes one row later. This is part of the sensitivity analysis.
- L and R are still measurement tools. This test asks what the network does when pushed along them, it does not assign them a mechanism.

## Hashes of the preregistered code

| file | sha256 |
|---|---|
| `two_phase.py` | `80c6eae4d92aad794e7ea9f7921539d09580c16cd503fb7a2dc2dbee91bd3b2e` |
| `two_phase_mock.py` | `19d10f8cfce1687aeeaa54d00684855ca72c3c17ceab6e7a9c21032bc899aaf1` |
| `crad_contract.py` | `8db45ad8edd557d662ee5a549e055c6a1f5b45346528fe174563e4914900157c` |
| `patch_probe.py` | `4cabb2e6ea0a5765cdd1da0f9f7fb169b828740c8eeca79ef793d01b4fe38f78` |
| `dose_probe.py` | `b2ddf2f56ea6576fb3bee3a9bd1772aa4d8aceb2e6f836f593c1061de0888aa9` |
| `capture.py` | `2536c0ba08823f2b49a49127baea9c991ef01fe296d37d77bc038b96a70720f4` |
| `src/crad_base.py` | `bd8d2f223bde8bce5549ce9a19f2e630ddce35c98c396b460860cafee47b38bb` |
| `src/CRAD_v1.3.py` | `7173a65c73fb942f1662e20bbafead720bb2b5b68fed6254b7ade9c84879ae67` |

The last six are lab modules that `two_phase.py` imports. They were not modified.
