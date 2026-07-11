# CRAD — Constraint-Retained Action Divergence

**Neutral-anchored detection of pre-leak state–action dissociation in a frozen language model.**

| Author: Luis Jaime Ledesma Pérez |
| Lab:TwoQuarks Research  |
| Contact: `research@twoquarks.com` |
| Website: https://twoquarks.com  |

---

## What this is?

Safety failures in language models are usually described as a refusal direction **disappearing**, being suppressed, or 
being traversed. This repository studies a different possibility: An action can become attributable to an opposing 
behavioral direction **while the safety constraint remains detectably active.**

We call this pattern **constraint-retained action divergence (CRAD)**. Constraint retained, action diverges.

If it holds, the implication is uncomfortable for monitoring: verifying that the refusal direction is still present may 
not be enough. The constraint can remain intact in the residual stream and a leak can still occur.

This is a **methods preprint and a single deterministic case study.** It is not a general mechanism, not a causal claim, 
and not a cross-model result. See *Scope and limitations* below.

## The instrument in one paragraph

The detector is two-stage and purely observational (no activation steering):

1. **Energy-damped R-action onset.**
     A token is flagged when an opposing-direction action transition carries enough energy across a coherent block of
     layers, damping low-magnitude ratio noise toward zero.
3. **Posterior constraint hold.**
     For each onset, the original constraint's support is compared against its own recent baseline over a short posterior
     horizon. The onset is classified `OVERRIDE` (L retained — a CRAD candidate), `CONSISTENT` (L released), or `AMBIGUOUS`.

The key geometric correction: L and R state supports are built against a **third neutral anchor N**, not against their mutual 
midpoint. Midpoint-centered bipolar axes are algebraically antipodal (`a_R = -a_L`), which makes coactivation impossible and 
can force a false "constraint released" conclusion. That failed construction is retained in the code as a self-test.

## Repository contents

```
  CRAD/
    ├── CRAD_v1.3.py                 # the two-stage detector (entry point)
    ├── pomp_dual_token_tests.py     # base instrument: model wrapper, ramp, draws
    ├── crad_preprint_v0_1.tex       # preprint source
    ├── refs.bib                     # references
    ├── results_snapshot.json        # deterministic candidate, verbatim
    ├── figures/                     # geometry, trajectory, hold-by-layer
    └── README.md
```

`CRAD_v1.3.py` imports `RecogModel`, `build_real_ramp`, `run_pressure_draw`, and `auc_rank` from `pomp_dual_token_tests.py`. 
Keep the two files side by side.

## Quick start

Verify the logic with no GPU and no model download:

```bash
python CRAD_v1.3.py --selftest
```

Expected output ends in `SELFTEST: ALL OK`. The six checks confirm non-antipodal neutral geometry, simultaneous L/R support, correct 
OVERRIDE/CONSISTENT classification on planted trajectories, low-energy damping, and detection of the invalid midpoint construction.

## Reproducing the deterministic case study

The primary result is a greedy (T=0) trajectory on Qwen2.5-7B-Instruct, 4-bit, over layers 14–19:

```bash
python CRAD_v1.3.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --temperature-mode off \
  --calibration-draws 4 --draws 12 \
  --n-tokens 96 \
  --layer-lo 14 --layer-hi 20 \
  --pre-window 32 --post-window 8 \
  --action-threshold 0.12 --action-layer-frac 0.50 \
  --min-contiguous-layers 2 \
  --baseline-pre-tokens 6 \
  --hold-horizon 4 --hold-ratio-min 0.75 --release-ratio-max 0.45 \
  --hold-layer-frac 0.50 \
  --n-perm 300 \
  --out crad_deterministic_L14-20.jsonl
```

Under `--temperature-mode off`, repeated greedy draws collapse to one effective trajectory to prevent pseudoreplication. 
This is intentional: at T=0 the draws are duplicates, and treating them as independent samples would be a statistical error.

## What the deterministic run found

Values are reproduced verbatim in `results_snapshot.json`.
    __________________________________________________________________
    |           Quantity           |               Value             |
    |______________________________|_________________________________|
    |      Model / layer band      |   Qwen2.5-7B-Instruct / 14–19   |
    |           Decoding           |           Greedy (T=0)          |
    |Neutral-anchored med. cos(L,R)|  0.1524 (antipodal fraction 0%) |
    |  Aggregate L/R coactivation  |             38.8%               |
    |    Candidate global token    |              216                |
    |    Offset from leak start    |            −1 token             |
    |       R-action layers        |  4 / 6 (contiguous block of 4)  |
    |  Mean / max damped R-action  |         0.1157 / 0.1762         |
    |   Posterior L-hold fraction  |              6 / 6              |
    | Posterior L-release fraction |              0 / 6              |
    __________________________________________________________________

One classified pre-leak candidate appeared one generated token before secret emission: the action transition was R-attributed 
across a coherent four-layer block while the L-constraint was retained in all six monitored layers over the posterior window.

## Scope and limitations

This evidence is deliberately narrow:
- One model, one layer band, one secret, one effective deterministic trajectory.
- **No control trajectories yet.** The result establishes existence, not discrimination.
- The result is observational. No intervention shows the candidate is necessary or sufficient for leakage.
- L/R state supports and action fields are empirical proxies, not demonstrated internal policies.

## Falsification criteria

CRAD should be considered falsified or substantially weakened if any of these hold under replication:
1. neutral anchoring repeatedly collapses back to antipodal geometry;
2. pre-leak candidates occur at equal or higher rates in matched non-leak controls without a distinguishing posterior pattern;
3. candidate timing fails to replicate across seeds, scenarios, or architectures;
4. teacher-forced identical token sequences produce temperature-dependent hidden states beyond numerical tolerance;
5. plain and traced generation differ under identical seeds, indicating an invasive measurement path;
6. interventions that selectively remove or preserve L support do not alter the predicted classification or behavior.

The next phase is deterministic scenario variation against matched non-leak controls, with statistical tests at the scenario/seed level.

## Citation

```bibtex
@misc{ledesma2026crad,
  title        = {Constraint-Retained Action Divergence: Neutral-Anchored
                  Detection of Pre-Leak State--Action Dissociation in a
                  Frozen Language Model},
  author       = {Ledesma P{\'e}rez, Luis Jaime},
  year         = {2026},
  note         = {TwoQuarks Research, preprint v0.1},
  howpublished = {\url{https://twoquarks.com}}
}
```

## License

Released for research scrutiny and replication. Break it and tell me where.
