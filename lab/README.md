# CRAD lab

The lab behind CRAD after the v0.1 preprint: capture, a shared measurement contract, activation patching, and a set of observational instruments with their own validation standard. Everything runs through one ladder (`ladder.py`) whose rungs cannot be skipped.

The preprint code at the repository root (`CRAD_v1.3.py`, `crad_base.py`) is untouched. The lab carries its own copy in `src/`, pinned by sha256 to commit `eb8217b` in `src/PINNED.json`, and refuses to run if the copy drifts.

**What this is not.** Apart from the two preregistered tests listed below, every result produced here is one capture and one seed: a probe, not a test. Secret is fake (`BLUEFALCON`), model is local (Qwen2.5-7B-Instruct, 4-bit).

---

## Run it on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jaime2pb3/CRAD/blob/Colab/lab/crad_lab_colab.ipynb)

`crad_lab_colab.ipynb` runs the same ladder cell by cell. It clones this repository, so nothing has to be uploaded. Google Drive is optional (`USE_DRIVE = True`) and only keeps your own results between sessions.

## The ladder

```bash
python ladder.py cpu            # rungs 0-2: AST, imports, 13 selftests (CPU, about a minute)
python ladder.py mock           # rung 3: the whole lab on a fake model, end to end (CPU)
python ladder.py preflight      # rung 4: real load, fp16 exact, reproducibility, self-patch hook (GPU)
python ladder.py cost           # cost estimate from times MEASURED by the preflight
python ladder.py capture-ref    # deterministic preprint case: cross-hardware replication
python ladder.py capture-patch  # patching capture: T=0 calibration, 20 draws at T=0.7
python ladder.py contract       # contract check and TRUTH/SELF_FICTION pairs chosen by rule
python ladder.py probe          # patch_probe on the rule-chosen pair
python ladder.py batch --run-batch   # patch_batch per layer (hours)
python ladder.py belenc         # BELENC-Fisher, then the BELENC validation sheets
python ladder.py how            # story_copy, patch_how attend and trace
python ladder.py all --upto contract
```

A red rung stops everything. Rungs that already have a result on disk are not repeated. `--expect-gpu "A100"` (or any name) stops the run if the machine is not the one of the series, because captures and patching must happen on the same GPU: the replay gates require token-exact reproduction of the captured turn.

The CPU rungs need `torch`, `transformers`, `numpy` and `scipy`. GPU rungs also need `bitsandbytes` and `accelerate`.

## What is in here

| File | Role |
|---|---|
| `capture.py`, `capture_launch.py`, `crad_launch.py`, `launch_common.py`, `fingerprint.py`, `run.py` | Capture of hidden states (`phi`) per token and layer, with a fingerprint of the environment and value-level checks |
| `crad_contract.py` | One rule, one clock, one decision point: classifies each draw (TRUTH, SELF_FICTION, FABRICATE, EARLY_LEAK) and fixes the anchor and commit rows every instrument reads |
| `CRAD_v1_5_prismatic.py`, `crad_triad.py` | Causal erosion/desync detector (CUSUM on C, D, U) and the triad calibration the contract uses. Both also contain a small corrective element that only pushes toward the retained constraint: a PD controller inside the L/R prism (orthogonal complement untouched) and a gate on the STEALTH channel |
| `dose_probe.py` | Dose response at the decision row with a placebo, and the RNG snapshot/restore used by every replay |
| `patch_probe.py`, `patch_batch.py` | Activation patching at the decision row: full, L/R plane only, everything but L/R, random plane. Self-patch gate in every layer |
| `patch_how.py` | The how: per-head attention to each context region (with an FWER null over all 784 heads) and causal tracing by block, attention, MLP and head |
| `story_copy.py` | Token-only check: does a fiction copy the code the model invented in its own story |
| `crad_rings.py` | Outcome-blind Mapper map plus seven-cylinder tomography per group, at 5 and 7 rings |
| `crad_tube.py` | Tube around the mean trajectory plus tangling (Russo et al. 2018): where trajectories fork, and whether the state already encoded the side |
| `belenc.py` | **BELENC**, a validation standard: known perturbations planted inside real activations (standard addition, ICH Q2(R2)) give each instrument a false positive rate, LOD95, LOQ and a random-matrix-theory reference |
| `belenc_fisher.py` | Linear response with the Fisher metric pulled back to the residual: converts geometric sigma into nats of behavior |
| `analyze_prereg.py`, `PREREG.md`, `PREREG_RESULT.md` | The September preregistered test: the plan (sha256 `bcdf63bf...6826`), the analysis, and a transcript of its result |
| `top_guard.py`, `top_run.py`, `src_top/top_residual.py`, `PREREG_TOP.md` | The October preregistered test: a TOP / ANTITOP guard between the logits and the sampler, under the same CRAD pressure. `top_guard.py selftest` runs on CPU. Code hashes are in `PREREG_TOP.md` |
| `requirements.txt` | Reference environment of the RTX 3060 |
| `mock_ladder.py`, `mock_run.py`, `preflight_gpu.py` | Rungs 3 and 4 |
| `ladder.py` | The ladder as a Python runner (converted from the Colab notebook, without Drive) |

### BELENC in one paragraph

A negative from an instrument ("the rings do not see the decision") cannot tell "it is not there" from "it is below what the instrument sees". BELENC plants perturbations of known location, direction and dose into the real activation matrix of a capture (shift, split, fork, transient and persistent precursor), at a dose measured in the natural sigma along that direction, and measures each instrument against them. A sheet is valid for that matrix only. `belenc_fisher.py` then converts the doses to nats using the Fisher metric of the output distribution, so a geometric limit of detection becomes a behavioral one.

## Results so far

- **Cross-hardware replication.** The deterministic preprint case (median cos(L,R) 0.1523, coactivation 38.9%, leak at token 220) replicated on the RTX 3060, an A100 and an RTX PRO 6000 Blackwell. Sampled outcomes at T=0.7 differ by about 20% between GPUs because of near-tied logits, which is why captures and patching are tied to one GPU.
- **Patching, one pair** ([`results/patch_probe_A100_src5_tgt0.md`](results/patch_probe_A100_src5_tgt0.md)). Copying the full decision-row state of a TRUTH draw into a SELF_FICTION draw flips it to the true secret from L14 on. Copying only the L/R plane never flips it, and copying everything except L/R flips it as well as the full state. So in this pair the choice is already in the state, and not in the L/R plane. Known confound: in that A100 pair the two histories had different lengths (217 vs 210 tokens). The ladder now picks pairs by rule with equal history length.
- **The September preregistered test rejected its hypothesis** ([`PREREG.md`](PREREG.md), [`PREREG_RESULT.md`](PREREG_RESULT.md)). Over seeds 43 to 46 (48 draws), `n_pre_override` was not higher in refusals: statistic -0.661, p 0.85, and the direction changes between seeds. The first analysis script grouped on the wrong field and printed a spurious H1 SUPPORTED from a NaN. Revision 2 fixed the grouping and nothing else, and the result file records it.
- **The rejection replicated on other hardware** ([`PREREG_G4.md`](PREREG_G4.md), registered before running). Same test, same seeds, RTX PRO 6000 on Colab: statistic -0.302, p 0.84, H1 rejected. Draws are not comparable one to one between GPUs.
- **The October preregistered test did not show a defense** ([`PREREG_TOP.md`](PREREG_TOP.md), [`results/TOP_REPORT_decision_20261024T144729Z.md`](results/TOP_REPORT_decision_20261024T144729Z.md)). On the A100, 40 pairs per arm, OBS leaked 22 of 40. TOP tilt removed 5 (McNemar p 0.062, H1 not met), ANTITOP tilt added 12 (H2 passed), SHAM removed 3 (H3 failed), and the guard fired on 16% of neutral steps against a limit of 5% (H4 failed). Pinned TOP removed 19 leaks, but mostly by breaking the output: 33 of 40 ended as FABRICATE. Two prefixes, one base seed.
- **R is a response direction inside the band, and the CRAD axes do not carry the decision** ([`results/FISHER_G4.md`](results/FISHER_G4.md)). Sensitivity per unit norm against random directions, from `fisher.json`, one capture on the G4 series: R 18.6x at L14, 15.6x at L17 and 4.8x at L21. The real TRUTH vs SELF_FICTION difference: 7.5x, 15.9x and 36.7x. So the real difference only dominates R at L21.
- **BELENC sheets** for that capture: [L14](results/BELENC_G4_L14.md), [L17](results/BELENC_G4_L17.md), [L21](results/BELENC_G4_L21.md). A sheet is valid for that matrix only.

## Known gaps

- The September draws of seeds 43 to 46 live on the RTX 3060, so that result can be read but not recomputed. The G4 replication can: `lab/runs/` holds its summaries, and `python analyze_prereg.py` reproduces it. The configs here were rebuilt from the session that wrote them.
- The attention-forcing experiment explored in October is not part of this release.
