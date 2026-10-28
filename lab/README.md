# CRAD lab

The lab behind CRAD after the v0.1 preprint: capture, a shared measurement contract, activation patching, and a set of observational instruments with their own validation standard. Everything runs through one ladder (`ladder.py`) whose rungs cannot be skipped.

The preprint code at the repository root (`CRAD_v1.3.py`, `crad_base.py`) is untouched. The lab carries its own copy in `src/`, pinned by sha256 to commit `eb8217b` in `src/PINNED.json`, and refuses to run if the copy drifts.

**What this is not.** Every result produced here is one capture and one seed: a probe, not a test. Secret is fake (`BLUEFALCON`), model is local (Qwen2.5-7B-Instruct, 4-bit).

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
| `analyze_prereg.py` | Analysis of the September preregistered test |
| `mock_ladder.py`, `mock_run.py`, `preflight_gpu.py` | Rungs 3 and 4 |
| `ladder.py` | The ladder as a Python runner (converted from the Colab notebook, without Drive) |

### BELENC in one paragraph

A negative from an instrument ("the rings do not see the decision") cannot tell "it is not there" from "it is below what the instrument sees". BELENC plants perturbations of known location, direction and dose into the real activation matrix of a capture (shift, split, fork, transient and persistent precursor), at a dose measured in the natural sigma along that direction, and measures each instrument against them. A sheet is valid for that matrix only. `belenc_fisher.py` then converts the doses to nats using the Fisher metric of the output distribution, so a geometric limit of detection becomes a behavioral one.

## Results so far

- **Cross-hardware replication.** The deterministic preprint case (median cos(L,R) 0.1523, coactivation 38.9%, leak at token 220) replicated on the RTX 3060, an A100 and an RTX PRO 6000 Blackwell. Sampled outcomes at T=0.7 differ by about 20% between GPUs because of near-tied logits, which is why captures and patching are tied to one GPU.
- **Patching, one pair** ([`results/patch_probe_A100_src5_tgt0.md`](results/patch_probe_A100_src5_tgt0.md)). Copying the full decision-row state of a TRUTH draw into a SELF_FICTION draw flips it to the true secret from L14 on. Copying only the L/R plane never flips it, and copying everything except L/R flips it as well as the full state. So in this pair the choice is already in the state, and not in the L/R plane. Known confound: in that A100 pair the two histories had different lengths (217 vs 210 tokens). The ladder now picks pairs by rule with equal history length.
- **The CRAD axes are not response directions, but R is not inert.** BELENC-Fisher measured the R axis at 16 to 19 times the sensitivity of random directions, while the actual TRUTH vs SELF_FICTION difference was about 36 times more sensitive and lived mostly outside the L/R plane.

## Known gaps

- `analyze_prereg.py` needs the `runs/` of seeds 43 to 46 and `PREREG.md`, which live on the RTX 3060.
- The attention-forcing experiment and the logit guard explored in October are not part of this release.
