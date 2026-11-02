# Preregistration — OVERRIDE direction, Qwen2.5-7B-Instruct, T=0.7

Written: 2026-09-24, BEFORE running seeds 43-46.

## Discovery data (already seen, exploratory only)
- Run 20260924T040503Z_Qwen2.5-7B-Instruct_nf4_L14-20_T0.7_s42
- n_pre_override: COMPLY mean 1.00 (n=8), REFUSE mean 2.25 (n=4), auc_sym 0.859, p=0.023 uncorrected
- Same direction in the historical TEMPon file (2.0 vs 1.0, p=0.10)

## Hypothesis (one-sided)
H1: pre-event OVERRIDE count is HIGHER in REFUSE draws than in COMPLY draws.

## Confirmatory data
Seeds 43, 44, 45, 46 only (48 draws). Seed 42 is excluded from the test because it generated the hypothesis.
Configs: configs/qwen2.5-7b_T0.7_L14-20_s{43..46}.json, identical to s42 except seed and name.

## Primary and only confirmatory feature
n_pre_override (as computed by CRAD v1.3, commit eb8217b, unmodified).

## Test
Statistic: mean(REFUSE) - mean(COMPLY) over the pooled 48 draws.
Null distribution: 10000 permutations of labels WITHIN each seed (stratified).
One-sided p = fraction of permutations with statistic >= observed. Alpha = 0.05.

## Decision rules
- p < 0.05 and statistic > 0: H1 supported (OVERRIDE marks resisted pressure, not leak precursor).
- statistic <= 0: H1 rejected.
- 0 < statistic, p >= 0.05: inconclusive; no claim either way.
- A seed with fewer than 2 draws of either label contributes no permutations and is reported as such.

## Not decided by this test
All other features are exploratory and reported without p-value claims.
