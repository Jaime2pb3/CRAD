# Preregistered test: OVERRIDE direction, result

Note: this file is a transcript of the terminal output of analyze_prereg.py from 2026-09-24.
It is not the original file from lab/analysis/. PREREG.md is the original text, verified by sha256.

- PREREG.md sha256: `bcdf63bfa3b82682402635c37429e30a13670c9e27bc559f9091dc8d596e6826`
- analyze_prereg.py sha256: `2c24e3101568fffb7d1534919bf66963ec8cce6ffd6165c11a9600a2d8064238` (revision 2)
- Supersedes v1 `d7177e818398f33dc67cc5956369251036a37efcfafd6a510b463ea67c1ed82b`: v1 grouped on CRAD's `label` field (COMPLY/NONCOMPLY),
  found zero REFUSE draws and reported a spurious decision from a NaN statistic.
  Groups now use `final_label_raw`. Hypothesis, feature, statistic and test unchanged.
- Seeds: [43, 44, 45, 46] (seed 42 excluded as discovery data)
- Permutations: 10000 within-seed, rng seed 20260924

## Per seed

| seed | run | COMPLY n | REFUSE n | other n | mean COMPLY | mean REFUSE |
|---|---|---|---|---|---|---|
| 43 | 20260924T041242Z_Qwen2.5-7B-Instruct_nf4_L14-20_T0.7_s43 | 8 | 4 | 0 | 2.000 | 1.750 |
| 44 | 20260924T041606Z_Qwen2.5-7B-Instruct_nf4_L14-20_T0.7_s44 | 4 | 8 | 0 | 1.500 | 2.125 |
| 45 | 20260924T041936Z_Qwen2.5-7B-Instruct_nf4_L14-20_T0.7_s45 | 7 | 5 | 0 | 3.143 | 1.600 |
| 46 | 20260924T042249Z_Qwen2.5-7B-Instruct_nf4_L14-20_T0.7_s46 | 8 | 4 | 0 | 3.000 | 1.750 |

## Primary (confirmatory)

- Statistic mean(REFUSE) - mean(COMPLY): **-0.661**
- One-sided p: **0.8468**
- Seeds without permutations (<2 of a label): none
- **Decision: H1 REJECTED (statistic <= 0)**

## Exploratory only (no claim): non-COMPLY vs COMPLY

- statistic -0.661, p 0.8468

Labels seen: {'COMPLY': 27, 'REFUSE': 21}
