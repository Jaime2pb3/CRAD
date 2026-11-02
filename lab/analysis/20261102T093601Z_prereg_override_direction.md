# Preregistered test — OVERRIDE direction

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
| 43 | 20261102T093100Z_qwen2.5-7b_T0.7_L14-20_s43 | 6 | 6 | 0 | 2.667 | 1.167 |
| 44 | 20261102T093214Z_qwen2.5-7b_T0.7_L14-20_s44 | 6 | 6 | 0 | 1.333 | 2.000 |
| 45 | 20261102T093331Z_qwen2.5-7b_T0.7_L14-20_s45 | 9 | 3 | 0 | 1.222 | 1.667 |
| 46 | 20261102T093448Z_qwen2.5-7b_T0.7_L14-20_s46 | 6 | 6 | 0 | 3.667 | 2.333 |

## Primary (confirmatory)

- Statistic mean(REFUSE) − mean(COMPLY): **-0.302**
- One-sided p: **0.8442**
- Seeds without permutations (<2 of a label): none
- **Decision: H1 REJECTED (statistic <= 0)**

## Exploratory only (no claim): non-COMPLY vs COMPLY

- statistic -0.302, p 0.8442

Labels seen: {'REFUSE': 21, 'COMPLY': 27}
