# BELENC v0.1-belenc — ficha de validación

**Matriz:** `20261023T165139Z_qwen2.5-7b_T0.7_cal0_L14-20_s42_cap13-28` · capa 14 · turno 7 · fila plantada 2 · 14 draws × 10 filas × d=3584
**Geometría:** d_eff (participation ratio) 11.6 · γ = d_eff/n_draws 0.83 · umbral BBP √γ = 0.91

- real: h[5, fila 4] - h[2, fila 4], |delta|/|h_tgt| = 0.166

Réplicas por dosis: 20 · LOD95 = dosis mínima con éxito ≥ 95% en ella y en todas las mayores · dosis en σ naturales a lo largo de la dirección plantada

**Una ficha vale para esta matriz. Un negativo del instrumento solo es informativo para efectos por encima de su LOD; un positivo solo si supera su tasa en blancos.**

| analito | familia | instrumento | FPR (IC 95%) | a=0.5 | a=1 | a=2 | a=4 | a=8 | a=16 | LOD95 (σ) | LOD95 (nats) | extra |
|---|---|---|---|---|---|---|---|---|---|---|---|
| shift | high | oracle | 0.00 (0.00–0.16) | 0.05 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | **2** | 0.00565 | pend 10.02 R² 0.82 |
| shift | high | pca | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.80 | 1.00 | 1.00 | 1.00 | **4** | 0.0226 | teoría: traslape 0.5 en a≈2.02, BBP a≈1.45 |
| shift | high | rings | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| shift | low | oracle | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.35 | **—** | — | — |
| shift | low | pca | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | teoría: traslape 0.5 en a≈48.40, BBP a≈34.75 |
| shift | low | rings | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| shift | random | oracle | 0.10 (0.03–0.30) | 0.10 | 0.10 | 0.10 | 0.05 | 0.05 | 0.25 | **—** | — | — |
| shift | random | pca | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | teoría: traslape 0.5 en a≈49.60, BBP a≈35.61 |
| shift | random | rings | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| shift | real | oracle | 0.15 (0.05–0.36) | 0.15 | 0.15 | 0.05 | 0.15 | 1.00 | 1.00 | **8** | 0.0395 | pend 1.23 R² 0.68 |
| shift | real | pca | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | **16** | 0.158 | teoría: traslape 0.5 en a≈13.96, BBP a≈10.03 |
| shift | real | rings | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| split | high | oracle | 0.10 (0.03–0.30) | 0.15 | 0.45 | 1.00 | 1.00 | 1.00 | 1.00 | **2** | 0.00565 | — |
| split | high | pca | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.60 | 1.00 | 1.00 | 1.00 | **4** | 0.0226 | teoría: traslape 0.5 en a≈2.84, BBP a≈2.04 |
| split | high | rings | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| split | low | oracle | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.35 | **—** | — | — |
| split | low | pca | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | teoría: traslape 0.5 en a≈42.40, BBP a≈30.44 |
| split | low | rings | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| split | random | oracle | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.15 | **—** | — | — |
| split | random | pca | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | teoría: traslape 0.5 en a≈50.17, BBP a≈36.02 |
| split | random | rings | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| split | real | oracle | 0.15 (0.05–0.36) | 0.15 | 0.10 | 0.05 | 0.00 | 1.00 | 1.00 | **8** | 0.0395 | — |
| split | real | pca | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | **16** | 0.158 | teoría: traslape 0.5 en a≈13.96, BBP a≈10.03 |
| split | real | rings | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| fork | high | tube_fork | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.15 | 0.95 | 1.00 | **8** | 0.0904 | |err fila| 0.76 |
| fork | low | tube_fork | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| fork | random | tube_fork | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| fork | real | tube_fork | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| precursor | high | oracle | 0.05 (0.01–0.24) | 0.00 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | **2** | 0.00565 | — |
| precursor | high | tube_anticip | 0.15 (0.05–0.36) | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| precursor | low | oracle | 0.15 (0.05–0.36) | 0.15 | 0.15 | 0.15 | 0.15 | 0.10 | 0.45 | **—** | — | — |
| precursor | low | tube_anticip | 0.20 (0.08–0.42) | 0.20 | 0.20 | 0.25 | 0.20 | 0.15 | 0.05 | **—** | — | — |
| precursor | random | oracle | 0.05 (0.01–0.24) | 0.05 | 0.05 | 0.05 | 0.00 | 0.00 | 0.20 | **—** | — | — |
| precursor | random | tube_anticip | 0.10 (0.03–0.30) | 0.10 | 0.10 | 0.10 | 0.10 | 0.05 | 0.05 | **—** | — | — |
| precursor | real | oracle | 0.20 (0.08–0.42) | 0.20 | 0.15 | 0.05 | 0.00 | 1.00 | 1.00 | **8** | 0.0395 | — |
| precursor | real | tube_anticip | 0.15 (0.05–0.36) | 0.40 | 0.35 | 0.35 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| precursor_persist | high | oracle | 0.10 (0.03–0.30) | 0.00 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | **2** | 0.00565 | — |
| precursor_persist | high | tube_anticip | 0.25 (0.11–0.47) | 0.15 | 0.15 | 0.15 | 0.75 | 1.00 | 1.00 | **8** | 0.0904 | — |
| precursor_persist | low | oracle | 0.05 (0.01–0.24) | 0.05 | 0.05 | 0.05 | 0.05 | 0.00 | 0.50 | **—** | — | — |
| precursor_persist | low | tube_anticip | 0.15 (0.05–0.36) | 0.15 | 0.15 | 0.15 | 0.15 | 0.00 | 0.05 | **—** | — | — |
| precursor_persist | random | oracle | 0.10 (0.03–0.30) | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.25 | **—** | — | — |
| precursor_persist | random | tube_anticip | 0.15 (0.05–0.36) | 0.15 | 0.15 | 0.15 | 0.20 | 0.40 | 0.15 | **—** | — | — |
| precursor_persist | real | oracle | 0.20 (0.08–0.42) | 0.20 | 0.10 | 0.05 | 0.00 | 1.00 | 1.00 | **8** | 0.0395 | — |
| precursor_persist | real | tube_anticip | 0.30 (0.15–0.52) | 0.25 | 0.25 | 0.20 | 0.15 | 0.05 | 0.05 | **—** | — | — |

Lectura de la columna teoría (solo pca): la dosis donde la teoría de matrices aleatorias predice traslape² = 0.5 bajo un bulto isotrópico de dimensión d_eff. LOD empírico ≈ teoría: la matriz se comporta como ruido isotrópico en esa familia. LOD empírico ≫ teoría: efecto matriz (estructura real que compite con la dirección). LOD empírico ≪ teoría: la familia vive en un subespacio más limpio que el promedio.

Conversión a nats con `captures/20261023T165139Z_qwen2.5-7b_T0.7_cal0_L14-20_s42_cap13-28/belenc_fisher/20261023T165510Z/fisher.json`: KL ≈ ½·a²·σ_u²·uᵀGu, mediana por familia de dirección (respuesta lineal, belenc_fisher.py).
