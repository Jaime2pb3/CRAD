# BELENC v0.1-belenc — ficha de validación

**Matriz:** `20261023T165139Z_qwen2.5-7b_T0.7_cal0_L14-20_s42_cap13-28` · capa 21 · turno 7 · fila plantada 2 · 14 draws × 10 filas × d=3584
**Geometría:** d_eff (participation ratio) 10.9 · γ = d_eff/n_draws 0.78 · umbral BBP √γ = 0.88

- real: h[5, fila 4] - h[2, fila 4], |delta|/|h_tgt| = 0.273

Réplicas por dosis: 20 · LOD95 = dosis mínima con éxito ≥ 95% en ella y en todas las mayores · dosis en σ naturales a lo largo de la dirección plantada

**Una ficha vale para esta matriz. Un negativo del instrumento solo es informativo para efectos por encima de su LOD; un positivo solo si supera su tasa en blancos.**

| analito | familia | instrumento | FPR (IC 95%) | a=0.5 | a=1 | a=2 | a=4 | a=8 | a=16 | LOD95 (σ) | LOD95 (nats) | extra |
|---|---|---|---|---|---|---|---|---|---|---|---|
| shift | high | oracle | 0.10 (0.03–0.30) | 0.10 | 0.45 | 1.00 | 1.00 | 1.00 | 1.00 | **2** | 0.0433 | pend 11.33 R² 0.85 |
| shift | high | pca | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.55 | 1.00 | 1.00 | 1.00 | **4** | 0.173 | teoría: traslape 0.5 en a≈2.28, BBP a≈1.64 |
| shift | high | rings | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| shift | low | oracle | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | 0.65 | **—** | — | — |
| shift | low | pca | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | **—** | — | teoría: traslape 0.5 en a≈37.92, BBP a≈27.35 |
| shift | low | rings | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| shift | random | oracle | 0.10 (0.03–0.30) | 0.10 | 0.10 | 0.10 | 0.10 | 0.05 | 0.40 | **—** | — | — |
| shift | random | pca | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | teoría: traslape 0.5 en a≈44.89, BBP a≈32.38 |
| shift | random | rings | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| shift | real | oracle | 0.10 (0.03–0.30) | 0.10 | 0.05 | 0.10 | 0.60 | 1.00 | 1.00 | **8** | 0.593 | pend 2.21 R² 0.78 |
| shift | real | pca | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | **8** | 0.593 | teoría: traslape 0.5 en a≈8.76, BBP a≈6.32 |
| shift | real | rings | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| split | high | oracle | 0.25 (0.11–0.47) | 0.10 | 0.20 | 1.00 | 1.00 | 1.00 | 1.00 | **2** | 0.0433 | — |
| split | high | pca | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.30 | 1.00 | 1.00 | 1.00 | **4** | 0.173 | teoría: traslape 0.5 en a≈3.14, BBP a≈2.26 |
| split | high | rings | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| split | low | oracle | 0.05 (0.01–0.24) | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.55 | **—** | — | — |
| split | low | pca | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | teoría: traslape 0.5 en a≈32.65, BBP a≈23.55 |
| split | low | rings | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| split | random | oracle | 0.05 (0.01–0.24) | 0.05 | 0.05 | 0.00 | 0.00 | 0.00 | 0.10 | **—** | — | — |
| split | random | pca | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | teoría: traslape 0.5 en a≈50.12, BBP a≈36.15 |
| split | random | rings | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| split | real | oracle | 0.15 (0.05–0.36) | 0.15 | 0.15 | 0.00 | 0.50 | 1.00 | 1.00 | **8** | 0.593 | — |
| split | real | pca | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | **8** | 0.593 | teoría: traslape 0.5 en a≈8.76, BBP a≈6.32 |
| split | real | rings | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| fork | high | tube_fork | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.05 | 0.85 | 1.00 | **16** | 2.77 | |err fila| 0.42 |
| fork | low | tube_fork | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| fork | random | tube_fork | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| fork | real | tube_fork | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.30 | **—** | — | |err fila| 1.00 |
| precursor | high | oracle | 0.05 (0.01–0.24) | 0.00 | 0.25 | 1.00 | 1.00 | 1.00 | 1.00 | **2** | 0.0433 | — |
| precursor | high | tube_anticip | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| precursor | low | oracle | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.60 | **—** | — | — |
| precursor | low | tube_anticip | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| precursor | random | oracle | 0.05 (0.01–0.24) | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.20 | **—** | — | — |
| precursor | random | tube_anticip | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| precursor | real | oracle | 0.15 (0.05–0.36) | 0.15 | 0.10 | 0.05 | 0.75 | 1.00 | 1.00 | **8** | 0.593 | — |
| precursor | real | tube_anticip | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| precursor_persist | high | oracle | 0.00 (0.00–0.16) | 0.00 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | **2** | 0.0433 | — |
| precursor_persist | high | tube_anticip | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.30 | 1.00 | 1.00 | **8** | 0.692 | — |
| precursor_persist | low | oracle | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.55 | **—** | — | — |
| precursor_persist | low | tube_anticip | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| precursor_persist | random | oracle | 0.15 (0.05–0.36) | 0.15 | 0.15 | 0.15 | 0.15 | 0.05 | 0.25 | **—** | — | — |
| precursor_persist | random | tube_anticip | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **—** | — | — |
| precursor_persist | real | oracle | 0.15 (0.05–0.36) | 0.10 | 0.10 | 0.05 | 0.55 | 1.00 | 1.00 | **8** | 0.593 | — |
| precursor_persist | real | tube_anticip | 0.00 (0.00–0.16) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.85 | **—** | — | — |

Lectura de la columna teoría (solo pca): la dosis donde la teoría de matrices aleatorias predice traslape² = 0.5 bajo un bulto isotrópico de dimensión d_eff. LOD empírico ≈ teoría: la matriz se comporta como ruido isotrópico en esa familia. LOD empírico ≫ teoría: efecto matriz (estructura real que compite con la dirección). LOD empírico ≪ teoría: la familia vive en un subespacio más limpio que el promedio.

Conversión a nats con `captures/20261023T165139Z_qwen2.5-7b_T0.7_cal0_L14-20_s42_cap13-28/belenc_fisher/20261023T165510Z/fisher.json`: KL ≈ ½·a²·σ_u²·uᵀGu, mediana por familia de dirección (respuesta lineal, belenc_fisher.py).
