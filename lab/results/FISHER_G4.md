# BELENC-Fisher v0.1.1-fisher — `20261023T165139Z_qwen2.5-7b_T0.7_cal0_L14-20_s42_cap13-28` · fila de ancla del turno 7

calibration_id `10e0b9192173750e` · draws [0, 1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 15, 16, 18, 19] · top-K 64 · masa top-K mediana 0.999

| capa | τ (nats) | τ TRUTH | τ FIC | cobertura | enriquecimiento E | fragilidad latente | χ high | χ low | χ random | χ real | χ L | χ R | L/rand | R/rand |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | 0.0128 | 0.0585 | 0.0118 | 0.0086 | 2.2 | 0.9914 | 0.00141 | 7.26e-06 | 6.04e-06 | 0.000618 | 0.000443 | 0.00331 | 73.46 | 547.81 |
| 17 | 0.0479 | 0.128 | 0.0413 | 0.0098 | 2.5 | 0.9902 | 0.0359 | 8.18e-05 | 1.73e-05 | 0.00225 | 0.000548 | 0.00606 | 31.64 | 349.74 |
| 21 | 0.256 | 0.484 | 0.222 | 0.0415 | 10.6 | 0.9585 | 0.0108 | 4.58e-05 | 7.17e-06 | 0.00926 | 0.000137 | 0.000807 | 19.06 | 112.65 |

## Linealidad del parche 5 → 2 (KL medida / predicha ½α²δᵀGδ)

| capa | α=0.05 | α=0.1 | α=0.25 | α=0.5 | α=1 | rango lineal α | ‖δ‖_G (√nats) |
|---|---|---|---|---|---|---|---|
| 14 | 1.16 | 1.06 | 0.91 | 0.75 | 0.47 | 0.25 | 2.02 |
| 17 | 0.98 | 1.01 | 0.91 | 0.70 | 0.38 | 0.25 | 2.98 |
| 21 | 1.00 | 1.01 | 0.82 | 0.57 | 0.30 | 0.25 | 3.71 |

χ = KL por σ² natural a lo largo de la dirección (½σ²uᵀGu). L/rand y R/rand cerca de 1: los ejes de CRAD no son direcciones de respuesta. E = cobertura / (dim/d): cuántas veces más sensibilidad cae en lo que el modelo varía que en un subespacio al azar. Pasa `fisher.json` a belenc.py con --fisher-json para leer los LOD en nats.

**Una captura, una semilla: una sonda, no una prueba.**
