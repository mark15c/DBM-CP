# Resumen ejecutivo: impacto económico del modelo propuesto

**Base de cálculo:** muestra fuera de tiempo (OOT, abril y mayo de 2024), con **10,955 renovaciones** y desempeño observado. Se usa el modelo de Manolo sin cambios (WOE + Regresión Logística, 18 variables) con su política **PD < 20%**, fijada en VALIDATION. Los montos corresponden a esta muestra y **no se extrapolan a toda la cartera**, porque no conocemos la tasa de muestreo. Para escalarlos, se reportan también por cada 1,000 renovaciones.

Se comparan tres escenarios sobre la misma población:

| | Escenario | Aprobación | Tasa de malos aprobados | Utilidad neta |
|---|---|---:|---:|---:|
| A | Situación actual (aprobar a todos) | 100.00% | 14.13% | S/ 2,159,405 |
| B | Modelo actual del banco, con el mismo volumen aprobado | 73.47% | 9.24% | S/ 3,028,275 |
| C | **Modelo propuesto, PD < 20%** | **73.47%** | **6.96%** | **S/ 3,702,635** |

---

## Título propuesto para la lámina

> **El modelo propuesto suma S/ 1.54 M de utilidad: S/ 0.87 M por tener una política de corte y S/ 0.67 M por ordenar mejor el riesgo que el modelo actual**

## Tres KPIs

| KPI | Frente a la situación actual (A) | Frente al modelo del banco (B) |
|---|---:|---:|
| **Pérdida evitada** | **S/ 2,914,600** (988 malos no renovados) | S/ 542,800 menos pérdida por malos |
| **Utilidad neta adicional** | **+S/ 1,543,230 (+71.5%)** · S/ 140,871 por cada 1,000 renovaciones | **+S/ 674,360 (+22.3%)** · S/ 61,557 por cada 1,000 renovaciones |
| **Tasa de malos de la cartera aprobada** | **14.13% → 6.96% (−7.17 pp)** | 9.24% → 6.96% (−2.28 pp) |

Cómo se descompone el ahorro total: **S/ 1,543,230 = S/ 868,870 (tener una política de corte, B − A) + S/ 674,360 (usar un mejor modelo, C − B).** El segundo monto es el que se atribuye exclusivamente al modelo de Manolo.

El % se reporta solo como dato secundario, porque depende del tamaño de la utilidad base: en la sensibilidad llega a +4,500% cuando esa base se acerca a cero. El KPI principal va en soles.

## Speech (unos 35 segundos)

> "Llevamos el modelo a soles. Tomamos once mil renovaciones de abril y mayo de 2024, un periodo que el modelo nunca vio, y comparamos tres escenarios. Si el banco renueva a todos, gana 2.2 millones. Con su modelo actual, aprobando tres de cada cuatro clientes, gana 3 millones. Con nuestro modelo, aprobando exactamente la misma cantidad, gana 3.7 millones. La tasa de malos baja de 14 a 7%. De los 1.5 millones adicionales, 670 mil soles se explican solo por ordenar mejor el riesgo. Además, el corte de 20% coincide con el punto de equilibrio económico, que está en 19.5%."

## Supuestos y fuente

La base está anonimizada: no trae monto, tasa, plazo, saldo, días de atraso ni recuperaciones. Por eso **todos los parámetros económicos son supuestos**. Están centralizados en `SUPUESTOS`, dentro de `impacto/impacto_economico.py`.

| Parámetro | Valor | Fuente |
|---|---:|---|
| Monto por crédito | S/ 5,000 | Supuesto (la base no trae monto) |
| TEA | 45% | Supuesto (la base no trae tasa) |
| Plazo | 12 meses | Supuesto (la base no trae plazo) |
| Costo de fondeo | 7% anual | Supuesto |
| Costo operativo | 12% anual | Supuesto |
| Saldo medio | 55% del monto | Supuesto (crédito amortizable) |
| LGD | 70% | Supuesto (la base no trae recuperaciones) |
| Factor EAD | 80% del monto | Supuesto |
| Costo de cobranza | S/ 150 por malo | Supuesto |
| Definición de malo | TARGET = 1 equivale a default con pérdida | Supuesto (la definición del TARGET no fue publicada) |
| **Ganancia por bueno aprobado** | **S/ 715** | 5,000 × 55% × (45% − 7% − 12%) × 1 año |
| **Pérdida por malo aprobado** | **S/ 2,950** | 5,000 × 80% × 70% + 150 |
| **PD de equilibrio** | **19.5%** | 715 / (715 + 2,950) |

Datos que salen del repositorio: el TARGET observado, la partición temporal (`config/modelo_config.json`), las PD del artefacto `model/modelo_logistico_woe.pkl` y el score del banco (`PREDICCION_MODELO_ANTERIOR`).

## Respuestas preparadas para el jurado

**1. "¿No hay sesgo porque solo ven a los clientes aprobados?"**
La base contiene solo clientes que el banco renovó, y de todos ellos se conoce el desempeño. Por eso, para el 26.5% que el modelo propuesto dejaría de renovar, la pérdida evitada y la ganancia sacrificada **se midieron, no se estimaron**: son clientes reales con resultado conocido. El límite está en otra parte: no vemos a quienes el proceso actual nunca renovó. El resultado es válido para la población que hoy se renueva. No afirmamos nada sobre solicitantes que hoy se rechazan, porque para ellos haría falta reject inference.

**2. "¿De dónde salen la LGD y los márgenes?"**
Son supuestos, porque la base está anonimizada, y los declaramos así. Por eso medimos la sensibilidad con ±20% en LGD y en margen (9 combinaciones). En **las 9 combinaciones el modelo propuesto gana más que el modelo del banco, entre S/ 545 mil y S/ 804 mil**. El ahorro total frente a aprobar a todos va de S/ 0.72 M a S/ 2.37 M. La PD de equilibrio se mueve entre 14% y 26%, y el corte de 20% queda en el centro de ese rango. Con los parámetros reales del banco basta cambiar el diccionario `SUPUESTOS` y volver a ejecutar.

**3. "¿El resultado es estable en el tiempo?"**
Sí. Con el umbral fijo PD < 20%, mes a mes:

| Mes | Muestra | Tasa de malos, modelo propuesto | Tasa de malos, modelo banco | Ahorro total | Ahorro por mejor modelo | Ahorro por cada 1,000 |
|---|---|---:|---:|---:|---:|---:|
| 2024-02 | VALIDATION | 8.00% | 10.32% | S/ 795,300 | S/ 326,185 | S/ 151,112 |
| 2024-03 | VALIDATION | 8.04% | 10.39% | S/ 758,005 | S/ 348,175 | S/ 142,482 |
| 2024-04 | OOT | 7.25% | 9.65% | S/ 761,595 | S/ 355,505 | S/ 140,386 |
| 2024-05 | OOT | 6.66% | 8.95% | S/ 781,635 | S/ 337,180 | S/ 141,344 |

Bootstrap en OOT (1,000 réplicas, umbrales congelados). Intervalos de confianza al 95%:
- Ahorro total: **S/ 1.36 M a S/ 1.73 M**.
- Ahorro por mejor modelo: **S/ 0.53 M a S/ 0.82 M**.
- Ninguna réplica dio ahorro negativo.

## Comparación de modelos: discriminación por periodo

| Modelo | Variables | Gini TRAIN | Gini VALID | Gini OOT | AUC TRAIN | AUC VALID | AUC OOT | KS TRAIN | KS VALID | KS OOT | Caída de Gini TRAIN→OOT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Modelo actual del banco | n.d. | 0.476 | 0.420 | 0.445 | 0.738 | 0.710 | 0.722 | 0.339 | 0.298 | 0.324 | 3.1 pts |
| **Regresión Logística WOE** | **18** | **0.599** | **0.582** | **0.593** | **0.800** | **0.791** | **0.796** | **0.445** | **0.432** | **0.450** | **0.6 pts** |
| CatBoost (benchmark) | 45 | 0.726 | 0.616 | 0.609 | 0.863 | 0.808 | 0.805 | 0.554 | 0.461 | 0.463 | 11.7 pts |

Qué muestra la tabla:
- La logística WOE supera al modelo actual en **+14.8 pts de Gini en OOT**. Es la brecha que explica los S/ 674 mil del "mejor modelo".
- Queda a solo **1.6 pts de CatBoost**, usando 18 variables en lugar de 45.
- Es el modelo más estable: pierde 0.6 pts de TRAIN a OOT, frente a 11.7 pts de CatBoost.

Fuentes:
- Modelo actual del banco: calculado sobre `PREDICCION_MODELO_ANTERIOR`. Para este modelo, "TRAIN" es solo el periodo, porque no fue entrenado con esta partición. Hay 3 casos sin score en TRAIN y 2 en VALIDATION.
- Logística: calculada con el modelo guardado de Manolo; reproduce exactamente sus métricas.
- CatBoost: cifras oficiales del notebook 03. Ese modelo no se guardó. Al reentrenar su configuración exacta (semilla 42) en otro equipo, las cifras difieren menos de 0.1 pts de Gini en VALIDATION y OOT (0.6165 y 0.6083). CatBoost varía según los hilos de CPU; ninguna conclusión cambia.

Gráfico: `comparacion_modelos.png`.

## Variables finales: patrones de riesgo coherentes

> **"Las variables finales muestran patrones de riesgo coherentes."**

**Cómo se pasó de 45 a 18 variables** (notebook 01 y lámina 7 del deck): 45 predictores utilizables → 42 tras el filtro de cobertura y estabilidad → 28 al quitar redundancia WOE → 23 al exigir señal y estabilidad → 20 en el modelo candidato → **18 finales** tras el Drop-One sobre VALIDATION.

**Monotonía del bad rate por bins** (calculada, `monotonia_variables.csv`):
- En desarrollo: **17 de 18** variables son monótonas (9 crecientes y 8 decrecientes).
- Fuera de tiempo (OOT): **15 de 18**.
- Las que no son monótonas en OOT tienen IV bajo: productos SSFF (0.045) y deuda promedio SSFF (0.031). La tercera, apalancamiento interno U1M, ya no era monótona en TRAIN.

**Tres variables fuertes** (gráfico `patrones_variables.png`). Se eligió la creciente de mayor IV en cada familia de riesgo:

| Variable | IV | Bad rate por bin, TRAIN → (OOT) |
|---|---:|---|
| Mora interna, 6 meses (`MEDIDA1_MORA_INTERNA_U6M`) | 0.47 | <4: 10.7% (8.6%) · 4–7: 16.9% (14.3%) · 7–13: 27.7% (23.9%) · ≥13: **48.4% (49.4%)** |
| Apalancamiento SSFF, 36 meses (`MEDIDA1_APALANCAMIENTO_SSFF_U36M`) | 0.12 | <0.26: 6.7% (7.1%) · 0.26–0.64: 12.0% (10.4%) · 0.64–0.86: 17.1% (15.4%) · ≥0.86: **20.7% (18.4%)** |
| Calificación SSFF, 60 meses (`MEDIDA1_CALIFICACION_SSFF_U60M`) | 0.12 | <1: 12.1% (10.9%) · 1–4: 19.7% (17.9%) · ≥4: **24.4% (19.7%)** |

Las tres crecen de forma ordenada en desarrollo y el orden se repite fuera de tiempo. En mora interna U6M, el bad rate del tramo más alto es 4.5 veces el del más bajo. La segunda variable más fuerte por IV es mora interna U1M (0.30), pero se dejó fuera para no repetir la familia de mora interna. Si prefieres el top 3 puro por IV, se cambia en `VARIABLES_LAMINA`.

### Hallazgo para Manolo: un centinela no se trata como especial

En `MEDIDA3_APALANCAMIENTO_INTERNO_U1M`, el valor centinela `333333344` **no llega a su bin especial al aplicar el modelo**.
- **Causa:** la variable es float32. `scorecardpy` guardó el valor especial como el texto `"3.3333334e+08"`, que no coincide con el dato al aplicar el modelo.
- **Qué pasa:** esos clientes caen en el bin `[1.28, inf)` y reciben su WOE (+0.283, bin con bad rate de 20.2%) en lugar del WOE especial (−0.079, bad rate de 15.0%).
- **A quién afecta:** 2,964 clientes en TRAIN (7.7%), 775 en VALIDATION (7.3%) y 761 en OOT (6.9%). Su bad rate real es de 12.6% a 15.0%.
- **Efecto:** su logit sube +0.29. Una PD de 15% pasa a cerca de 19%, así que el modelo sobreestima su riesgo.
- **Qué sigue siendo válido:** el entrenamiento de Manolo usó la misma función, así que el modelo es consistente y las métricas reportadas son reales.
- **Qué no se cumple:** lo que dicen el notebook 02 y la lámina 6 del deck ("centinelas conservados como bins independientes") no es cierto para esta variable.
- **Qué hacer:** corregirlo exige reentrenar, lo que queda fuera de este trabajo y es decisión de Manolo. Conviene saberlo antes de que el jurado de Model Risk lo encuentre.

## Calibración por bandas de PD (riesgo estimado frente a riesgo observado)

Es una réplica en código del backtesting de Manolo (notebook 01), calculada sobre las PD reales y verificada contra su tabla de VALIDATION. Gráfico: `calibracion_bandas.png`; datos: `calibracion_bandas.csv`.

| Muestra | Bandas sin diferencia significativa (independencia) | Vasicek con rho de 0.5% a 3% (todos los rho) | Vasicek con rho ≥ 2% |
|---|---:|---:|---:|
| VALIDATION | 6 de 7 | **7 de 7** | 7 de 7 |
| OOT | 2 de 7 | 4 de 7 | 7 de 7 |

**Ojo con OOT (Manolo no lo reportó):** en OOT la PD **sobreestima el riesgo en las 7 bandas**, con brechas de +0.05 a +6.19 pp. Es la misma señal de la calibración global, que muestra +2.4 pp. El error va hacia el lado conservador y no afecta el ordenamiento (el Gini OOT es estable). Su único efecto económico es rechazar algunos clientes buenos de más, que es por qué en OOT aprobar el 80% rinde S/ 33 mil más. Si el jurado pregunta, la respuesta es: "el ranking se mantiene y la política se fijó en VALIDATION; el nivel de PD se debe monitorear y, si la brecha persiste, recalibrar el intercepto sin tocar el ranking".

## Otros escenarios de corte (OOT, umbrales fijados en VALIDATION)

| Escenario | Umbral de PD | Aprobación | Tasa de malos aprobados | Utilidad neta | Ahorro frente a actual |
|---|---:|---:|---:|---:|---:|
| C. PD < 20% (Manolo) | 20.00% | 73.47% | 6.96% | S/ 3,702,635 | S/ 1,543,230 |
| D. Óptimo económico elegido en VALIDATION (73%) | 19.20% | 71.81% | 6.71% | S/ 3,689,785 | S/ 1,530,380 |
| E. 90% de aprobación | 36.98% | 89.09% | 9.71% | S/ 3,503,980 | S/ 1,344,575 |
| F. 80% de aprobación | 23.46% | 78.81% | 7.70% | S/ 3,736,085 | S/ 1,576,680 |

En VALIDATION, el óptimo económico (73% de aprobación, PD < 19.2%) prácticamente coincide con la política de Manolo (74.6%, PD < 20%): son dos criterios independientes que llegan al mismo corte.

En OOT, aprobar el 80% habría dado S/ 33 mil más que PD < 20%. La diferencia no es significativa: el intervalo al 95% de un bootstrap pareado va de −S/ 35 mil a +S/ 94 mil. Probablemente se debe a que el modelo sobreestima la PD en 2.4 pp en OOT. Esto no justifica mover el corte con OOT. Lo que corresponde es monitorear la calibración, algo que ya está en el plan de monitoreo.

## Limitaciones

1. **Todos los parámetros económicos son supuestos.** Monto, tasa, LGD, costos y definición de malo no vienen en la base. Las cifras son órdenes de magnitud, y la sensibilidad acota su rango.
2. **Monto uniforme por crédito.** Sin montos reales, cada cliente pesa lo mismo. Si los clientes riesgosos tuvieran montos distintos, el resultado cambiaría.
3. **La escala corresponde a la muestra**: 2 meses y 10,955 renovaciones, con una tasa de muestreo desconocida. No se extrapola a la cartera total.
4. **El resultado vale para la población que hoy se renueva.** No cubre a los solicitantes que el proceso actual rechaza (no hay reject inference).
5. **La comparación con el banco es a igual volumen aprobado.** El score del banco no está calibrado (su promedio es 30% frente a una tasa de malos de 14%), así que se compara por ranking y no por umbral. Hay 2 casos sin score en VALIDATION.
6. **El bootstrap asume clientes independientes**, así que ignora la correlación entre defaults (el análisis Vasicek de Manolo sí la considera). El intervalo real es algo más ancho.
7. **No se incluye el valor de largo plazo del cliente.** Rechazar la renovación de un buen cliente puede costar más que el margen de un solo crédito.
