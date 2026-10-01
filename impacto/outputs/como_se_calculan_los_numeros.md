# Cómo se construyen los números del gráfico de impacto económico

Documento de respaldo para el gráfico `barras_impacto.png` ("S/ 1.54 M más de utilidad: S/ 0.87 M por tener una política y S/ 0.67 M por un mejor modelo"). Explica, paso a paso, de dónde sale cada cifra, para poder defenderla ante el jurado.

**En una frase:** cada barra es un **conteo de clientes reales** (buenos o malos, aprobados o rechazados) multiplicado por un **valor unitario en soles**. Los conteos salen del modelo y de la base; los valores unitarios son supuestos declarados.

---

## 1. Los ingredientes

### 1.1 La muestra: 10,955 renovaciones reales de OOT

| Dato | Valor | De dónde sale |
|---|---:|---|
| Periodo | abril y mayo de 2024 (`CODMES` 202404 y 202405) | Partición de Manolo, `config/modelo_config.json` |
| Clientes | **10,955** | Filas de la base en esos dos meses |
| Malos (`TARGET = 1`) | **1,548** (14.13%) | Desempeño observado, no estimado |
| Buenos (`TARGET = 0`) | **9,407** | 10,955 − 1,548 |

Se usa OOT porque es un periodo que el modelo **nunca vio**: no se usó para entrenar ni para elegir el corte. Es la prueba más honesta de cómo le iría al banco.

### 1.2 La PD de cada cliente: el modelo de Manolo, sin tocarlo

1. Se carga el modelo guardado `model/modelo_logistico_woe.pkl` (WOE + Regresión Logística, 18 variables) con la función de Manolo `src/predict.py`. **No se reentrena nada.**
2. Se aplica a los 10,955 clientes y se obtiene una PD (probabilidad de ser malo) para cada uno.
3. **Verificación:** con esas PD, el AUC en OOT da 0.796370 y el KS 0.449504. Son exactamente los valores que reportó Manolo, así que estamos usando su mismo modelo.

### 1.3 El valor de cada cliente en soles (supuestos)

La base está anonimizada: no trae monto, tasa, plazo ni recuperaciones. Por eso los valores unitarios se construyen con supuestos declarados:

**Ganancia por cliente bueno aprobado = S/ 715**

```
monto × saldo medio × (TEA − costo de fondeo − costo operativo) × plazo en años
5,000 × 55%         × (45%  − 7%              − 12%)            × 1
= 5,000 × 0.55 × 0.26 = S/ 715
```

Es el margen financiero que deja un crédito que se paga bien durante un año, calculado sobre el saldo promedio (el crédito se va amortizando).

**Pérdida por cliente malo aprobado = S/ 2,950**

```
monto × EAD × LGD + costo de cobranza
5,000 × 80% × 70% + 150
= 2,800 + 150 = S/ 2,950
```

Es lo que se pierde cuando un crédito cae en default: parte del saldo expuesto no se recupera, y además se gasta en cobranza.

**Cliente rechazado = S/ 0.** Si no se le presta, ni gana ni pierde.

> Un malo "cuesta" lo mismo que lo que ganan unos 4 buenos (2,950 / 715 ≈ 4.1). Por eso filtrar malos vale tanto.

---

## 2. Los tres escenarios

Los tres escenarios se evalúan sobre **los mismos 10,955 clientes y su mismo desempeño real**. Lo único que cambia es a quién se le aprueba la renovación.

| | Escenario | Regla de aprobación | Aprobados |
|---|---|---|---:|
| **A** | Situación actual | Aprobar a todos. La base solo contiene clientes que el banco renovó, así que esto es lo que efectivamente pasó. | 10,955 (100%) |
| **B** | Modelo actual del banco | Ordenar por su score `PREDICCION_MODELO_ANTERIOR` y aprobar a los **8,049** de menor riesgo | 8,049 (73.47%) |
| **C** | Modelo propuesto | Aprobar si **PD < 20%**. Es la política de Manolo, fijada en VALIDATION (febrero–marzo 2024), antes de mirar OOT. | 8,049 (73.47%) |

**¿Por qué B aprueba exactamente 8,049?** Para comparar los dos modelos **a igual volumen**: ambos aprueban la misma cantidad de clientes y la única diferencia es *a quiénes* eligen. No se puede usar el mismo umbral de 20% para el modelo del banco porque su score no está calibrado: su promedio es 30%, cuando la tasa real de malos es 14%.

---

## 3. Del modelo a los conteos

Al cruzar la decisión de cada escenario con el `TARGET` observado de cada cliente, se obtiene esta tabla. **Es la base de todo el gráfico.**

| Escenario | Buenos aprobados | Malos aprobados | Buenos rechazados | Malos rechazados | Tasa de malos de los aprobados |
|---|---:|---:|---:|---:|---:|
| A. Aprobar a todos | 9,407 | 1,548 | 0 | 0 | 14.13% |
| B. Modelo del banco | 7,305 | 744 | 2,102 | 804 | 9.24% |
| C. Modelo propuesto | **7,489** | **560** | 1,918 | **988** | **6.96%** |

Lectura rápida:
- Ambos modelos rechazan a 2,906 clientes.
- **El nuestro rechaza 988 malos; el del banco, solo 804.**
- Con el mismo número de aprobados, el modelo propuesto deja pasar **184 malos menos** (744 − 560), y en su lugar aprueba a 184 buenos más.

---

## 4. De los conteos a los soles: cada barra del gráfico

Cada barra es un conteo de la tabla anterior multiplicado por S/ 715 o S/ 2,950.

### Panel "Ganancia de buenos" = buenos aprobados × S/ 715

| Barra | Cálculo | Resultado | En el gráfico |
|---|---|---:|---:|
| Actual | 9,407 × 715 | S/ 6,726,005 | S/ 6.73 M |
| Modelo banco | 7,305 × 715 | S/ 5,223,075 | S/ 5.22 M |
| Modelo propuesto | 7,489 × 715 | S/ 5,354,635 | S/ 5.35 M |

### Panel "Pérdida por malos" = malos aprobados × S/ 2,950

| Barra | Cálculo | Resultado | En el gráfico |
|---|---|---:|---:|
| Actual | 1,548 × 2,950 | S/ 4,566,600 | S/ 4.57 M |
| Modelo banco | 744 × 2,950 | S/ 2,194,800 | S/ 2.19 M |
| Modelo propuesto | 560 × 2,950 | S/ 1,652,000 | S/ 1.65 M |

### Panel "Utilidad neta" = ganancia de buenos − pérdida por malos

| Barra | Cálculo | Resultado | En el gráfico |
|---|---|---:|---:|
| Actual | 6,726,005 − 4,566,600 | S/ 2,159,405 | S/ 2.16 M |
| Modelo banco | 5,223,075 − 2,194,800 | S/ 3,028,275 | S/ 3.03 M |
| Modelo propuesto | 5,354,635 − 1,652,000 | S/ 3,702,635 | S/ 3.70 M |

---

## 5. La descomposición del titular: S/ 1.54 M = S/ 0.87 M + S/ 0.67 M

| Tramo | Cálculo | Resultado | Qué significa |
|---|---|---:|---|
| **Valor de tener una política** (B − A) | 3,028,275 − 2,159,405 | **S/ 868,870** | Lo que gana el banco solo por filtrar, aunque sea con el modelo que ya tiene |
| **Valor del mejor modelo** (C − B) | 3,702,635 − 3,028,275 | **S/ 674,360** | Lo que se gana solo por ordenar mejor el riesgo, aprobando la misma cantidad de clientes |
| **Total** (C − A) | 3,702,635 − 2,159,405 | **S/ 1,543,230** | La suma de los dos tramos |

**La cifra de S/ 674,360 se puede verificar a mano.** Con el mismo volumen, el modelo propuesto cambia 184 malos por 184 buenos. Cada intercambio evita una pérdida de S/ 2,950 y suma una ganancia de S/ 715:

```
184 × (2,950 + 715) = 184 × 3,665 = S/ 674,360
```

### La otra forma de verlo: pérdida evitada menos ganancia sacrificada

Frente a aprobar a todos, rechazar clientes tiene dos efectos:

| | Modelo banco | Modelo propuesto |
|---|---:|---:|
| Pérdida evitada (malos rechazados × 2,950) | 804 × 2,950 = S/ 2,371,800 | 988 × 2,950 = **S/ 2,914,600** |
| Ganancia sacrificada (buenos rechazados × 715) | 2,102 × 715 = S/ 1,502,930 | 1,918 × 715 = S/ 1,371,370 |
| **Ahorro neto** | **S/ 868,870** | **S/ 1,543,230** |

Da lo mismo por los dos caminos: el código verifica en cada ejecución que `utilidad actual + pérdida evitada − ganancia sacrificada = utilidad del escenario`.

---

## 6. Preguntas probables del jurado y cómo responderlas

**"¿Las pérdidas las calcularon con la PD del modelo?"**
No. La PD **solo decide a quién aprobar**. Las ganancias y pérdidas se calculan con el **desempeño real observado** (`TARGET`) de cada cliente. Por eso el resultado no depende de que la PD esté bien calibrada, solo de que ordene bien.

**"¿Por qué la ganancia de buenos baja con el modelo? ¿No es peor?"**
Porque todo filtro rechaza también a algunos buenos (1,918 en nuestro caso). Es el costo de la política, y se ve en la barra. Aun así, por cada sol de ganancia sacrificada se evitan más de dos soles de pérdida (2,914,600 frente a 1,371,370), así que la utilidad neta sube.

**"¿Por qué la situación actual es 'aprobar a todos'?"**
Porque la base solo contiene clientes a los que el banco sí renovó, y conocemos el resultado de todos. Aprobar a todos reproduce exactamente lo que ocurrió.

**"¿No eligieron el corte de 20% para que salga bien en OOT?"**
No. El corte lo fijó Manolo en VALIDATION (febrero–marzo 2024) con el objetivo de bad rate de 8% del reto, y se aplicó sin cambios en OOT. Además, en VALIDATION el corte que maximiza la utilidad es aprobar el 73%, prácticamente lo mismo. Con estos supuestos, el punto de equilibrio económico es PD = 715 / (715 + 2,950) = **19.5%**, casi igual al corte de 20%.

**"¿De dónde salen S/ 715 y S/ 2,950?"**
Son supuestos declarados, porque la base está anonimizada. Por eso se hizo sensibilidad con ±20% en LGD y en margen (9 combinaciones). En **todas**, el modelo propuesto le gana al del banco por entre S/ 545 mil y S/ 804 mil. Con los parámetros reales del banco basta cambiar el diccionario `SUPUESTOS` y volver a ejecutar.

**"¿Es plata de todo el banco?"**
No. Es la muestra OOT: 10,955 renovaciones en dos meses. No se extrapola porque no conocemos la tasa de muestreo. Para escalarlo se reporta por cada 1,000 renovaciones: **S/ 140,871 de ahorro total**, de los cuales **S/ 61,557 son atribuibles al mejor modelo**.

**"¿Qué tan seguro es el número?"**
- **Bootstrap (1,000 réplicas):** el ahorro total va de S/ 1.36 M a S/ 1.73 M, y el del mejor modelo de S/ 0.53 M a S/ 0.82 M (intervalos al 95%). Ninguna réplica dio negativo.
- **Mes a mes:** el ahorro es estable, entre S/ 758 mil y S/ 795 mil por mes.

**"¿Por qué no se cuentan los intereses que el malo pagó antes de caer?"**
Por prudencia: asumimos que un malo no deja ningún ingreso. Si se contaran, la pérdida por malo sería menor.

---

## 7. Cómo reproducirlo

```powershell
.\.venv\Scripts\python.exe impacto\impacto_economico.py
```

- **Tabla completa de escenarios:** `impacto/outputs/escenarios.csv`, en la muestra `OOT`, escenarios A, B y C.
- **Lógica:** `impacto/impacto_economico.py`. Las funciones clave son `economia_cliente` (S/ 715 y S/ 2,950), `evaluar_politica` (conteos y soles) y `decisiones` (reglas A, B y C).
- **Supuestos:** están todos en el diccionario `SUPUESTOS` al inicio del archivo, cada uno con su fuente.

---

## 8. Límites que conviene reconocer si preguntan

1. Los valores en soles dependen de supuestos. **El ranking y los conteos de clientes, no:** son datos reales.
2. Se asume el mismo monto para todos los créditos, porque la base no trae montos.
3. El resultado vale para la población que hoy se renueva. No cubre a los solicitantes que el banco rechaza hoy.
4. Un malo se trata como default con pérdida completa, porque la definición exacta del `TARGET` no fue publicada.
