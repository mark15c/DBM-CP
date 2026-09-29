# Entrega final — Riesgo crediticio Compartamos

## Entrega consolidada

Esta es la única carpeta que debe entregarse o ejecutarse. Contiene una sola
versión de cada notebook y no incluye variantes históricas ni nombres con
`_corregido`.

| Orden | Archivo | Propósito |
|---:|---|---|
| 1 | `notebooks/01_desarrollo_modelo_logistico.ipynb` | Desarrollo y validación del modelo WOE + Regresión Logística. Incluye selección de variables, backtesting de calibración, sensibilidad Vasicek y la definición de la política. |
| 2 | `notebooks/02_modelo_logistico_final.ipynb` | Flujo reproducible final. Reconstruye el modelo, confirma las métricas, ejecuta la política y genera el artefacto de producción. |
| 3 | `notebooks/03_benchmark_catboost.ipynb` | Benchmark independiente de CatBoost; documenta la comparación y no sustituye al modelo seleccionado. |

El modelo seleccionado se mantiene sin cambios: **WOE + Regresión Logística con 18 variables**. La política se mantiene sin cambios: **habilitar renovación si `PD < 20%`**.

## Estructura

```text
Entrega_Compartamos_Final/
├── config/modelo_config.json
├── data/raw/base_analytics_lab_061628.parquet
├── model/                         # El notebook 02 guarda aquí el .pkl
├── notebooks/
│   ├── 01_desarrollo_modelo_logistico.ipynb
│   ├── 02_modelo_logistico_final.ipynb
│   └── 03_benchmark_catboost.ipynb
├── src/predict.py
├── requirements.txt
└── presentacion_final_compartamos.pptx
```

Las rutas se resuelven automáticamente si los notebooks se abren desde la raíz del proyecto o desde la carpeta `notebooks/`. No se deben mover archivos de esta estructura.

## Requisitos

- Python 3.11.9
- El archivo `data/raw/base_analytics_lab_061628.parquet` debe permanecer incluido.
- `requirements.txt` fija una combinación compatible de dependencias. En particular, no debe actualizarse Pandas a la versión 3 o superior, porque esa versión deja de ser compatible con `scorecardpy==0.1.9.7`.

Desde la raíz de esta carpeta, crear un entorno e instalar las dependencias:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Orden exacto de ejecución

1. Abrir `notebooks/01_desarrollo_modelo_logistico.ipynb`.
2. Elegir **Restart Kernel and Run All Cells** y esperar que termine sin errores.
3. Abrir `notebooks/02_modelo_logistico_final.ipynb`.
4. Elegir **Restart Kernel and Run All Cells**. Esta ejecución crea `model/modelo_logistico_woe.pkl` y verifica su reproducibilidad.
5. Confirmar que la celda final de reproducibilidad reporte una diferencia máxima igual a `0` o numéricamente despreciable (por ejemplo, del orden de `1e-15`).
6. Abrir `notebooks/03_benchmark_catboost.ipynb`.
7. Elegir **Restart Kernel and Run All Cells**. Este último paso es de comparación; no modifica el modelo ni la política.

No se ejecutan los notebooks fuera de orden ni se usa OOT para recalibrar la política. La regla metodológica es: **TRAIN aprende → VALIDATION decide → OOT confirma**.

## Resultados de referencia

| Métrica | TRAIN | VALIDATION | OOT |
|---|---:|---:|---:|
| Gini — Logística WOE | 0.5990 | 0.5816 | 0.5927 |
| Gini — CatBoost | 0.7262 | 0.6162 | 0.6091 |

Con la política `PD < 20%`:

| Muestra | Aprobación | Bad Rate aprobados | Malos en rechazados |
|---|---:|---:|---:|
| VALIDATION | 74.56% | 8.02% | 59.99% |
| OOT | 73.47% | 6.96% | 63.82% |

CatBoost queda como benchmark: mejora el Gini OOT en aproximadamente 1.64 puntos, pero no reemplaza la logística por criterios de explicabilidad, auditabilidad y facilidad de implementación.

## Artefacto de producción

Al finalizar el notebook 02, el artefacto será:

```text
model/modelo_logistico_woe.pkl
```

`src/predict.py` contiene las funciones `cargar_modelo()` y `predecir_pd()` para aplicar el artefacto a una base nueva que contenga las 18 variables requeridas.
