"""Impacto económico del modelo de Manolo (WOE + Regresión Logística, política PD < 20%).

No reentrena ni modifica nada del trabajo original: carga `model/modelo_logistico_woe.pkl`
con `src/predict.py` y lo aplica a VALIDATION y OOT.

Regla metodológica (la misma de Manolo): TRAIN aprende → VALIDATION decide → OOT confirma.
Todo umbral de PD (óptimo económico, 90 %, 80 %) se fija en VALIDATION y se aplica sin
cambios en OOT. El P&L usa el TARGET observado; la PD solo se usa para ordenar y decidir.

Escenarios comparados (misma población, mismo TARGET observado):
    A. Situación actual  : aprobar a todos (la base solo contiene clientes renovados).
    B. Modelo del banco  : PREDICCION_MODELO_ANTERIOR, aprobando el mismo número de
                           clientes que la política de Manolo (comparación a igual volumen).
    C. Modelo propuesto  : PD < 20 % (política de Manolo, congelada en VALIDATION).
    Ahorro total (C − A) = valor de tener política (B − A) + valor del mejor modelo (C − B).

Uso:
    python impacto/impacto_economico.py
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import scorecardpy as sc
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "impacto" / "outputs"
sys.path.insert(0, str(ROOT / "src"))
from predict import cargar_modelo, predecir_pd  # noqa: E402  (código de Manolo, sin cambios)

# ============================================================
# SUPUESTOS (único lugar donde se definen)
# La base está anonimizada: no trae monto, tasa, plazo, saldo,
# días de atraso ni recuperaciones. Todo lo económico es supuesto.
# ============================================================
SUPUESTOS = {
    "monto": {"valor": 5_000, "fuente": "Supuesto: la base no trae monto desembolsado (S/ por crédito)"},
    "tea": {"valor": 0.45, "fuente": "Supuesto: TEA de referencia para crédito de renovación; la base no trae tasa"},
    "plazo_meses": {"valor": 12, "fuente": "Supuesto: la base no trae plazo"},
    "costo_fondeo": {"valor": 0.07, "fuente": "Supuesto: costo de fondeo anual"},
    "costo_operativo": {"valor": 0.12, "fuente": "Supuesto: costo operativo anual sobre saldo"},
    "saldo_medio": {"valor": 0.55, "fuente": "Supuesto: saldo medio de un crédito amortizable (% del monto)"},
    "lgd": {"valor": 0.70, "fuente": "Supuesto: pérdida dado el incumplimiento; la base no trae recuperaciones"},
    "ead": {"valor": 0.80, "fuente": "Supuesto: exposición al incumplimiento (% del monto)"},
    "costo_cobranza": {"valor": 150, "fuente": "Supuesto: costo de cobranza por crédito malo (S/)"},
    "definicion_malo": {"valor": "TARGET = 1", "fuente": "Supuesto: todo TARGET = 1 es un default con pérdida (definición no publicada)"},
}

# Métricas publicadas por Manolo (notebook 02, celdas 18 y 25). Si no se reproducen, se detiene.
REFERENCIA = {
    "TRAIN": {"AUC": 0.799507, "KS": 0.445362},
    "VALIDATION": {"AUC": 0.790788, "KS": 0.432179},
    "OOT": {"AUC": 0.796370, "KS": 0.449504, "Aprobacion_%": 73.47, "Bad_Rate_Aprobados_%": 6.96},
}
SCORE_BANCO = "PREDICCION_MODELO_ANTERIOR"


def valores(supuestos=SUPUESTOS):
    """Devuelve {parámetro: valor} a partir del diccionario de supuestos."""
    return {k: v["valor"] for k, v in supuestos.items()}


# ============================================================
# 1. PREDICCIONES DEL MODELO DE MANOLO
# ============================================================
def cargar_base():
    """Config del split, base completa y artefacto guardado del modelo (sin reentrenar)."""
    cfg = json.loads((ROOT / "config" / "modelo_config.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(ROOT / "data" / "raw" / "base_analytics_lab_061628.parquet")
    return cfg, df, cargar_modelo(ROOT / "model" / "modelo_logistico_woe.pkl")


def cargar_predicciones():
    """PD del modelo de Manolo y score del banco para TRAIN, VALIDATION y OOT.

    Usa el artefacto guardado (sin reentrenar) y el split de config/modelo_config.json.
    TRAIN solo se usa para comparar métricas de discriminación; la economía nunca lo usa.
    Devuelve un DataFrame con MUESTRA, CODMES, TARGET, PD, SCORE_BANCO.
    """
    cfg, df, artefacto = cargar_base()
    mes = df[cfg["time_column"]]

    partes = []
    for muestra, filtro in [("TRAIN", mes <= cfg["train_max"]),
                            ("VALIDATION", mes.isin(cfg["validation_months"])),
                            ("OOT", mes.isin(cfg["oot_months"]))]:
        base = df[filtro]
        pred = predecir_pd(base, artefacto)
        assert pred.index.equals(base.index), "predecir_pd cambió el orden de las filas"
        partes.append(pd.DataFrame({
            "MUESTRA": muestra,
            "CODMES": base[cfg["time_column"]].to_numpy(),
            "TARGET": base[cfg["target"]].astype(int).to_numpy(),
            "PD": pred["PD"].to_numpy(),
            "SCORE_BANCO": base[SCORE_BANCO].to_numpy(dtype=float),
        }))
    return pd.concat(partes, ignore_index=True), artefacto["cutoff_pd"]


def ks(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.max(tpr - fpr))


def verificar_reproduccion(datos, cutoff, tol=1e-6):
    """Compara AUC/KS y la política PD < cutoff contra lo publicado por Manolo.

    Lanza AssertionError si no coincide. Devuelve la tabla de métricas (incluye el score del banco).
    """
    filas = []
    for muestra, d in datos.groupby("MUESTRA", sort=False):
        ok = d["SCORE_BANCO"].notna()
        auc = roc_auc_score(d["TARGET"], d["PD"])
        auc_banco = roc_auc_score(d.loc[ok, "TARGET"], d.loc[ok, "SCORE_BANCO"])
        fila = {
            "Muestra": muestra, "N": len(d), "Bad_Rate_%": d["TARGET"].mean() * 100,
            "AUC": auc, "Gini": 2 * auc - 1, "KS": ks(d["TARGET"], d["PD"]),
            "AUC_banco": auc_banco, "Gini_banco": 2 * auc_banco - 1,
            "KS_banco": ks(d.loc[ok, "TARGET"], d.loc[ok, "SCORE_BANCO"]),
            "Score_banco_NaN": int((~ok).sum()),
        }
        ref = REFERENCIA[muestra]
        assert abs(fila["AUC"] - ref["AUC"]) < tol, f"AUC {muestra} no reproduce: {fila['AUC']:.6f} vs {ref['AUC']}"
        assert abs(fila["KS"] - ref["KS"]) < tol, f"KS {muestra} no reproduce: {fila['KS']:.6f} vs {ref['KS']}"
        if "Aprobacion_%" in ref:
            ap = d["PD"] < cutoff
            assert round(ap.mean() * 100, 2) == ref["Aprobacion_%"], "Aprobación de la política no reproduce"
            assert round(d.loc[ap, "TARGET"].mean() * 100, 2) == ref["Bad_Rate_Aprobados_%"], "Bad rate no reproduce"
        filas.append(fila)
    return pd.DataFrame(filas)


# Métricas de CatBoost reportadas por Manolo (notebook 03, celda 15). El modelo no se guardó, así que se
# citan. Reentrenar su configuración exacta (semilla 42) en otro equipo da Gini OOT 0.6083 vs 0.6091 y
# VALIDATION 0.6165 vs 0.6162 (CatBoost varía con los hilos de CPU); no cambia ninguna conclusión.
CATBOOST_REPORTADO = {
    "TRAIN": {"AUC": 0.863099, "KS": 0.554235},
    "VALIDATION": {"AUC": 0.808093, "KS": 0.460563},
    "OOT": {"AUC": 0.804570, "KS": 0.462571},
}
MODELOS = {  # nombre: (n.º de variables, fuente de las métricas)
    "Modelo actual del banco": ("n.d.", f"Calculado sobre la columna {SCORE_BANCO}"),
    "Regresión Logística WOE": (18, "Calculado con model/modelo_logistico_woe.pkl"),
    "CatBoost (benchmark)": (45, "Reportado en notebooks/03_benchmark_catboost.ipynb"),
}
MUESTRAS = ["TRAIN", "VALIDATION", "OOT"]


def comparar_modelos(metricas):
    """Gini, AUC y KS en TRAIN, VALIDATION y OOT de los tres modelos, en formato ancho.

    `metricas` es la salida de verificar_reproduccion (logística y banco, ya calculados).
    """
    filas = []
    for _, m in metricas.iterrows():
        filas += [
            {"modelo": "Modelo actual del banco", "muestra": m["Muestra"], "AUC": m["AUC_banco"], "KS": m["KS_banco"]},
            {"modelo": "Regresión Logística WOE", "muestra": m["Muestra"], "AUC": m["AUC"], "KS": m["KS"]},
            {"modelo": "CatBoost (benchmark)", "muestra": m["Muestra"], **CATBOOST_REPORTADO[m["Muestra"]]},
        ]
    t = pd.DataFrame(filas)
    t["Gini"] = 2 * t["AUC"] - 1
    ancho = t.pivot(index="modelo", columns="muestra", values=["Gini", "AUC", "KS"])
    ancho.columns = [f"{met}_{mu}" for met, mu in ancho.columns]
    ancho = ancho[[f"{met}_{mu}" for met in ["Gini", "AUC", "KS"] for mu in MUESTRAS]].reindex(list(MODELOS))
    ancho["caida_gini_train_oot_pts"] = (ancho["Gini_TRAIN"] - ancho["Gini_OOT"]) * 100
    ancho.insert(0, "variables", [MODELOS[m][0] for m in ancho.index])
    ancho["fuente"] = [MODELOS[m][1] for m in ancho.index]
    return ancho.reset_index()


# Variables para la lámina de patrones: la creciente de mayor IV en cada familia de riesgo distinta.
# ponytail: elección fija; cambiar aquí (p. ej. MEDIDA1_MORA_INTERNA_U1M, IV 0.30) si se prefiere el top-3 puro por IV.
VARIABLES_LAMINA = {
    "MEDIDA1_MORA_INTERNA_U6M": "Mora interna (6 meses)",
    "MEDIDA1_APALANCAMIENTO_SSFF_U36M": "Apalancamiento SSFF (36 meses)",
    "MEDIDA1_CALIFICACION_SSFF_U60M": "Calificación SSFF (60 meses)",
}


def etiqueta_bin(b):
    """'[-inf,4.0)' → '< 4', '[4.0,7.0)' → '4–7', '[13.0,inf)' → '≥ 13', 'missing' → 'Sin dato'."""
    if b == "missing":
        return "Sin dato"
    if not b.startswith("["):
        return f"{b} (especial)"
    lo, hi = (f"{float(x):g}" for x in b[1:-1].split(","))
    return f"< {hi}" if lo == "-inf" else f"≥ {lo}" if hi == "inf" else f"{lo}–{hi}"


def patrones_variables():
    """Bad rate por bin WOE de las 18 variables en TRAIN y OOT, tal como el modelo asigna los bins al aplicarse.

    woebin_ply devuelve el WOE, no el bin; como el WOE es único por bin, se recupera el bin desde él.
    `clientes_artefacto` y `bad_rate_artefacto` son los del binning guardado. Difieren solo en
    MEDIDA3_APALANCAMIENTO_INTERNO_U1M: el centinela 333333344 (float32) se guardó como el texto
    '3.3333334e+08', no coincide al aplicar y esos clientes caen en el bin [1.28, inf). Pasa igual en el
    entrenamiento de Manolo, así que el modelo es consistente, pero el centinela no se trata como especial.
    """
    cfg, df, art = cargar_base()
    bins, variables = art["bins_woe"], art["variables_finales"]
    mes = df[cfg["time_column"]]
    filas = []
    for muestra, filtro in [("TRAIN", mes <= cfg["train_max"]), ("OOT", mes.isin(cfg["oot_months"]))]:
        base = df[filtro]
        woe = sc.woebin_ply(base[variables], bins, print_step=0)
        for v in variables:
            t = bins[v]
            assert t["woe"].round(10).is_unique, f"WOE repetido en {v}"
            b = woe[f"{v}_woe"].round(10).map(dict(zip(t["woe"].round(10), t["bin"])))
            g = base[cfg["target"]].groupby(b.to_numpy()).agg(["size", "mean"])
            for orden, fila in enumerate(t.itertuples()):
                filas.append({"muestra": muestra, "variable": v, "iv": fila.total_iv, "orden": orden, "bin": fila.bin,
                              "etiqueta": etiqueta_bin(fila.bin), "especial": bool(fila.is_special_values),
                              "clientes": int(g["size"].get(fila.bin, 0)), "bad_rate": g["mean"].get(fila.bin, np.nan),
                              "clientes_artefacto": fila.count, "bad_rate_artefacto": fila.badprob})
    p = pd.DataFrame(filas)
    tr = p[(p["muestra"] == "TRAIN") & p["variable"].isin(list(VARIABLES_LAMINA))]
    assert (tr["clientes"] == tr["clientes_artefacto"]).all() and np.allclose(tr["bad_rate"], tr["bad_rate_artefacto"]), \
        "Bad rate TRAIN de las variables de la lámina no coincide con el artefacto"
    p.loc[p["muestra"] != "TRAIN", ["clientes_artefacto", "bad_rate_artefacto"]] = np.nan  # el artefacto es de TRAIN
    # especiales y 'Sin dato' al final, fuera del orden de riesgo
    p["orden"] = p["orden"] + p["especial"] * 100
    return p.sort_values(["muestra", "variable", "orden"], ignore_index=True)


def monotonia(patrones):
    """Dirección del bad rate a lo largo de los bins normales (sin especiales ni 'Sin dato'), por variable y muestra."""
    def direccion(g):
        d = np.diff(g.loc[~g["especial"], "bad_rate"].to_numpy())
        return "creciente" if (d > 0).all() else "decreciente" if (d < 0).all() else "no monótona"
    t = patrones.groupby(["variable", "muestra"], sort=False).apply(direccion).unstack("muestra")
    t.columns = [f"direccion_{c}" for c in t.columns]
    t.insert(0, "iv", patrones.groupby("variable")["iv"].first())
    return t.sort_values("iv", ascending=False).reset_index()


# Backtesting de calibración por bandas: réplica del notebook 01 de Manolo (celdas 130 y 133).
BANDAS_PD = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 1.00]
ETIQUETAS_PD = ["0%-5%", "5%-10%", "10%-15%", "15%-20%", "20%-30%", "30%-40%", "40%+"]
RHOS_VASICEK = [0.005, 0.01, 0.02, 0.03]
# Tabla publicada por Manolo para VALIDATION (PD promedio % y bad rate %), para verificar la réplica.
REFERENCIA_BANDAS = {"pd": [2.7265, 7.3944, 12.3804, 17.3540, 24.1481, 34.5475, 57.6278],
                     "bad_rate": [2.9379, 6.5728, 11.4668, 17.3617, 20.7041, 34.0792, 54.9020]}


def calibracion_bandas(datos, confianza=0.95):
    """PD promedio vs bad rate observado por banda de PD, en cada muestra.

    Incluye el test bajo independencia (p-value) y la sensibilidad Vasicek: la banda es
    compatible con un rho si su bad rate cae dentro del intervalo de un factor para ese rho.
    """
    zc = norm.ppf(0.5 + confianza / 2)
    filas = []
    for muestra, d in datos.groupby("MUESTRA", sort=False):
        banda = pd.cut(d["PD"], bins=BANDAS_PD, labels=ETIQUETAS_PD, include_lowest=True, right=False)
        for b, g in d.groupby(banda, observed=True):
            p, n, obs = g["PD"].to_numpy(), len(g), int(g["TARGET"].sum())
            z = (obs - p.sum()) / np.sqrt((p * (1 - p)).sum())
            fila = {"muestra": muestra, "banda_pd": str(b), "clientes": n,
                    "pd_promedio_pct": p.mean() * 100, "bad_rate_pct": obs / n * 100,
                    "gap_pp": (p.mean() - obs / n) * 100, "p_value_independencia": 2 * (1 - norm.cdf(abs(z)))}
            corte = norm.ppf(np.clip(p.mean(), 1e-8, 1 - 1e-8))
            for rho in RHOS_VASICEK:
                inf = norm.cdf((corte - np.sqrt(rho) * zc) / np.sqrt(1 - rho))
                sup = norm.cdf((corte + np.sqrt(rho) * zc) / np.sqrt(1 - rho))
                fila[f"vasicek_rho_{rho * 100:g}pct_compatible"] = bool(inf <= obs / n <= sup)
            filas.append(fila)
    t = pd.DataFrame(filas)
    t["compatible_independencia"] = t["p_value_independencia"] >= 1 - confianza
    t["compatible_vasicek_todos_rho"] = t.filter(like="vasicek_").all(axis=1)
    v = t[t["muestra"] == "VALIDATION"]
    assert np.allclose(v["pd_promedio_pct"], REFERENCIA_BANDAS["pd"], atol=1e-3), "PD por banda no reproduce"
    assert np.allclose(v["bad_rate_pct"], REFERENCIA_BANDAS["bad_rate"], atol=1e-3), "Bad rate por banda no reproduce"
    return t


# ============================================================
# 2. ECONOMÍA POR CLIENTE Y EVALUACIÓN DE POLÍTICAS
# ============================================================
def economia_cliente(p, factor_margen=1.0, factor_lgd=1.0):
    """Ganancia por bueno aprobado y pérdida por malo aprobado, en S/.

    bueno: monto × saldo_medio × (TEA − fondeo − costo operativo) × plazo en años
    malo : monto × EAD × LGD + costo de cobranza
    Los factores sirven para la sensibilidad (±20 %).
    """
    margen = (p["tea"] - p["costo_fondeo"] - p["costo_operativo"]) * factor_margen
    ganancia = p["monto"] * p["saldo_medio"] * margen * p["plazo_meses"] / 12
    perdida = p["monto"] * p["ead"] * p["lgd"] * factor_lgd + p["costo_cobranza"]
    return ganancia, perdida


def pd_equilibrio(ganancia, perdida):
    """PD a partir de la cual aprobar deja de ser rentable: G / (G + L)."""
    return ganancia / (ganancia + perdida)


def aprobar_menores(score, k):
    """Aprueba los k clientes de menor score (los NaN nunca se aprueban)."""
    rango = pd.Series(score).rank(method="first", na_option="keep").to_numpy()
    return rango <= k


def evaluar_politica(y, aprobado, ganancia, perdida):
    """Resultado económico de una decisión de aprobación sobre clientes con TARGET observado.

    Rechazado = resultado 0: si era malo cuenta como pérdida evitada, si era bueno como
    ganancia sacrificada. Ahorro vs situación actual (aprobar a todos) = PE − GS.
    """
    y = np.asarray(y, dtype=int)
    a = np.asarray(aprobado, dtype=bool)
    buenos_ap, malos_ap = int(((y == 0) & a).sum()), int(((y == 1) & a).sum())
    buenos_re, malos_re = int(((y == 0) & ~a).sum()), int(((y == 1) & ~a).sum())
    r = {
        "n": len(y),
        "aprobados": int(a.sum()),
        "aprobacion_pct": a.mean() * 100,
        "tasa_malos_aprobados_pct": y[a].mean() * 100 if a.any() else np.nan,
        "ganancia_buenos": buenos_ap * ganancia,
        "perdida_malos": malos_ap * perdida,
        "perdida_evitada": malos_re * perdida,
        "ganancia_sacrificada": buenos_re * ganancia,
    }
    r["utilidad_neta"] = r["ganancia_buenos"] - r["perdida_malos"]
    r["ahorro_vs_actual"] = r["perdida_evitada"] - r["ganancia_sacrificada"]
    return r


def umbral_para_aprobacion(pd_validation, pct):
    """Umbral de PD que aprueba `pct` (0–1) de VALIDATION. Se congela y se aplica en OOT."""
    return float(np.quantile(pd_validation, pct))


# ============================================================
# 3. CURVA DE UTILIDAD Y ESCENARIOS
# ============================================================
def curva_utilidad(datos, ganancia, perdida, pcts=np.arange(40, 101)):
    """Utilidad neta vs % de aprobación para ambos modelos, en VALIDATION y OOT (descriptiva)."""
    filas = []
    for muestra, d in datos.groupby("MUESTRA", sort=False):
        for modelo, col in [("Modelo propuesto", "PD"), ("Modelo banco", "SCORE_BANCO")]:
            for pct in pcts:
                k = round(pct / 100 * len(d))
                r = evaluar_politica(d["TARGET"], aprobar_menores(d[col], k), ganancia, perdida)
                filas.append({"muestra": muestra, "modelo": modelo, "aprobacion_objetivo_pct": int(pct), **r})
    return pd.DataFrame(filas)


def optimo_validation(curva):
    """% de aprobación que maximiza la utilidad del modelo propuesto en VALIDATION."""
    c = curva[(curva["muestra"] == "VALIDATION") & (curva["modelo"] == "Modelo propuesto")]
    return int(c.loc[c["utilidad_neta"].idxmax(), "aprobacion_objetivo_pct"])


def decisiones(d, cutoff, umbrales):
    """Vector de aprobación por escenario para una muestra (o un mes) `d`.

    El modelo del banco aprueba el mismo número de clientes que la política PD < cutoff.
    """
    politica = (d["PD"] < cutoff).to_numpy()
    dec = {
        "A. Situación actual (aprobar a todos)": np.ones(len(d), dtype=bool),
        "B. Modelo del banco (igual volumen)": aprobar_menores(d["SCORE_BANCO"], int(politica.sum())),
        f"C. Modelo propuesto PD < {cutoff:.0%}": politica,
    }
    for nombre, u in umbrales.items():
        dec[nombre] = (d["PD"] < u).to_numpy()
    return dec


def tabla_escenarios(datos, cutoff, umbrales, ganancia, perdida):
    """Tabla de escenarios por muestra, con las identidades contables verificadas."""
    filas = []
    for muestra, d in datos.groupby("MUESTRA", sort=False):
        for escenario, aprobado in decisiones(d, cutoff, umbrales).items():
            filas.append({"muestra": muestra, "escenario": escenario,
                          **evaluar_politica(d["TARGET"], aprobado, ganancia, perdida)})
    t = pd.DataFrame(filas)
    t["umbral_pd"] = t["escenario"].map({**umbrales, f"C. Modelo propuesto PD < {cutoff:.0%}": cutoff})
    t["utilidad_neta_por_1000"] = t["utilidad_neta"] / t["n"] * 1000
    base = t[t["escenario"].str.startswith("A.")].set_index("muestra")["utilidad_neta"]
    banco = t[t["escenario"].str.startswith("B.")].set_index("muestra")["utilidad_neta"]
    t["ahorro_vs_banco"] = t["utilidad_neta"] - t["muestra"].map(banco)
    verificar_cuadre(t, base)
    return t


def verificar_cuadre(t, utilidad_base):
    """ganancia − pérdida = utilidad y U_actual + PE − GS = U_escenario, fila por fila."""
    assert np.allclose(t["ganancia_buenos"] - t["perdida_malos"], t["utilidad_neta"])
    esperado = t["muestra"].map(utilidad_base) + t["perdida_evitada"] - t["ganancia_sacrificada"]
    assert np.allclose(esperado, t["utilidad_neta"]), "No cuadra U_actual + PE − GS = U_escenario"


def estabilidad_mensual(datos, cutoff, ganancia, perdida):
    """Ahorro de la política PD < cutoff (umbral fijo) mes a mes, con su descomposición."""
    filas = []
    for (muestra, mes), d in datos.groupby(["MUESTRA", "CODMES"], sort=False):
        dec = decisiones(d, cutoff, {})
        u = {k[0]: evaluar_politica(d["TARGET"], a, ganancia, perdida) for k, a in dec.items()}
        filas.append({
            "muestra": muestra, "codmes": int(mes), "n": len(d),
            "bad_rate_pct": d["TARGET"].mean() * 100,
            "aprobacion_pct": u["C"]["aprobacion_pct"],
            "tasa_malos_aprobados_pct": u["C"]["tasa_malos_aprobados_pct"],
            "tasa_malos_banco_pct": u["B"]["tasa_malos_aprobados_pct"],
            "ahorro_total": u["C"]["utilidad_neta"] - u["A"]["utilidad_neta"],
            "ahorro_politica": u["B"]["utilidad_neta"] - u["A"]["utilidad_neta"],
            "ahorro_modelo": u["C"]["utilidad_neta"] - u["B"]["utilidad_neta"],
        })
    t = pd.DataFrame(filas)
    t["ahorro_total_por_1000"] = t["ahorro_total"] / t["n"] * 1000
    return t


# ============================================================
# 4. ROBUSTEZ
# ============================================================
def bootstrap_ahorro(d, cutoff, ganancia, perdida, n_replicas=1000, semilla=42):
    """IC 95 % del ahorro en OOT remuestreando clientes (umbrales congelados, sin re-optimizar).

    ponytail: bootstrap i.i.d.; ignora la correlación entre defaults (Vasicek),
    así que el IC es optimista. Bootstrap por bloques/mes si hubiera más meses OOT.
    """
    y = d["TARGET"].to_numpy()
    valor = np.where(y == 0, ganancia, -perdida)  # aporte de cada cliente si se aprueba
    dec = decisiones(d, cutoff, {})
    a, b, c = (dec[k].astype(float) * valor for k in dec)
    aportes = {"ahorro_total": c - a, "ahorro_politica": b - a, "ahorro_modelo": c - b}
    idx = np.random.default_rng(semilla).integers(0, len(d), size=(n_replicas, len(d)))
    filas = []
    for nombre, v in aportes.items():
        rep = v[idx].sum(axis=1)
        filas.append({"metrica": nombre, "estimado": v.sum(), "media_bootstrap": rep.mean(),
                      "ic95_inf": np.percentile(rep, 2.5), "ic95_sup": np.percentile(rep, 97.5),
                      "prob_ahorro_positivo": (rep > 0).mean(), "replicas": n_replicas})
    return pd.DataFrame(filas)


def sensibilidad(d, cutoff, p, factores=(0.8, 1.0, 1.2)):
    """Ahorro en OOT a umbral fijo variando LGD y margen ±20 %, con la PD de equilibrio resultante."""
    dec = decisiones(d, cutoff, {})
    filas = []
    for f_lgd in factores:
        for f_margen in factores:
            g, l = economia_cliente(p, factor_margen=f_margen, factor_lgd=f_lgd)
            u = {k[0]: evaluar_politica(d["TARGET"], a, g, l)["utilidad_neta"] for k, a in dec.items()}
            filas.append({
                "factor_lgd": f_lgd, "lgd": p["lgd"] * f_lgd, "factor_margen": f_margen,
                "ganancia_por_bueno": g, "perdida_por_malo": l, "pd_equilibrio_pct": pd_equilibrio(g, l) * 100,
                "utilidad_actual": u["A"], "utilidad_banco": u["B"], "utilidad_propuesto": u["C"],
                "ahorro_total": u["C"] - u["A"], "ahorro_politica": u["B"] - u["A"],
                "ahorro_modelo": u["C"] - u["B"],
                "ahorro_total_pct": (u["C"] - u["A"]) / abs(u["A"]) * 100,
            })
    return pd.DataFrame(filas)


# ============================================================
# 5. GRÁFICOS (16:9, fondo blanco, un acento + grises)
# ============================================================
# Fucsia solo para el modelo propuesto; los demás modelos en grises pizarra para que no compitan.
ACENTO, GRIS_OSCURO, GRIS, GRIS_CLARO, TEXTO = "#D10463", "#637083", "#A7B1BF", "#DDE3EB", "#172033"
AMBAR, PIZARRA_CLARO = "#F5CB71", "#C3C9CF"  # modelo actual del banco y CatBoost en la comparación


def soles_m(x):
    return f"S/ {x / 1e6:,.2f} M"


def guardar(fig, ruta, intentos=5):
    """Guarda en PNG 200 dpi. Reintenta si un visor abierto bloquea el archivo un instante (Windows)."""
    import time
    for i in range(intentos):
        try:
            return fig.savefig(ruta, dpi=200)
        except OSError:
            if i == intentos - 1:
                raise
            time.sleep(0.5)


def _estilo():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 15,
        "axes.edgecolor": GRIS_CLARO, "axes.labelcolor": TEXTO, "xtick.color": TEXTO, "ytick.color": TEXTO,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    return plt


def graficar_barras(esc_oot, ruta):
    """Actual vs banco vs modelo propuesto: ganancia de buenos, pérdida de malos y utilidad neta."""
    plt = _estilo()
    t = esc_oot.set_index(esc_oot["escenario"].str[0])
    filas = ["A", "B", "C"]
    etiquetas = ["Actual\n(aprobar a todos)", "Modelo\nbanco", "Modelo\npropuesto"]
    colores = [GRIS_CLARO, GRIS, ACENTO]
    u = t.loc[filas, "utilidad_neta"].to_numpy()
    fig, ejes = plt.subplots(1, 3, figsize=(13.33, 7.5))
    for ax, col, titulo in zip(ejes, ["ganancia_buenos", "perdida_malos", "utilidad_neta"],
                               ["Ganancia de buenos", "Pérdida por malos", "Utilidad neta"]):
        v = t.loc[filas, col].to_numpy()
        ax.bar(etiquetas, v, color=colores, width=0.65)
        for i, x in enumerate(v):
            ax.text(i, x, soles_m(x), ha="center", va="bottom", fontsize=14,
                    color=ACENTO if i == 2 else TEXTO, fontweight="bold" if i == 2 else "normal")
        ax.set_title(titulo, fontsize=17, color=TEXTO, pad=12, loc="left")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=13, length=0)
        ax.set_ylim(0, t[col].max() * 1.3)
    # Descomposición del ahorro, sobre la etiqueta de valor de cada barra
    for i, texto in [(1, f"+{soles_m(u[1] - u[0])} política"), (2, f"+{soles_m(u[2] - u[1])} mejor modelo")]:
        ejes[2].annotate(texto, (i, u[i]), xytext=(0, 26), textcoords="offset points", ha="center",
                         fontsize=12, color=GRIS_OSCURO)
    fig.suptitle(f"{soles_m(u[2] - u[0])} más de utilidad: {soles_m(u[1] - u[0])} por tener una política "
                 f"y {soles_m(u[2] - u[1])} por un mejor modelo", fontsize=20, color=TEXTO, x=0.02, ha="left",
                 fontweight="bold")
    fig.text(0.02, 0.905, f"Muestra fuera de tiempo (OOT, abr–may 2024, {int(t.loc['A', 'n']):,} renovaciones). "
             "Supuestos económicos iniciales; resultado válido para la población que hoy se renueva.",
             fontsize=13, color=GRIS_OSCURO)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    guardar(fig, ruta)
    plt.close(fig)


def graficar_curva(curva, esc_oot, ruta):
    """Utilidad neta vs % de aprobación en OOT, ambos modelos, con la política propuesta resaltada."""
    plt = _estilo()
    c = curva[curva["muestra"] == "OOT"]
    t = esc_oot.set_index(esc_oot["escenario"].str[0])
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    prop = c[c["modelo"] == "Modelo propuesto"]
    maximo = prop.loc[prop["utilidad_neta"].idxmax()]
    ax.axhline(maximo["utilidad_neta"] / 1e6, color=GRIS, lw=1, ls="--")
    ax.text(101.5, maximo["utilidad_neta"] / 1e6, f"Máximo observado en OOT\n{soles_m(maximo['utilidad_neta'])} "
            f"({maximo['aprobacion_pct']:.0f}% aprob.)", ha="right", va="bottom", fontsize=12, color=GRIS_OSCURO)
    for modelo, color, ancho in [("Modelo banco", GRIS, 2.5), ("Modelo propuesto", ACENTO, 3.5)]:
        m = c[c["modelo"] == modelo]
        ax.plot(m["aprobacion_pct"], m["utilidad_neta"] / 1e6, color=color, lw=ancho)
        ax.text(m["aprobacion_pct"].iloc[0] - 0.5, m["utilidad_neta"].iloc[0] / 1e6, modelo,
                ha="right", va="center", fontsize=14, color=ACENTO if color == ACENTO else GRIS_OSCURO,
                fontweight="bold")
    ap, un = t.loc["C", "aprobacion_pct"], t.loc["C", "utilidad_neta"] / 1e6
    ub = t.loc["B", "utilidad_neta"] / 1e6
    ax.vlines(ap, ub, un, color=GRIS_OSCURO, lw=1.2, ls=":")
    ax.scatter([ap], [un], s=160, color=ACENTO, zorder=5)
    ax.scatter([ap], [ub], s=90, color=GRIS, zorder=5)
    ax.annotate(f"Política PD < 20%\n{ap:.1f}% aprobación · {soles_m(un * 1e6)}", (ap, un), xytext=(-14, 14),
                textcoords="offset points", ha="right", fontsize=14, color=ACENTO, fontweight="bold")
    ax.annotate(f"+{soles_m((un - ub) * 1e6)} vs banco\na igual aprobación", (ap, (un + ub) / 2), xytext=(12, -10),
                textcoords="offset points", fontsize=13, color=GRIS_OSCURO)
    ax.scatter([100], [t.loc["A", "utilidad_neta"] / 1e6], s=90, color=GRIS_OSCURO, zorder=5)
    ax.annotate("Situación actual\n(aprobar a todos)", (100, t.loc["A", "utilidad_neta"] / 1e6), xytext=(-14, 0),
                textcoords="offset points", ha="right", va="center", fontsize=13, color=GRIS_OSCURO)
    ax.set_xlabel("% de clientes con renovación aprobada")
    ax.set_ylabel("Utilidad neta (S/ millones)")
    ax.set_xlim(32, 102)
    ax.grid(axis="y", color=GRIS_CLARO, lw=0.8)
    ax.set_axisbelow(True)
    fig.suptitle(f"PD < 20% logra el {un * 1e6 / maximo['utilidad_neta']:.0%} de la utilidad máxima y supera al "
                 "modelo del banco en todo el rango", fontsize=20, color=TEXTO, x=0.02, ha="left", fontweight="bold")
    fig.text(0.02, 0.905, "Utilidad neta en OOT (abr–may 2024) según el % aprobado; cada modelo aprueba primero a "
             "sus clientes de menor riesgo. Corte fijado en VALIDATION.", fontsize=13, color=GRIS_OSCURO)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    guardar(fig, ruta)
    plt.close(fig)


def graficar_comparacion(comp, ruta):
    """Gini por periodo (TRAIN → VALIDATION → OOT) de los tres modelos, una línea por modelo."""
    plt = _estilo()
    c = comp.set_index("modelo")
    # (color, grosor, desplazamiento de la etiqueta en puntos): CatBoost arriba y la logística abajo, porque
    # en VALIDATION y OOT sus valores están a menos de 0.035 y las etiquetas chocarían.
    estilos = {"Regresión Logística WOE": (ACENTO, 5, -26),
               "Modelo actual del banco": (AMBAR, 3, -24),
               "CatBoost (benchmark)": (PIZARRA_CLARO, 3, 14)}
    x = np.arange(len(MUESTRAS))
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    for modelo, (color, grosor, dy) in estilos.items():
        v = c.loc[modelo, [f"Gini_{m}" for m in MUESTRAS]].to_numpy(dtype=float)
        clave = color == ACENTO
        etiqueta = modelo if modelo == "Modelo actual del banco" else f"{modelo} · {c.loc[modelo, 'variables']} variables"
        ax.plot(x, v, color=color, lw=grosor, marker="o", ms=13 if clave else 10, mec="white", mew=1.5,
                label=etiqueta, zorder=4 if clave else 3)
        for xi, val in zip(x, v):
            ax.annotate(f"{val:.3f}", (xi, val), xytext=(0, dy), textcoords="offset points", ha="center",
                        fontsize=15 if clave else 13, color=ACENTO if clave else GRIS_OSCURO,
                        fontweight="bold" if clave else "normal")
    ax.set_xticks(x, ["TRAIN\n(jun 2023 – ene 2024)", "VALIDATION\n(feb – mar 2024)", "OOT\n(abr – may 2024)"])
    ax.tick_params(axis="x", length=0, labelsize=14)
    ax.set_xlim(-0.3, len(MUESTRAS) - 0.7)
    ax.set_ylabel("Gini")
    ax.set_ylim(0.35, 0.8)
    ax.grid(axis="y", color=GRIS_CLARO, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=13)
    lr, banco, cb = (c.loc[m] for m in ["Regresión Logística WOE", "Modelo actual del banco", "CatBoost (benchmark)"])
    fig.suptitle(f"La logística WOE gana +{(lr['Gini_OOT'] - banco['Gini_OOT']) * 100:.1f} pts de Gini OOT al modelo "
                 f"actual y queda a {(cb['Gini_OOT'] - lr['Gini_OOT']) * 100:.1f} pts de CatBoost",
                 fontsize=19, color=TEXTO, x=0.02, ha="left", fontweight="bold")
    fig.text(0.02, 0.905, f"Gini por periodo. CatBoost usa 45 variables y pierde {cb['caida_gini_train_oot_pts']:.1f} pts "
             f"de TRAIN a OOT; la logística usa 18 y pierde {lr['caida_gini_train_oot_pts']:.1f} pts.",
             fontsize=13, color=GRIS_OSCURO)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    guardar(fig, ruta)
    plt.close(fig)


def graficar_patrones(pat, mono, ruta):
    """Bad rate por bin (TRAIN en barras, OOT en puntos) de las variables de la lámina, un panel por variable."""
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt = _estilo()
    rgb = np.array(mcolors.to_rgb(ACENTO))
    n_bins = [int(((pat["muestra"] == "TRAIN") & (pat["variable"] == v)).sum()) for v in VARIABLES_LAMINA]
    fig, ejes = plt.subplots(1, len(VARIABLES_LAMINA), figsize=(13.33, 7.5), sharey=True,
                             gridspec_kw={"width_ratios": n_bins})
    for ax, (v, nombre) in zip(ejes, VARIABLES_LAMINA.items()):
        tr = pat[(pat["muestra"] == "TRAIN") & (pat["variable"] == v)].reset_index(drop=True)
        oot = pat[(pat["muestra"] == "OOT") & (pat["variable"] == v)].set_index("bin").loc[tr["bin"]]
        normal = ~tr["especial"].to_numpy()
        # degradé fucsia: más intenso a más riesgo; 'Sin dato' en gris
        intensidad = np.linspace(0.35, 1, normal.sum())
        colores = [tuple(1 - f * (1 - rgb)) for f in intensidad] + [GRIS_CLARO] * (~normal).sum()
        x = np.arange(len(tr))
        y_tr, y_oot = tr["bad_rate"].to_numpy() * 100, oot["bad_rate"].to_numpy() * 100
        ax.bar(x, y_tr, color=colores, width=0.68)
        ax.plot(x, y_oot, ls="none", color=TEXTO, marker="o", ms=9, mec="white", mew=1.2, zorder=5)
        for xi, a, b, es_normal in zip(x, y_tr, y_oot, normal):
            ax.annotate(f"{a:.1f}%", (xi, max(a, b)), xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=13, color=ACENTO if es_normal else GRIS_OSCURO, fontweight="bold" if es_normal else "normal")
        ax.set_xticks(x, tr["etiqueta"])
        ax.tick_params(axis="x", length=0, labelsize=12)
        ax.set_title(nombre, loc="left", fontsize=16, color=TEXTO, fontweight="bold", pad=26)
        ax.text(0, 1.02, f"IV {tr['iv'].iloc[0]:.2f} · {v}", transform=ax.transAxes, fontsize=10.5, color=GRIS_OSCURO)
        ax.grid(axis="y", color=GRIS_CLARO, lw=0.8)
        ax.set_axisbelow(True)
    ejes[0].set_ylabel("Bad rate (%)")
    ejes[0].set_ylim(0, 58)
    fig.legend(handles=[Patch(color=ACENTO, label="Bad rate por bin en desarrollo (TRAIN)"),
                        Line2D([], [], color=TEXTO, marker="o", ms=9, ls="none", label="Bad rate fuera de tiempo (OOT, abr–may 2024)"),
                        Patch(color=GRIS_CLARO, label="Sin dato (bin aparte)")],
               loc="lower center", ncol=3, frameon=False, fontsize=13)
    n = len(mono)
    mono_tr = int((mono["direccion_TRAIN"] != "no monótona").sum())
    mono_oot = int((mono["direccion_OOT"] != "no monótona").sum())
    fig.suptitle("Las variables finales muestran patrones de riesgo coherentes", fontsize=22, color=TEXTO, x=0.02,
                 ha="left", fontweight="bold")
    fig.text(0.02, 0.915, f"De 45 variables candidatas a {n} finales: {mono_tr} de {n} tienen bad rate monótono por bins "
             f"en desarrollo y {mono_oot} de {n} fuera de tiempo.\nEn estas tres, las más fuertes de cada familia "
             "de riesgo, a mayor valor, mayor riesgo.", fontsize=13, color=GRIS_OSCURO, va="top", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.85))
    guardar(fig, ruta)
    plt.close(fig)


MAGENTA, MAGENTA_OSCURO, DORADO, ROSA = ACENTO, "#8F1150", "#E0A31C", "#FBE4EF"


def graficar_calibracion(cal, ruta, muestra="VALIDATION"):
    """Riesgo estimado (PD promedio) vs riesgo observado (bad rate) por banda de PD, con la nota Vasicek."""
    from matplotlib.patches import FancyBboxPatch
    plt = _estilo()
    t = cal[cal["muestra"] == muestra]
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    for col, color, nombre, offset, ha in [("pd_promedio_pct", MAGENTA, "PD promedio estimada", (0, 12), "center"),
                                            ("bad_rate_pct", DORADO, "Bad rate observado", (10, -22), "left")]:
        v = t[col].to_numpy()
        ax.plot(x, v, color=color, lw=3, marker="o", ms=11, mec="white", mew=1.5, label=nombre, zorder=3)
        for xi, vi in zip(x, v):
            ax.annotate(f"{vi:.2f}", (xi, vi), xytext=offset, textcoords="offset points", ha=ha,
                        fontsize=14, color=color, fontweight="bold")
    ax.set_xticks(x, t["banda_pd"])
    ax.tick_params(axis="x", length=0, labelsize=14)
    ax.set_xlabel("Banda de PD", fontsize=15, color=GRIS_OSCURO)
    ax.set_ylabel("Riesgo (%)", fontsize=15, color=GRIS_OSCURO)
    ax.set_ylim(-6, 70)  # margen bajo el 0 para la etiqueta de la primera banda
    ax.set_yticks(range(0, 71, 10))
    ax.grid(color=GRIS_CLARO, lw=0.8, ls="--")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=14)
    fig.text(0.04, 0.93, "Riesgo estimado vs. riesgo observado", fontsize=32, color=MAGENTA_OSCURO, fontweight="bold")
    fig.text(0.04, 0.865, "Validación por bandas de PD", fontsize=20, color=GRIS_OSCURO)

    # Nota Vasicek calculada: bandas compatibles para todos los rho del rango
    n_ok, n = int(t["compatible_vasicek_todos_rho"].sum()), len(t)
    rango = f"{min(RHOS_VASICEK):.1%}–{max(RHOS_VASICEK):.0%}"
    fig.add_artist(FancyBboxPatch((0.25, 0.015), 0.5, 0.06, boxstyle="round,pad=0.005,rounding_size=0.012",
                                  transform=fig.transFigure, facecolor=ROSA, edgecolor="none"))
    partes = [fig.text(0, 0.045, f"Sensibilidad Vasicek (ρ = {rango}):", fontsize=14, color=MAGENTA_OSCURO,
                       fontweight="bold", va="center"),
              fig.text(0, 0.045, f" {n_ok} de {n} bandas compatibles", fontsize=14, color=MAGENTA_OSCURO, va="center")]
    fig.tight_layout(rect=(0, 0.07, 1, 0.84))
    # centra la frase (negrita + normal) dentro del recuadro
    render = fig.canvas.get_renderer()
    anchos = [p.get_window_extent(render).width / fig.bbox.width for p in partes]
    x0 = 0.5 - sum(anchos) / 2
    partes[0].set_x(x0)
    partes[1].set_x(x0 + anchos[0])
    guardar(fig, ruta)
    plt.close(fig)


# ============================================================
# 6. EJECUCIÓN COMPLETA
# ============================================================
def ejecutar(guardar=True):
    """Corre todo el análisis. Devuelve un dict con las tablas; si `guardar`, escribe impacto/outputs/."""
    p = valores()
    ganancia, perdida = economia_cliente(p)
    todos, cutoff = cargar_predicciones()
    metricas = verificar_reproduccion(todos, cutoff)
    datos = todos[todos["MUESTRA"] != "TRAIN"]  # la economía nunca usa TRAIN

    curva = curva_utilidad(datos, ganancia, perdida)
    pct_opt = optimo_validation(curva)
    pd_val = datos.loc[datos["MUESTRA"] == "VALIDATION", "PD"]
    umbrales = {
        f"D. Óptimo económico (elegido en VALIDATION: {pct_opt}% aprob.)": umbral_para_aprobacion(pd_val, pct_opt / 100),
        "E. 90% de aprobación (umbral de VALIDATION)": umbral_para_aprobacion(pd_val, 0.90),
        "F. 80% de aprobación (umbral de VALIDATION)": umbral_para_aprobacion(pd_val, 0.80),
    }
    escenarios = tabla_escenarios(datos, cutoff, umbrales, ganancia, perdida)
    oot = datos[datos["MUESTRA"] == "OOT"]
    res = {
        "supuestos": pd.DataFrame([{"parametro": k, **v} for k, v in SUPUESTOS.items()]),
        "metricas": metricas,
        "comparacion": comparar_modelos(metricas),
        "calibracion": calibracion_bandas(datos),
        "patrones": (patrones := patrones_variables()),
        "monotonia": monotonia(patrones),
        "ganancia_por_bueno": ganancia,
        "perdida_por_malo": perdida,
        "pd_equilibrio": pd_equilibrio(ganancia, perdida),
        "optimo_validation_pct": pct_opt,
        "escenarios": escenarios,
        "curva": curva,
        "estabilidad": estabilidad_mensual(datos, cutoff, ganancia, perdida),
        "bootstrap": bootstrap_ahorro(oot, cutoff, ganancia, perdida),
        "sensibilidad": sensibilidad(oot, cutoff, p),
    }
    if pct_opt == 100:
        print("ALERTA: el óptimo en VALIDATION es aprobar al 100 %; revisar supuestos antes de seguir.")

    if guardar:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        escenarios.round(4).to_csv(OUT_DIR / "escenarios.csv", index=False, encoding="utf-8-sig")
        curva.round(4).to_csv(OUT_DIR / "curva_utilidad.csv", index=False, encoding="utf-8-sig")
        res["bootstrap"].round(4).to_csv(OUT_DIR / "bootstrap_ic.csv", index=False, encoding="utf-8-sig")
        res["sensibilidad"].round(4).to_csv(OUT_DIR / "sensibilidad.csv", index=False, encoding="utf-8-sig")
        res["estabilidad"].round(4).to_csv(OUT_DIR / "estabilidad_mensual.csv", index=False, encoding="utf-8-sig")
        res["comparacion"].round(6).to_csv(OUT_DIR / "comparacion_modelos.csv", index=False, encoding="utf-8-sig")
        graficar_comparacion(res["comparacion"], OUT_DIR / "comparacion_modelos.png")
        res["calibracion"].round(4).to_csv(OUT_DIR / "calibracion_bandas.csv", index=False, encoding="utf-8-sig")
        graficar_calibracion(res["calibracion"], OUT_DIR / "calibracion_bandas.png")
        res["patrones"].round(4).to_csv(OUT_DIR / "patrones_variables.csv", index=False, encoding="utf-8-sig")
        res["monotonia"].round(4).to_csv(OUT_DIR / "monotonia_variables.csv", index=False, encoding="utf-8-sig")
        graficar_patrones(res["patrones"], res["monotonia"], OUT_DIR / "patrones_variables.png")
        esc_oot = escenarios[escenarios["muestra"] == "OOT"]
        graficar_barras(esc_oot, OUT_DIR / "barras_impacto.png")
        graficar_curva(curva, esc_oot, OUT_DIR / "curva_utilidad.png")
    return res


if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 30,
                  "display.float_format", lambda x: f"{x:,.2f}")
    r = ejecutar()
    f4 = lambda x: f"{x:.4f}"  # noqa: E731
    print(r["metricas"].to_string(index=False, float_format=f4))
    print("\n=== Comparación de modelos ===\n" + r["comparacion"].drop(columns="fuente").to_string(index=False, float_format=f4))
    cal_cols = ["muestra", "banda_pd", "clientes", "pd_promedio_pct", "bad_rate_pct", "gap_pp", "p_value_independencia",
                "compatible_independencia", "compatible_vasicek_todos_rho"]
    print("\n=== Calibración por bandas ===\n" + r["calibracion"][cal_cols].to_string(index=False, float_format=f4))
    print(f"\nGanancia por bueno: S/ {r['ganancia_por_bueno']:,.0f} · Pérdida por malo: S/ {r['perdida_por_malo']:,.0f}"
          f" · PD de equilibrio: {r['pd_equilibrio']:.1%} · Óptimo VALIDATION: {r['optimo_validation_pct']}% aprob.")
    cols = ["escenario", "umbral_pd", "aprobacion_pct", "tasa_malos_aprobados_pct", "ganancia_buenos",
            "perdida_malos", "utilidad_neta", "perdida_evitada", "ganancia_sacrificada", "ahorro_vs_actual",
            "ahorro_vs_banco"]
    for m in ["VALIDATION", "OOT"]:
        print(f"\n=== Escenarios {m} ===")
        print(r["escenarios"].loc[r["escenarios"]["muestra"] == m, cols].to_string(index=False))
    print("\n=== Estabilidad mensual ===\n" + r["estabilidad"].to_string(index=False))
    print("\n=== Bootstrap OOT ===\n" + r["bootstrap"].to_string(index=False))
    print("\n=== Sensibilidad OOT ===\n" + r["sensibilidad"].to_string(index=False))
    print("\nCuadres OK. Salidas en", OUT_DIR)
