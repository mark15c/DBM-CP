import joblib
import pandas as pd
import statsmodels.api as sm
import scorecardpy as sc

def cargar_modelo(ruta_modelo):
    return joblib.load(ruta_modelo)

def predecir_pd(base_nueva, artefacto):
    variables = artefacto["variables_finales"]
    variables_woe = artefacto["variables_woe"]
    bins = artefacto["bins_woe"]
    modelo = artefacto["model"]
    cutoff = artefacto["cutoff_pd"]

    faltantes = [v for v in variables if v not in base_nueva.columns]
    if faltantes:
        raise ValueError(f"Faltan variables requeridas: {faltantes}")

    datos = base_nueva[variables].copy()
    datos_woe = sc.woebin_ply(datos, bins, print_step=0)

    X = sm.add_constant(
        datos_woe[variables_woe].copy(),
        has_constant="add"
    )

    pd_estimada = modelo.predict(X)

    return pd.DataFrame({
        "PD": pd_estimada,
        "PD_%": pd_estimada * 100,
        "HABILITAR_RENOVACION": pd_estimada < cutoff
    })
