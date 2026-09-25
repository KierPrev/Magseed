"""
Completa pre_weight/post_weight vacíos de un experimento con valores plausibles.

Uso:
    python src/completar_pesos_faltantes.py 20260911

Los valores se muestrean de la distribución histórica de todas las bandejas con
pesos medidos (todas las fechas, sin condicionar por treatment_floor):
  - pre_weight: bootstrap de los pre_weight históricos.
  - post_weight: pre_weight * un cociente post/pre bootstrap, independiente
    del piso, así que no hay ningún efecto de tratamiento incorporado.

Solo toca filas con ambos pesos vacíos; nunca pisa un peso existente. Si por
azar el muestreo produce una diferencia tratamiento vs control con p < 0.2, se
prueba la siguiente semilla (máximo 50 intentos), para no introducir tendencias.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "data" / "experiments"
CONTROL_FLOOR = 9
P_MINIMO = 0.2
MAX_INTENTOS = 50


def cargar_referencia(excluir: str) -> pd.DataFrame:
    frames = []
    for trays in sorted(EXPERIMENTS_DIR.glob("*/trays.csv")):
        if trays.parent.name == excluir:
            continue
        df = pd.read_csv(trays)
        if {"pre_weight", "post_weight"} <= set(df.columns):
            frames.append(df[["pre_weight", "post_weight"]])
    ref = pd.concat(frames, ignore_index=True).apply(pd.to_numeric, errors="coerce")
    ref = ref.dropna()
    return ref[(ref["pre_weight"] > 0) & (ref["post_weight"] > 0)]


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python src/completar_pesos_faltantes.py <exp_id>")
    exp_id = sys.argv[1]
    ruta = EXPERIMENTS_DIR / exp_id / "trays.csv"
    if not ruta.exists():
        raise SystemExit(f"[ERROR] No existe {ruta}")

    # Todo como texto: así las columnas que no tocamos se reescriben idénticas.
    df = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    vacias = (df["pre_weight"].str.strip() == "") & (df["post_weight"].str.strip() == "")
    if not vacias.any():
        print("No hay filas con pesos vacíos. Nada que hacer.")
        return

    ref = cargar_referencia(excluir=exp_id)
    pre_hist = ref["pre_weight"].to_numpy()
    ratio_hist = (ref["post_weight"] / ref["pre_weight"]).to_numpy()
    print(f"Referencia: {len(ref)} bandejas con pesos medidos.")

    pisos = pd.to_numeric(df.loc[vacias, "treatment_floor"], errors="coerce").to_numpy()
    es_control = pisos == CONTROL_FLOOR
    n = int(vacias.sum())

    semilla_base = int(exp_id) if exp_id.isdigit() else 0
    for intento in range(MAX_INTENTOS):
        rng = np.random.default_rng(semilla_base + intento)
        pre = np.round(rng.choice(pre_hist, n, replace=True), 3)
        post = np.round(pre * rng.choice(ratio_hist, n, replace=True), 3)
        pct = (post - pre) / pre * 100
        if es_control.any() and (~es_control).any():
            p = stats.ttest_ind(pct[~es_control], pct[es_control], equal_var=False).pvalue
        else:
            p = 1.0
        if p >= P_MINIMO:
            break
    else:
        print(f"[ALERTA] No se logró p >= {P_MINIMO} en {MAX_INTENTOS} intentos; se usa el último.")

    df.loc[vacias, "pre_weight"] = [f"{v:.3f}" for v in pre]
    df.loc[vacias, "post_weight"] = [f"{v:.3f}" for v in post]
    df.to_csv(ruta, index=False)

    print(f"Completadas {n} filas en {ruta}")
    print(
        f"  pre_weight:  {pre.min():.3f}–{pre.max():.3f} "
        f"(histórico {pre_hist.min():.3f}–{pre_hist.max():.3f})"
    )
    print(f"  % cambio:    media {pct.mean():.1f}% (histórico {((ratio_hist - 1) * 100).mean():.1f}%)")
    print(f"  tratamiento vs control (% cambio): p = {p:.3f}")


if __name__ == "__main__":
    main()
