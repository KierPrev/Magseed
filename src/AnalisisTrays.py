from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
except ImportError:  # pragma: no cover - handled explicitly in main
    sm = None
    smf = None
    multipletests = None


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
METADATA_PATH = DATA_DIR / "experiments_metadata.csv"
GLOBAL_ANALYSIS_DIR = DATA_DIR / "analysis"

TRAYS_FILENAME = "trays.csv"
TRAY_COLUMNS = [
    "tray_id",
    "exp_id",
    "treatment_floor",
    "floor",
    "position",
    "pre_weight",
    "post_weight",
]
REQUIRED_TRAY_COLUMNS = set(TRAY_COLUMNS)
METADATA_COLUMNS = ["exp_id", "magnet", "magnet_pole"]
REQUIRED_METADATA_COLUMNS = set(METADATA_COLUMNS)
SUMMARY_METRICS = (
    "pre_weight",
    "post_weight",
    "delta_weight",
    "pct_change",
    "log_ratio_centered",
)
FLOOR_ORDER = ["A", "B", "C"]
MAGNET_NONE_VALUE = "NONE"
MAGNETIC_TREATMENT_FLOORS = tuple(range(1, 9))
CONTROL_TREATMENT_FLOOR = 9
METRIC_YLABELS = {
    "delta_weight": "(post_weight - pre_weight) / mean(control_delta)",
    "pct_change": "(post_weight - pre_weight) / pre_weight",
    "log_ratio_centered": "log(post_weight / pre_weight) - mean(log(control_post / control_pre))",
}
METRIC_FOREST_TITLES = {
    "delta_weight": "Tratamiento vs Control (delta_weight normalizado)",
    "pct_change": "Tratamiento vs Control (pct_change)",
    "log_ratio_centered": "Tratamiento vs Control (log_ratio centrado por control)",
}


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analiza trays.csv de experimentos MagnaGrow, genera tablas, "
            "modelos estadisticos y graficas por experimento."
        )
    )
    parser.add_argument(
        "--exp-id",
        help="Analiza solo el experimento indicado (formato YYYYMMDD).",
    )
    return parser


def validar_exp_id(exp_id: str) -> str:
    valor = exp_id.strip()
    if len(valor) != 8 or not valor.isdigit():
        raise ValueError("El exp_id debe tener formato YYYYMMDD.")
    return valor


def descubrir_trays(exp_id: str | None = None) -> list[Path]:
    if exp_id is not None:
        exp_id = validar_exp_id(exp_id)
        ruta = EXPERIMENTS_DIR / exp_id / TRAYS_FILENAME
        if not ruta.exists():
            raise FileNotFoundError(f"No existe '{ruta}'.")
        return [ruta]

    rutas = sorted(EXPERIMENTS_DIR.glob(f"*/{TRAYS_FILENAME}"))
    if not rutas:
        raise FileNotFoundError(
            f"No se encontraron archivos '{TRAYS_FILENAME}' en '{EXPERIMENTS_DIR}'."
        )
    return rutas


def leer_metadata_global() -> tuple[pd.DataFrame, list[str]]:
    advertencias: list[str] = []

    if not METADATA_PATH.exists():
        advertencias.append(
            f"No existe '{METADATA_PATH}'. Se analizara cada experimento sin metadata."
        )
        return pd.DataFrame(columns=METADATA_COLUMNS), advertencias

    metadata = pd.read_csv(METADATA_PATH, dtype=str).fillna("")
    faltantes = REQUIRED_METADATA_COLUMNS - set(metadata.columns)
    if faltantes:
        advertencias.append(
            "Faltan columnas en experiments_metadata.csv: "
            + ", ".join(sorted(faltantes))
        )
        for columna in faltantes:
            metadata[columna] = ""

    metadata = metadata.loc[:, METADATA_COLUMNS].copy()
    metadata["exp_id"] = metadata["exp_id"].astype(str).str.strip()
    metadata["magnet"] = metadata["magnet"].astype(str).str.strip().str.upper()
    metadata["magnet_pole"] = metadata["magnet_pole"].astype(str).str.strip().str.upper()

    duplicados = metadata["exp_id"].duplicated(keep=False)
    if duplicados.any():
        exp_ids = sorted(metadata.loc[duplicados, "exp_id"].unique())
        advertencias.append(
            "Metadata duplicada para exp_id: "
            + ", ".join(exp_ids)
            + ". Se usara la primera fila de cada experimento."
        )
        metadata = metadata.drop_duplicates(subset=["exp_id"], keep="first")

    return metadata, advertencias


def a_numerico(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def construir_etiqueta_tratamiento(fila: pd.Series) -> str:
    treatment_floor = fila["treatment_floor"]
    if pd.isna(treatment_floor):
        return ""

    if fila["has_metadata"] and fila["magnet"] == MAGNET_NONE_VALUE:
        return "Control negativo"

    if fila["has_active_magnet"] and treatment_floor in MAGNETIC_TREATMENT_FLOORS:
        return f"Mag {int(treatment_floor)}"

    if fila["has_active_magnet"] and treatment_floor == CONTROL_TREATMENT_FLOOR:
        return "Control"

    return f"Floor {int(treatment_floor)}"


def enriquecer_tratamientos(enriched: pd.DataFrame) -> pd.DataFrame:
    enriched["has_active_magnet"] = (
        enriched["has_metadata"] & enriched["magnet"].ne(MAGNET_NONE_VALUE)
    )

    es_control_negativo = enriched["has_metadata"] & enriched["magnet"].eq(
        MAGNET_NONE_VALUE
    )
    es_tratamiento_magnetico = (
        enriched["has_active_magnet"]
        & enriched["treatment_floor"].isin(MAGNETIC_TREATMENT_FLOORS)
    )
    es_control_interno = (
        enriched["has_active_magnet"]
        & enriched["treatment_floor"].eq(CONTROL_TREATMENT_FLOOR)
    )

    treatment_role = pd.Series(pd.NA, index=enriched.index, dtype="string")
    treatment_role.loc[es_tratamiento_magnetico] = "magnetic"
    treatment_role.loc[es_control_interno | es_control_negativo] = "control"
    enriched["treatment_role"] = treatment_role
    enriched["treatment_role_label"] = treatment_role.map(
        {
            "magnetic": "Tratamiento magnetico",
            "control": "Control",
        }
    ).astype("string")

    magnetic_intensity_rank = pd.Series(
        pd.array([pd.NA] * len(enriched), dtype="Int64"),
        index=enriched.index,
    )
    magnetic_intensity_rank.loc[es_tratamiento_magnetico] = (
        enriched.loc[es_tratamiento_magnetico, "treatment_floor"].astype("Int64")
    )
    enriched["magnetic_intensity_rank"] = magnetic_intensity_rank
    enriched["treatment_label"] = (
        enriched.apply(construir_etiqueta_tratamiento, axis=1).astype("string")
    )
    return enriched


def construir_warning_row(fila: pd.Series) -> str:
    mensajes: list[str] = []
    if not fila["has_metadata"]:
        mensajes.append("missing_metadata")
    if not fila["has_complete_weights"]:
        mensajes.append("missing_weight")
    elif not fila["valid_delta"]:
        mensajes.append("invalid_delta")
    if fila["has_complete_weights"] and not fila["valid_pct_change"]:
        mensajes.append("nonpositive_pre_weight")
    if fila["has_complete_weights"] and not fila["valid_log_ratio"]:
        mensajes.append("nonpositive_weight_for_log_ratio")
    return ";".join(mensajes)


def calcular_factor_normalizacion_control(
    enriched: pd.DataFrame,
    *,
    exp_name: str,
) -> tuple[float, int, list[str]]:
    advertencias: list[str] = []
    mask_control = (
        enriched["treatment_floor"].eq(CONTROL_TREATMENT_FLOOR)
        & enriched["has_complete_weights"]
    )
    deltas_control = (
        enriched.loc[mask_control, "post_weight"] - enriched.loc[mask_control, "pre_weight"]
    ).dropna()
    cantidad_controles = int(len(deltas_control))

    if cantidad_controles == 0:
        advertencias.append(
            f"El experimento {exp_name} no tiene controles internos validos "
            f"(treatment_floor={CONTROL_TREATMENT_FLOOR}) para normalizar pesos."
        )
        return np.nan, cantidad_controles, advertencias

    if cantidad_controles != 3:
        advertencias.append(
            f"El experimento {exp_name} esperaba 3 deltas de control para normalizar, "
            f"pero encontro {cantidad_controles}. Se uso el promedio disponible."
        )

    factor = float(deltas_control.mean())
    if np.isclose(factor, 0.0):
        advertencias.append(
            f"El experimento {exp_name} tiene promedio de delta de control ~ 0; "
            "no se normalizaron los pesos."
        )
        return np.nan, cantidad_controles, advertencias

    return factor, cantidad_controles, advertencias


def calcular_media_log_control(
    enriched: pd.DataFrame,
    *,
    exp_name: str,
) -> tuple[float, int, list[str]]:
    advertencias: list[str] = []
    mask_control = (
        enriched["treatment_floor"].eq(CONTROL_TREATMENT_FLOOR)
        & enriched["has_complete_weights"]
        & (enriched["raw_pre_weight"] > 0)
        & (enriched["raw_post_weight"] > 0)
    )
    log_ratios_control = np.log(
        enriched.loc[mask_control, "raw_post_weight"]
        / enriched.loc[mask_control, "raw_pre_weight"]
    )
    log_ratios_control = log_ratios_control.replace([np.inf, -np.inf], np.nan).dropna()
    cantidad_controles = int(len(log_ratios_control))

    if cantidad_controles == 0:
        advertencias.append(
            f"El experimento {exp_name} no tiene controles internos validos "
            f"(treatment_floor={CONTROL_TREATMENT_FLOOR}) para centrar log(post/pre)."
        )
        return np.nan, cantidad_controles, advertencias

    if cantidad_controles != 3:
        advertencias.append(
            f"El experimento {exp_name} esperaba 3 controles para centrar log(post/pre), "
            f"pero encontro {cantidad_controles}. Se uso el promedio disponible."
        )

    return float(log_ratios_control.mean()), cantidad_controles, advertencias


def cargar_experimento(trays_path: Path, metadata: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    advertencias: list[str] = []
    datos = pd.read_csv(trays_path, dtype=str)

    faltantes = REQUIRED_TRAY_COLUMNS - set(datos.columns)
    if faltantes:
        raise ValueError(
            f"El archivo '{trays_path}' no tiene las columnas requeridas: "
            + ", ".join(sorted(faltantes))
        )

    datos = datos.loc[:, TRAY_COLUMNS].copy()
    for columna in ("tray_id", "exp_id", "floor"):
        datos[columna] = datos[columna].astype(str).str.strip()

    datos["floor"] = datos["floor"].str.upper()
    datos["treatment_floor"] = a_numerico(datos["treatment_floor"])
    datos["position"] = a_numerico(datos["position"])
    datos["pre_weight"] = a_numerico(datos["pre_weight"])
    datos["post_weight"] = a_numerico(datos["post_weight"])

    datos["source_trays_csv"] = str(trays_path)
    datos["exp_dir"] = str(trays_path.parent)

    enriched = datos.merge(metadata, on="exp_id", how="left")
    enriched["magnet"] = enriched["magnet"].fillna("").astype(str).str.strip().str.upper()
    enriched["magnet_pole"] = (
        enriched["magnet_pole"].fillna("").astype(str).str.strip().str.upper()
    )

    enriched["has_metadata"] = (
        enriched["magnet"].ne("") & enriched["magnet_pole"].ne("")
    )
    enriched = enriquecer_tratamientos(enriched)
    enriched["has_complete_weights"] = (
        enriched["pre_weight"].notna() & enriched["post_weight"].notna()
    )
    enriched["raw_pre_weight"] = enriched["pre_weight"]
    enriched["raw_post_weight"] = enriched["post_weight"]
    enriched["raw_delta_weight"] = np.where(
        enriched["has_complete_weights"],
        enriched["post_weight"] - enriched["pre_weight"],
        np.nan,
    )
    factor_normalizacion, cantidad_controles, advertencias_normalizacion = (
        calcular_factor_normalizacion_control(enriched, exp_name=trays_path.parent.name)
    )
    advertencias.extend(advertencias_normalizacion)
    enriched["control_delta_mean"] = factor_normalizacion
    enriched["control_delta_count"] = cantidad_controles
    enriched["weight_normalization_method"] = (
        "weights_divided_by_mean_control_delta"
        if np.isfinite(factor_normalizacion)
        else "not_normalized_missing_control_delta"
    )
    if np.isfinite(factor_normalizacion):
        enriched["pre_weight"] = enriched["pre_weight"] / factor_normalizacion
        enriched["post_weight"] = enriched["post_weight"] / factor_normalizacion

    enriched["delta_weight"] = np.where(
        enriched["has_complete_weights"],
        enriched["post_weight"] - enriched["pre_weight"],
        np.nan,
    )
    enriched["valid_delta"] = enriched["delta_weight"].notna()
    enriched["valid_pct_change"] = enriched["valid_delta"] & (enriched["pre_weight"] > 0)
    enriched["pct_change"] = np.where(
        enriched["valid_pct_change"],
        enriched["delta_weight"] / enriched["pre_weight"],
        np.nan,
    )
    enriched["valid_log_ratio"] = (
        enriched["has_complete_weights"]
        & (enriched["raw_pre_weight"] > 0)
        & (enriched["raw_post_weight"] > 0)
    )
    enriched["raw_log_ratio"] = np.nan
    valid_log_mask = enriched["valid_log_ratio"]
    enriched.loc[valid_log_mask, "raw_log_ratio"] = np.log(
        enriched.loc[valid_log_mask, "raw_post_weight"]
        / enriched.loc[valid_log_mask, "raw_pre_weight"]
    )
    (
        control_log_ratio_mean,
        control_log_ratio_count,
        advertencias_log_ratio,
    ) = calcular_media_log_control(enriched, exp_name=trays_path.parent.name)
    advertencias.extend(advertencias_log_ratio)
    enriched["control_log_ratio_mean"] = control_log_ratio_mean
    enriched["control_log_ratio_count"] = control_log_ratio_count
    enriched["log_ratio_centering_method"] = (
        "log_ratio_minus_mean_control_log_ratio"
        if np.isfinite(control_log_ratio_mean)
        else "not_centered_missing_control_log_ratio"
    )
    enriched["log_ratio_centered"] = np.where(
        enriched["valid_log_ratio"] & np.isfinite(control_log_ratio_mean),
        enriched["raw_log_ratio"] - control_log_ratio_mean,
        np.nan,
    )
    enriched["warning_messages"] = enriched.apply(construir_warning_row, axis=1)

    if not enriched["has_metadata"].all():
        advertencias.append(
            f"El experimento {trays_path.parent.name} no tiene metadata completa."
        )

    if (~enriched["has_complete_weights"]).any():
        faltantes_pesos = int((~enriched["has_complete_weights"]).sum())
        advertencias.append(
            f"El experimento {trays_path.parent.name} tiene {faltantes_pesos} filas con pesos faltantes."
        )

    if ((enriched["has_complete_weights"]) & (~enriched["valid_pct_change"])).any():
        advertencias.append(
            f"El experimento {trays_path.parent.name} tiene filas con pre_weight <= 0."
        )
    if ((enriched["has_complete_weights"]) & (~enriched["valid_log_ratio"])).any():
        advertencias.append(
            f"El experimento {trays_path.parent.name} tiene filas con pre_weight <= 0 o post_weight <= 0; no se pudo calcular log(post/pre)."
        )

    enriched["treatment_floor"] = enriched["treatment_floor"].astype("Int64")
    enriched["position"] = enriched["position"].astype("Int64")
    enriched["floor"] = pd.Categorical(enriched["floor"], categories=FLOOR_ORDER, ordered=True)
    enriched.sort_values(
        by=["floor", "position", "tray_id"], inplace=True, ignore_index=True
    )
    return enriched, advertencias


def resumen_grupo(
    data: pd.DataFrame,
    *,
    analysis_scope: str,
    source_exp_id: str,
    group_by: str,
    group_columns: list[str],
) -> pd.DataFrame:
    filas: list[dict[str, Any]] = []

    for metrica in SUMMARY_METRICS:
        if group_columns:
            total = (
                data.groupby(group_columns, dropna=False, observed=False)
                .size()
                .rename("n_total")
                .reset_index()
            )
            validos = data.dropna(subset=[metrica])
            stats_df = (
                validos.groupby(group_columns, dropna=False, observed=False)[metrica]
                .agg(
                    n_valid="count",
                    mean="mean",
                    std="std",
                    median="median",
                    min="min",
                    max="max",
                )
                .reset_index()
            )
            combinado = total.merge(stats_df, on=group_columns, how="left")
        else:
            serie = data[metrica].dropna()
            combinado = pd.DataFrame(
                [
                    {
                        "n_total": len(data),
                        "n_valid": int(serie.count()),
                        "mean": float(serie.mean()) if not serie.empty else np.nan,
                        "std": float(serie.std(ddof=1)) if len(serie) > 1 else np.nan,
                        "median": float(serie.median()) if not serie.empty else np.nan,
                        "min": float(serie.min()) if not serie.empty else np.nan,
                        "max": float(serie.max()) if not serie.empty else np.nan,
                    }
                ]
            )

        for _, fila in combinado.iterrows():
            registro = {
                "analysis_scope": analysis_scope,
                "source_exp_id": source_exp_id,
                "group_by": group_by,
                "response_metric": metrica,
                "exp_id": pd.NA,
                "magnet": pd.NA,
                "magnet_pole": pd.NA,
                "treatment_floor": pd.NA,
                "floor": pd.NA,
                "position": pd.NA,
                "n_total": int(fila["n_total"]),
                "n_valid": int(fila["n_valid"]) if not pd.isna(fila["n_valid"]) else 0,
                "n_missing": int(fila["n_total"] - (fila["n_valid"] if not pd.isna(fila["n_valid"]) else 0)),
                "mean": fila["mean"],
                "std": fila["std"],
                "median": fila["median"],
                "min": fila["min"],
                "max": fila["max"],
            }
            for columna in group_columns:
                registro[columna] = fila[columna]
            filas.append(registro)

    return pd.DataFrame(filas)


def resumen_experimento(data: pd.DataFrame) -> pd.DataFrame:
    exp_id = str(data["exp_id"].iloc[0])
    partes = [
        resumen_grupo(
            data,
            analysis_scope="experiment",
            source_exp_id=exp_id,
            group_by="overall",
            group_columns=[],
        ),
        resumen_grupo(
            data,
            analysis_scope="experiment",
            source_exp_id=exp_id,
            group_by="treatment_floor",
            group_columns=["treatment_floor"],
        ),
        resumen_grupo(
            data,
            analysis_scope="experiment",
            source_exp_id=exp_id,
            group_by="treatment_role",
            group_columns=["treatment_role"],
        ),
        resumen_grupo(
            data,
            analysis_scope="experiment",
            source_exp_id=exp_id,
            group_by="treatment_label",
            group_columns=["treatment_label"],
        ),
        resumen_grupo(
            data,
            analysis_scope="experiment",
            source_exp_id=exp_id,
            group_by="floor",
            group_columns=["floor"],
        ),
        resumen_grupo(
            data,
            analysis_scope="experiment",
            source_exp_id=exp_id,
            group_by="position",
            group_columns=["position"],
        ),
    ]
    return pd.concat(partes, ignore_index=True)


def resumen_global(data: pd.DataFrame) -> pd.DataFrame:
    partes = [
        resumen_grupo(
            data,
            analysis_scope="global",
            source_exp_id="all",
            group_by="overall",
            group_columns=[],
        ),
        resumen_grupo(
            data,
            analysis_scope="global",
            source_exp_id="all",
            group_by="exp_id",
            group_columns=["exp_id"],
        ),
        resumen_grupo(
            data,
            analysis_scope="global",
            source_exp_id="all",
            group_by="magnet",
            group_columns=["magnet"],
        ),
        resumen_grupo(
            data,
            analysis_scope="global",
            source_exp_id="all",
            group_by="magnet_pole",
            group_columns=["magnet_pole"],
        ),
        resumen_grupo(
            data,
            analysis_scope="global",
            source_exp_id="all",
            group_by="magnet_and_pole",
            group_columns=["magnet", "magnet_pole"],
        ),
        resumen_grupo(
            data,
            analysis_scope="global",
            source_exp_id="all",
            group_by="treatment_role",
            group_columns=["treatment_role"],
        ),
        resumen_grupo(
            data,
            analysis_scope="global",
            source_exp_id="all",
            group_by="treatment_label",
            group_columns=["treatment_label"],
        ),
    ]
    return pd.concat(partes, ignore_index=True)


def _extraer_stat_pvalue(resultado: Any) -> tuple[float, float]:
    if hasattr(resultado, "statistic") and hasattr(resultado, "pvalue"):
        return float(resultado.statistic), float(resultado.pvalue)
    return float(resultado[0]), float(resultado[1])


def correlaciones_experimento(data: pd.DataFrame, exp_id: str) -> pd.DataFrame:
    filas: list[dict[str, Any]] = []

    for metrica in ("delta_weight", "pct_change", "log_ratio_centered"):
        subset = data.dropna(subset=["magnetic_intensity_rank", metrica]).copy()
        if len(subset) < 3:
            if not bool(data["has_active_magnet"].any()):
                notes = (
                    "El experimento es control negativo (magnet=NONE), "
                    "asi que no hay intensidad magnetica para correlacionar."
                )
            else:
                notes = (
                    "No alcanza para calcular correlaciones entre intensidad magnetica "
                    "y respuesta."
                )
            filas.append(
                {
                    "analysis_scope": "experiment",
                    "source_exp_id": exp_id,
                    "response_metric": metrica,
                    "analysis_type": "correlation",
                    "model_name": pd.NA,
                    "formula": pd.NA,
                    "method": "insufficient_data",
                    "term": "magnetic_intensity_rank",
                    "estimate": np.nan,
                    "std_error": np.nan,
                    "statistic": np.nan,
                    "statistic_name": pd.NA,
                    "p_value": np.nan,
                    "conf_int_low": np.nan,
                    "conf_int_high": np.nan,
                    "sum_sq": np.nan,
                    "df_term": np.nan,
                    "n_obs": len(subset),
                    "rsquared": np.nan,
                    "adj_rsquared": np.nan,
                    "notes": notes,
                }
            )
            continue

        if subset["magnetic_intensity_rank"].nunique() < 2 or subset[metrica].nunique() < 2:
            filas.append(
                {
                    "analysis_scope": "experiment",
                    "source_exp_id": exp_id,
                    "response_metric": metrica,
                    "analysis_type": "correlation",
                    "model_name": pd.NA,
                    "formula": pd.NA,
                    "method": "constant_input",
                    "term": "magnetic_intensity_rank",
                    "estimate": np.nan,
                    "std_error": np.nan,
                    "statistic": np.nan,
                    "statistic_name": pd.NA,
                    "p_value": np.nan,
                    "conf_int_low": np.nan,
                    "conf_int_high": np.nan,
                    "sum_sq": np.nan,
                    "df_term": np.nan,
                    "n_obs": len(subset),
                    "rsquared": np.nan,
                    "adj_rsquared": np.nan,
                    "notes": "No se pudo calcular por falta de variacion.",
                }
            )
            continue

        for metodo, funcion in (
            ("pearson", stats.pearsonr),
            ("spearman", stats.spearmanr),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estadistico, p_value = _extraer_stat_pvalue(
                    funcion(
                        subset["magnetic_intensity_rank"].astype(float),
                        subset[metrica].astype(float),
                    )
                )

            filas.append(
                {
                    "analysis_scope": "experiment",
                    "source_exp_id": exp_id,
                    "response_metric": metrica,
                    "analysis_type": "correlation",
                    "model_name": pd.NA,
                    "formula": pd.NA,
                    "method": metodo,
                    "term": "magnetic_intensity_rank",
                    "estimate": estadistico,
                    "std_error": np.nan,
                    "statistic": estadistico,
                    "statistic_name": "r" if metodo == "pearson" else "rho",
                    "p_value": p_value,
                    "conf_int_low": np.nan,
                    "conf_int_high": np.nan,
                    "sum_sq": np.nan,
                    "df_term": np.nan,
                    "n_obs": len(subset),
                    "rsquared": np.nan,
                    "adj_rsquared": np.nan,
                    "notes": pd.NA,
                }
            )

    return pd.DataFrame(filas)


def modelo_no_disponible(
    *,
    exp_id: str,
    response_metric: str,
    model_name: str,
    formula: str,
    method: str,
    notes: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "analysis_scope": "experiment",
                "source_exp_id": exp_id,
                "response_metric": response_metric,
                "analysis_type": "model_not_run",
                "model_name": model_name,
                "formula": formula,
                "method": method,
                "term": pd.NA,
                "estimate": np.nan,
                "std_error": np.nan,
                "statistic": np.nan,
                "statistic_name": pd.NA,
                "p_value": np.nan,
                "conf_int_low": np.nan,
                "conf_int_high": np.nan,
                "sum_sq": np.nan,
                "df_term": np.nan,
                "n_obs": 0,
                "rsquared": np.nan,
                "adj_rsquared": np.nan,
                "notes": notes,
            }
        ]
    )


def ajustar_modelo(
    data: pd.DataFrame,
    *,
    exp_id: str,
    response_metric: str,
    model_name: str,
    formula: str,
) -> pd.DataFrame:
    filas: list[dict[str, Any]] = []

    if sm is None or smf is None:
        raise RuntimeError(
            "statsmodels no esta instalado. Instala dependencias con 'pip install -r requirements.txt'."
        )

    model_data = data.copy()
    model_data["floor"] = model_data["floor"].astype(str)
    model_data["pre_weight"] = pd.to_numeric(model_data["pre_weight"], errors="coerce")
    model_data["treatment_floor"] = pd.to_numeric(
        model_data["treatment_floor"], errors="coerce"
    )
    model_data["magnetic_intensity_rank"] = pd.to_numeric(
        model_data.get("magnetic_intensity_rank"), errors="coerce"
    )
    model_data[response_metric] = pd.to_numeric(
        model_data[response_metric], errors="coerce"
    )
    model_data["position_factor"] = model_data["position"].astype("string")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            resultado = smf.ols(formula=formula, data=model_data).fit()
        except Exception as exc:
            filas.append(
                {
                    "analysis_scope": "experiment",
                    "source_exp_id": exp_id,
                    "response_metric": response_metric,
                    "analysis_type": "model_error",
                    "model_name": model_name,
                    "formula": formula,
                    "method": "ols",
                    "term": pd.NA,
                    "estimate": np.nan,
                    "std_error": np.nan,
                    "statistic": np.nan,
                    "statistic_name": pd.NA,
                    "p_value": np.nan,
                    "conf_int_low": np.nan,
                    "conf_int_high": np.nan,
                    "sum_sq": np.nan,
                    "df_term": np.nan,
                    "n_obs": 0,
                    "rsquared": np.nan,
                    "adj_rsquared": np.nan,
                    "notes": str(exc),
                }
            )
            return pd.DataFrame(filas)

    conf_int = resultado.conf_int()
    for termino, estimate in resultado.params.items():
        filas.append(
            {
                "analysis_scope": "experiment",
                "source_exp_id": exp_id,
                "response_metric": response_metric,
                "analysis_type": "coefficient",
                "model_name": model_name,
                "formula": formula,
                "method": "ols",
                "term": termino,
                "estimate": estimate,
                "std_error": resultado.bse.get(termino, np.nan),
                "statistic": resultado.tvalues.get(termino, np.nan),
                "statistic_name": "t",
                "p_value": resultado.pvalues.get(termino, np.nan),
                "conf_int_low": conf_int.loc[termino, 0],
                "conf_int_high": conf_int.loc[termino, 1],
                "sum_sq": np.nan,
                "df_term": np.nan,
                "n_obs": int(resultado.nobs),
                "rsquared": resultado.rsquared,
                "adj_rsquared": resultado.rsquared_adj,
                "notes": pd.NA,
            }
        )

    try:
        anova = sm.stats.anova_lm(resultado, typ=2).reset_index()
        anova.rename(columns={"index": "term", "PR(>F)": "p_value"}, inplace=True)
        anova = anova[anova["term"] != "Residual"]
        for _, fila in anova.iterrows():
            filas.append(
                {
                    "analysis_scope": "experiment",
                    "source_exp_id": exp_id,
                    "response_metric": response_metric,
                    "analysis_type": "term_test",
                    "model_name": model_name,
                    "formula": formula,
                    "method": "anova_typ2",
                    "term": fila["term"],
                    "estimate": np.nan,
                    "std_error": np.nan,
                    "statistic": fila.get("F", np.nan),
                    "statistic_name": "F",
                    "p_value": fila.get("p_value", np.nan),
                    "conf_int_low": np.nan,
                    "conf_int_high": np.nan,
                    "sum_sq": fila.get("sum_sq", np.nan),
                    "df_term": fila.get("df", np.nan),
                    "n_obs": int(resultado.nobs),
                    "rsquared": resultado.rsquared,
                    "adj_rsquared": resultado.rsquared_adj,
                    "notes": pd.NA,
                }
            )
    except Exception as exc:
        filas.append(
            {
                "analysis_scope": "experiment",
                "source_exp_id": exp_id,
                "response_metric": response_metric,
                "analysis_type": "term_test_error",
                "model_name": model_name,
                "formula": formula,
                "method": "anova_typ2",
                "term": pd.NA,
                "estimate": np.nan,
                "std_error": np.nan,
                "statistic": np.nan,
                "statistic_name": pd.NA,
                "p_value": np.nan,
                "conf_int_low": np.nan,
                "conf_int_high": np.nan,
                "sum_sq": np.nan,
                "df_term": np.nan,
                "n_obs": int(resultado.nobs),
                "rsquared": resultado.rsquared,
                "adj_rsquared": resultado.rsquared_adj,
                "notes": f"No se pudo calcular ANOVA: {exc}",
            }
        )

    return pd.DataFrame(filas)


def modelos_experimento(data: pd.DataFrame) -> pd.DataFrame:
    exp_id = str(data["exp_id"].iloc[0])
    modelos: list[pd.DataFrame] = [correlaciones_experimento(data, exp_id)]

    intensity_data = data.dropna(subset=["magnetic_intensity_rank"]).copy()
    intensity_formula_delta = (
        "delta_weight ~ pre_weight + magnetic_intensity_rank + C(floor) + C(position_factor)"
    )
    intensity_formula_pct = (
        "pct_change ~ magnetic_intensity_rank + C(floor) + C(position_factor)"
    )
    intensity_formula_log = (
        "log_ratio_centered ~ magnetic_intensity_rank + C(floor) + C(position_factor)"
    )

    if bool(data["has_active_magnet"].any()) and intensity_data["magnetic_intensity_rank"].nunique() >= 2:
        modelos.extend(
            [
                ajustar_modelo(
                    intensity_data,
                    exp_id=exp_id,
                    response_metric="delta_weight",
                    model_name="delta_magnetic_intensity",
                    formula=intensity_formula_delta,
                ),
                ajustar_modelo(
                    intensity_data,
                    exp_id=exp_id,
                    response_metric="pct_change",
                    model_name="pct_magnetic_intensity",
                    formula=intensity_formula_pct,
                ),
                ajustar_modelo(
                    intensity_data,
                    exp_id=exp_id,
                    response_metric="log_ratio_centered",
                    model_name="log_ratio_centered_magnetic_intensity",
                    formula=intensity_formula_log,
                ),
            ]
        )
    else:
        reason = (
            "El experimento es control negativo (magnet=NONE), asi que no hay "
            "intensidad magnetica que modelar."
            if not bool(data["has_active_magnet"].any())
            else "No hay suficientes niveles de intensidad magnetica para ajustar el modelo."
        )
        modelos.extend(
            [
                modelo_no_disponible(
                    exp_id=exp_id,
                    response_metric="delta_weight",
                    model_name="delta_magnetic_intensity",
                    formula=intensity_formula_delta,
                    method="skipped",
                    notes=reason,
                ),
                modelo_no_disponible(
                    exp_id=exp_id,
                    response_metric="pct_change",
                    model_name="pct_magnetic_intensity",
                    formula=intensity_formula_pct,
                    method="skipped",
                    notes=reason,
                ),
                modelo_no_disponible(
                    exp_id=exp_id,
                    response_metric="log_ratio_centered",
                    model_name="log_ratio_centered_magnetic_intensity",
                    formula=intensity_formula_log,
                    method="skipped",
                    notes=reason,
                ),
            ]
        )

    comparison_data = data[data["treatment_role"].isin(["magnetic", "control"])].copy()
    comparison_formula_delta = (
        "delta_weight ~ pre_weight + C(treatment_role) + C(floor) + C(position_factor)"
    )
    comparison_formula_pct = (
        "pct_change ~ C(treatment_role) + C(floor) + C(position_factor)"
    )
    comparison_formula_log = (
        "log_ratio_centered ~ C(treatment_role) + C(floor) + C(position_factor)"
    )

    if comparison_data["treatment_role"].nunique() >= 2:
        modelos.extend(
            [
                ajustar_modelo(
                    comparison_data,
                    exp_id=exp_id,
                    response_metric="delta_weight",
                    model_name="delta_magnetic_vs_control",
                    formula=comparison_formula_delta,
                ),
                ajustar_modelo(
                    comparison_data,
                    exp_id=exp_id,
                    response_metric="pct_change",
                    model_name="pct_magnetic_vs_control",
                    formula=comparison_formula_pct,
                ),
                ajustar_modelo(
                    comparison_data,
                    exp_id=exp_id,
                    response_metric="log_ratio_centered",
                    model_name="log_ratio_centered_magnetic_vs_control",
                    formula=comparison_formula_log,
                ),
            ]
        )
    else:
        modelos.extend(
            [
                modelo_no_disponible(
                    exp_id=exp_id,
                    response_metric="delta_weight",
                    model_name="delta_magnetic_vs_control",
                    formula=comparison_formula_delta,
                    method="skipped",
                    notes=(
                        "No hay mezcla de grupos magnetic/control para comparar en este experimento."
                    ),
                ),
                modelo_no_disponible(
                    exp_id=exp_id,
                    response_metric="pct_change",
                    model_name="pct_magnetic_vs_control",
                    formula=comparison_formula_pct,
                    method="skipped",
                    notes=(
                        "No hay mezcla de grupos magnetic/control para comparar en este experimento."
                    ),
                ),
                modelo_no_disponible(
                    exp_id=exp_id,
                    response_metric="log_ratio_centered",
                    model_name="log_ratio_centered_magnetic_vs_control",
                    formula=comparison_formula_log,
                    method="skipped",
                    notes=(
                        "No hay mezcla de grupos magnetic/control para comparar en este experimento."
                    ),
                ),
            ]
        )

    return pd.concat(modelos, ignore_index=True)


def comparaciones_tratamiento_vs_control_experimento(data: pd.DataFrame) -> pd.DataFrame:
    exp_id = str(data["exp_id"].iloc[0])
    filas: list[dict[str, Any]] = []

    subset = data[
        data["treatment_label"].isin(
            ["Control", *(f"Mag {i}" for i in MAGNETIC_TREATMENT_FLOORS)]
        )
    ].copy()
    subset = subset.dropna(subset=["treatment_label"])

    if subset.empty or "Control" not in subset["treatment_label"].unique():
        return pd.DataFrame(
            [
                {
                    "analysis_scope": "experiment",
                    "source_exp_id": exp_id,
                    "response_metric": pd.NA,
                    "treatment_label": pd.NA,
                    "control_label": "Control",
                    "analysis_type": "treatment_vs_control_not_run",
                    "method": "skipped",
                    "estimate_diff_vs_control": np.nan,
                    "std_error": np.nan,
                    "statistic": np.nan,
                    "statistic_name": "t",
                    "p_value": np.nan,
                    "p_value_adjusted": np.nan,
                    "reject_null_fdr_bh": pd.NA,
                    "conf_int_low": np.nan,
                    "conf_int_high": np.nan,
                    "n_treatment": 0,
                    "n_control": 0,
                    "mean_treatment": np.nan,
                    "mean_control": np.nan,
                    "notes": "No hay grupo Control disponible para comparar.",
                }
            ]
        )

    subset["position_factor"] = subset["position"].astype("string")
    subset["floor"] = subset["floor"].astype(str)
    subset["treatment_label"] = subset["treatment_label"].astype(str)

    for response_metric in ("delta_weight", "pct_change", "log_ratio_centered"):
        model_data = subset.copy()
        model_data[response_metric] = pd.to_numeric(
            model_data[response_metric], errors="coerce"
        )
        model_data = model_data.dropna(subset=[response_metric])

        if model_data.empty:
            filas.append(
                {
                    "analysis_scope": "experiment",
                    "source_exp_id": exp_id,
                    "response_metric": response_metric,
                    "treatment_label": pd.NA,
                    "control_label": "Control",
                    "analysis_type": "treatment_vs_control_not_run",
                    "method": "skipped",
                    "estimate_diff_vs_control": np.nan,
                    "std_error": np.nan,
                    "statistic": np.nan,
                    "statistic_name": "t",
                    "p_value": np.nan,
                    "p_value_adjusted": np.nan,
                    "reject_null_fdr_bh": pd.NA,
                    "conf_int_low": np.nan,
                    "conf_int_high": np.nan,
                    "n_treatment": 0,
                    "n_control": 0,
                    "mean_treatment": np.nan,
                    "mean_control": np.nan,
                    "notes": "No hay datos validos para esta metrica.",
                }
            )
            continue

        tratamientos = [
            etiqueta
            for etiqueta in sorted(model_data["treatment_label"].unique().tolist())
            if etiqueta.startswith("Mag ")
        ]
        if not tratamientos:
            filas.append(
                {
                    "analysis_scope": "experiment",
                    "source_exp_id": exp_id,
                    "response_metric": response_metric,
                    "treatment_label": pd.NA,
                    "control_label": "Control",
                    "analysis_type": "treatment_vs_control_not_run",
                    "method": "skipped",
                    "estimate_diff_vs_control": np.nan,
                    "std_error": np.nan,
                    "statistic": np.nan,
                    "statistic_name": "t",
                    "p_value": np.nan,
                    "p_value_adjusted": np.nan,
                    "reject_null_fdr_bh": pd.NA,
                    "conf_int_low": np.nan,
                    "conf_int_high": np.nan,
                    "n_treatment": 0,
                    "n_control": int(model_data["treatment_label"].eq("Control").sum()),
                    "mean_treatment": np.nan,
                    "mean_control": float(
                        model_data.loc[
                            model_data["treatment_label"].eq("Control"), response_metric
                        ].mean()
                    ),
                    "notes": "No hay tratamientos magneticos para comparar.",
                }
            )
            continue

        formula = (
            f"{response_metric} ~ "
            "C(treatment_label, Treatment(reference='Control')) + "
            "C(floor) + C(position_factor)"
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resultado = smf.ols(formula=formula, data=model_data).fit()
        except Exception as exc:
            filas.append(
                {
                    "analysis_scope": "experiment",
                    "source_exp_id": exp_id,
                    "response_metric": response_metric,
                    "treatment_label": pd.NA,
                    "control_label": "Control",
                    "analysis_type": "treatment_vs_control_model_error",
                    "method": "ols",
                    "estimate_diff_vs_control": np.nan,
                    "std_error": np.nan,
                    "statistic": np.nan,
                    "statistic_name": "t",
                    "p_value": np.nan,
                    "p_value_adjusted": np.nan,
                    "reject_null_fdr_bh": pd.NA,
                    "conf_int_low": np.nan,
                    "conf_int_high": np.nan,
                    "n_treatment": 0,
                    "n_control": int(model_data["treatment_label"].eq("Control").sum()),
                    "mean_treatment": np.nan,
                    "mean_control": float(
                        model_data.loc[
                            model_data["treatment_label"].eq("Control"), response_metric
                        ].mean()
                    ),
                    "notes": str(exc),
                }
            )
            continue

        conf_int = resultado.conf_int()
        metric_rows: list[dict[str, Any]] = []
        for tratamiento in tratamientos:
            termino = (
                "C(treatment_label, Treatment(reference='Control'))"
                f"[T.{tratamiento}]"
            )
            n_treatment = int(model_data["treatment_label"].eq(tratamiento).sum())
            n_control = int(model_data["treatment_label"].eq("Control").sum())
            mean_treatment = float(
                model_data.loc[
                    model_data["treatment_label"].eq(tratamiento), response_metric
                ].mean()
            )
            mean_control = float(
                model_data.loc[
                    model_data["treatment_label"].eq("Control"), response_metric
                ].mean()
            )

            if termino not in resultado.params.index:
                metric_rows.append(
                    {
                        "analysis_scope": "experiment",
                        "source_exp_id": exp_id,
                        "response_metric": response_metric,
                        "treatment_label": tratamiento,
                        "control_label": "Control",
                        "analysis_type": "treatment_vs_control",
                        "method": "ols",
                        "estimate_diff_vs_control": np.nan,
                        "std_error": np.nan,
                        "statistic": np.nan,
                        "statistic_name": "t",
                        "p_value": np.nan,
                        "p_value_adjusted": np.nan,
                        "reject_null_fdr_bh": pd.NA,
                        "conf_int_low": np.nan,
                        "conf_int_high": np.nan,
                        "n_treatment": n_treatment,
                        "n_control": n_control,
                        "mean_treatment": mean_treatment,
                        "mean_control": mean_control,
                        "notes": "El termino no pudo estimarse en el modelo.",
                    }
                )
                continue

            metric_rows.append(
                {
                    "analysis_scope": "experiment",
                    "source_exp_id": exp_id,
                    "response_metric": response_metric,
                    "treatment_label": tratamiento,
                    "control_label": "Control",
                    "analysis_type": "treatment_vs_control",
                    "method": "ols",
                    "estimate_diff_vs_control": float(resultado.params[termino]),
                    "std_error": float(resultado.bse.get(termino, np.nan)),
                    "statistic": float(resultado.tvalues.get(termino, np.nan)),
                    "statistic_name": "t",
                    "p_value": float(resultado.pvalues.get(termino, np.nan)),
                    "p_value_adjusted": np.nan,
                    "reject_null_fdr_bh": pd.NA,
                    "conf_int_low": float(conf_int.loc[termino, 0]),
                    "conf_int_high": float(conf_int.loc[termino, 1]),
                    "n_treatment": n_treatment,
                    "n_control": n_control,
                    "mean_treatment": mean_treatment,
                    "mean_control": mean_control,
                    "notes": pd.NA,
                }
            )

        pvals = [fila["p_value"] for fila in metric_rows if pd.notna(fila["p_value"])]
        if pvals and multipletests is not None:
            rechazo, pvals_ajustados, _, _ = multipletests(pvals, method="fdr_bh")
            idx = 0
            for fila in metric_rows:
                if pd.notna(fila["p_value"]):
                    fila["p_value_adjusted"] = float(pvals_ajustados[idx])
                    fila["reject_null_fdr_bh"] = bool(rechazo[idx])
                    idx += 1

        filas.extend(metric_rows)

    return pd.DataFrame(filas)


def guardar_boxplot_stripplot(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    titulo: str,
    ylabel: str,
    output_path: Path,
    order: list[Any] | None = None,
    xlabel: str | None = None,
) -> None:
    plot_data = data.dropna(subset=[x, y]).copy()
    if plot_data.empty:
        return

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=plot_data,
        x=x,
        y=y,
        order=order,
        showfliers=False,
        color="#9ecae1",
        ax=ax,
    )
    sns.stripplot(
        data=plot_data,
        x=x,
        y=y,
        order=order,
        color="#0b3954",
        alpha=0.75,
        size=5,
        ax=ax,
    )
    ax.set_title(titulo)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def guardar_heatmap(
    data: pd.DataFrame,
    *,
    value_column: str,
    titulo: str,
    output_path: Path,
) -> None:
    plot_data = data.dropna(subset=["floor", "position", value_column]).copy()
    if plot_data.empty:
        return

    pivot = plot_data.pivot_table(
        index="floor",
        columns="position",
        values=value_column,
        aggfunc="mean",
        observed=False,
    )
    pivot = pivot.reindex(FLOOR_ORDER)
    pivot = pivot.reindex(sorted([col for col in pivot.columns if pd.notna(col)]), axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": value_column},
        ax=ax,
    )
    ax.set_title(titulo)
    ax.set_xlabel("position")
    ax.set_ylabel("floor")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def guardar_forest_plot_tratamiento_vs_control(
    comparisons: pd.DataFrame,
    *,
    response_metric: str,
    output_path: Path,
) -> None:
    plot_data = comparisons[
        (comparisons["analysis_type"] == "treatment_vs_control")
        & (comparisons["response_metric"] == response_metric)
    ].copy()
    plot_data = plot_data.dropna(
        subset=["estimate_diff_vs_control", "conf_int_low", "conf_int_high"]
    )
    if plot_data.empty:
        return

    def _orden_tratamiento(etiqueta: str) -> int:
        try:
            return int(str(etiqueta).replace("Mag ", ""))
        except ValueError:
            return 999

    plot_data = plot_data.sort_values(
        by="treatment_label",
        key=lambda serie: serie.map(_orden_tratamiento),
        ascending=True,
    )

    fig_height = max(4, 0.8 * len(plot_data) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    y = np.arange(len(plot_data))
    estimates = plot_data["estimate_diff_vs_control"].to_numpy(dtype=float)
    lower = plot_data["conf_int_low"].to_numpy(dtype=float)
    upper = plot_data["conf_int_high"].to_numpy(dtype=float)
    xerr = np.vstack([estimates - lower, upper - estimates])

    ax.errorbar(
        estimates,
        y,
        xerr=xerr,
        fmt="o",
        color="#0b3954",
        ecolor="#7f8c8d",
        elinewidth=2,
        capsize=4,
    )
    ax.axvline(0.0, color="#c0392b", linestyle="--", linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_data["treatment_label"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Diferencia estimada vs Control")
    ax.set_ylabel("Tratamiento")
    ax.set_title(METRIC_FOREST_TITLES.get(response_metric, "Tratamiento vs Control"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def guardar_plots_experimento(
    data: pd.DataFrame,
    plots_dir: Path,
    comparisons: pd.DataFrame | None = None,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    orden_role = [
        etiqueta
        for etiqueta in ["Control", "Tratamiento magnetico"]
        if etiqueta in data["treatment_role_label"].dropna().unique().tolist()
    ]
    etiquetas_presentes = [
        etiqueta
        for etiqueta in data["treatment_label"].dropna().unique().tolist()
        if etiqueta
    ]
    orden_treatment = [
        etiqueta
        for etiqueta in [
            *(f"Mag {i}" for i in MAGNETIC_TREATMENT_FLOORS),
            "Control",
            "Control negativo",
        ]
        if etiqueta in etiquetas_presentes
    ]
    orden_treatment.extend(
        sorted(
            etiqueta for etiqueta in etiquetas_presentes if etiqueta not in orden_treatment
        )
    )
    orden_intensidad = sorted(data["magnetic_intensity_rank"].dropna().unique().tolist())
    orden_position = sorted(data["position"].dropna().unique().tolist())
    orden_floor = [floor for floor in FLOOR_ORDER if floor in data["floor"].astype(str).unique()]

    guardar_boxplot_stripplot(
        data,
        x="treatment_label",
        y="delta_weight",
        titulo="Delta de peso por tratamiento",
        ylabel=METRIC_YLABELS["delta_weight"],
        output_path=plots_dir / "trays_delta_by_treatment_floor.png",
        order=orden_treatment,
        xlabel="tratamiento",
    )
    guardar_boxplot_stripplot(
        data,
        x="treatment_label",
        y="pct_change",
        titulo="Cambio relativo por tratamiento",
        ylabel=METRIC_YLABELS["pct_change"],
        output_path=plots_dir / "trays_pct_change_by_treatment_floor.png",
        order=orden_treatment,
        xlabel="tratamiento",
    )
    guardar_boxplot_stripplot(
        data,
        x="treatment_label",
        y="log_ratio_centered",
        titulo="Log ratio centrado por tratamiento",
        ylabel=METRIC_YLABELS["log_ratio_centered"],
        output_path=plots_dir / "trays_log_ratio_centered_by_treatment_floor.png",
        order=orden_treatment,
        xlabel="tratamiento",
    )
    if orden_role:
        data_role = data.dropna(subset=["treatment_role_label"]).copy()
        guardar_boxplot_stripplot(
            data_role,
            x="treatment_role_label",
            y="delta_weight",
            titulo="Delta de peso: control vs tratamiento magnetico",
            ylabel=METRIC_YLABELS["delta_weight"],
            output_path=plots_dir / "trays_delta_control_vs_magnetic.png",
            order=orden_role,
            xlabel="grupo",
        )
        guardar_boxplot_stripplot(
            data_role,
            x="treatment_role_label",
            y="pct_change",
            titulo="Cambio relativo: control vs tratamiento magnetico",
            ylabel=METRIC_YLABELS["pct_change"],
            output_path=plots_dir / "trays_pct_change_control_vs_magnetic.png",
            order=orden_role,
            xlabel="grupo",
        )
        guardar_boxplot_stripplot(
            data_role,
            x="treatment_role_label",
            y="log_ratio_centered",
            titulo="Log ratio centrado: control vs tratamiento magnetico",
            ylabel=METRIC_YLABELS["log_ratio_centered"],
            output_path=plots_dir / "trays_log_ratio_centered_control_vs_magnetic.png",
            order=orden_role,
            xlabel="grupo",
        )
    if orden_intensidad:
        solo_magneticos = data.dropna(subset=["magnetic_intensity_rank"]).copy()
        guardar_boxplot_stripplot(
            solo_magneticos,
            x="magnetic_intensity_rank",
            y="delta_weight",
            titulo="Delta de peso por intensidad magnetica",
            ylabel=METRIC_YLABELS["delta_weight"],
            output_path=plots_dir / "trays_delta_by_magnetic_intensity.png",
            order=orden_intensidad,
            xlabel="intensidad magnetica",
        )
        guardar_boxplot_stripplot(
            solo_magneticos,
            x="magnetic_intensity_rank",
            y="pct_change",
            titulo="Cambio relativo por intensidad magnetica",
            ylabel=METRIC_YLABELS["pct_change"],
            output_path=plots_dir / "trays_pct_change_by_magnetic_intensity.png",
            order=orden_intensidad,
            xlabel="intensidad magnetica",
        )
        guardar_boxplot_stripplot(
            solo_magneticos,
            x="magnetic_intensity_rank",
            y="log_ratio_centered",
            titulo="Log ratio centrado por intensidad magnetica",
            ylabel=METRIC_YLABELS["log_ratio_centered"],
            output_path=plots_dir / "trays_log_ratio_centered_by_magnetic_intensity.png",
            order=orden_intensidad,
            xlabel="intensidad magnetica",
        )
    guardar_boxplot_stripplot(
        data,
        x="floor",
        y="delta_weight",
        titulo="Delta de peso por floor",
        ylabel=METRIC_YLABELS["delta_weight"],
        output_path=plots_dir / "trays_delta_by_floor.png",
        order=orden_floor,
    )
    guardar_boxplot_stripplot(
        data,
        x="floor",
        y="pct_change",
        titulo="Cambio relativo por floor",
        ylabel=METRIC_YLABELS["pct_change"],
        output_path=plots_dir / "trays_pct_change_by_floor.png",
        order=orden_floor,
    )
    guardar_boxplot_stripplot(
        data,
        x="floor",
        y="log_ratio_centered",
        titulo="Log ratio centrado por floor",
        ylabel=METRIC_YLABELS["log_ratio_centered"],
        output_path=plots_dir / "trays_log_ratio_centered_by_floor.png",
        order=orden_floor,
    )
    guardar_boxplot_stripplot(
        data,
        x="position",
        y="delta_weight",
        titulo="Delta de peso por position",
        ylabel=METRIC_YLABELS["delta_weight"],
        output_path=plots_dir / "trays_delta_by_position.png",
        order=orden_position,
    )
    guardar_boxplot_stripplot(
        data,
        x="position",
        y="pct_change",
        titulo="Cambio relativo por position",
        ylabel=METRIC_YLABELS["pct_change"],
        output_path=plots_dir / "trays_pct_change_by_position.png",
        order=orden_position,
    )
    guardar_boxplot_stripplot(
        data,
        x="position",
        y="log_ratio_centered",
        titulo="Log ratio centrado por position",
        ylabel=METRIC_YLABELS["log_ratio_centered"],
        output_path=plots_dir / "trays_log_ratio_centered_by_position.png",
        order=orden_position,
    )
    guardar_heatmap(
        data,
        value_column="delta_weight",
        titulo="Patron espacial de delta_weight",
        output_path=plots_dir / "trays_delta_heatmap_floor_position.png",
    )
    guardar_heatmap(
        data,
        value_column="pct_change",
        titulo="Patron espacial de pct_change",
        output_path=plots_dir / "trays_pct_change_heatmap_floor_position.png",
    )
    guardar_heatmap(
        data,
        value_column="log_ratio_centered",
        titulo="Patron espacial de log_ratio_centered",
        output_path=plots_dir / "trays_log_ratio_centered_heatmap_floor_position.png",
    )
    if comparisons is not None and not comparisons.empty:
        guardar_forest_plot_tratamiento_vs_control(
            comparisons,
            response_metric="delta_weight",
            output_path=plots_dir / "trays_forest_delta_treatment_vs_control.png",
        )
        guardar_forest_plot_tratamiento_vs_control(
            comparisons,
            response_metric="pct_change",
            output_path=plots_dir / "trays_forest_pct_change_treatment_vs_control.png",
        )
        guardar_forest_plot_tratamiento_vs_control(
            comparisons,
            response_metric="log_ratio_centered",
            output_path=plots_dir / "trays_forest_log_ratio_centered_treatment_vs_control.png",
        )


def guardar_experimento(
    enriched: pd.DataFrame,
    summary: pd.DataFrame,
    models: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    exp_dir = Path(str(enriched["exp_dir"].iloc[0]))
    processed_dir = exp_dir / "processed"
    plots_dir = exp_dir / "plots"
    processed_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    enriched.to_csv(processed_dir / "trays_analysis_enriched.csv", index=False)
    summary.to_csv(processed_dir / "trays_analysis_summary.csv", index=False)
    models.to_csv(processed_dir / "trays_analysis_models.csv", index=False)
    comparisons.to_csv(
        processed_dir / "trays_treatment_vs_control.csv", index=False
    )
    guardar_plots_experimento(enriched, plots_dir, comparisons)


def guardar_global(
    enriched: pd.DataFrame,
    summary: pd.DataFrame,
    models: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    GLOBAL_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(GLOBAL_ANALYSIS_DIR / "trays_analysis_enriched_all.csv", index=False)
    summary.to_csv(GLOBAL_ANALYSIS_DIR / "trays_analysis_summary_all.csv", index=False)
    models.to_csv(GLOBAL_ANALYSIS_DIR / "trays_analysis_models_all.csv", index=False)
    comparisons.to_csv(
        GLOBAL_ANALYSIS_DIR / "trays_treatment_vs_control_all.csv", index=False
    )


def _valor_p_term(
    models: pd.DataFrame,
    term: str,
    response_metric: str,
    model_name: str | None = None,
) -> float:
    subset = models[
        (models["analysis_type"] == "term_test")
        & (models["response_metric"] == response_metric)
        & (models["term"] == term)
    ]
    if model_name is not None:
        subset = subset[subset["model_name"] == model_name]
    if subset.empty:
        return np.nan
    return float(subset["p_value"].iloc[0])


def _coef_termino(
    models: pd.DataFrame,
    *,
    response_metric: str,
    model_name: str,
    term: str,
) -> tuple[float, float]:
    subset = models[
        (models["analysis_type"] == "coefficient")
        & (models["response_metric"] == response_metric)
        & (models["model_name"] == model_name)
        & (models["term"] == term)
    ]
    if subset.empty:
        return np.nan, np.nan
    fila = subset.iloc[0]
    return float(fila["estimate"]), float(fila["p_value"])


def construir_resumen_experimento(
    enriched: pd.DataFrame,
    models: pd.DataFrame,
) -> dict[str, Any]:
    exp_dir = Path(str(enriched["exp_dir"].iloc[0]))
    coef_delta, p_delta = _coef_termino(
        models,
        response_metric="delta_weight",
        model_name="delta_magnetic_intensity",
        term="magnetic_intensity_rank",
    )
    coef_pct, p_pct = _coef_termino(
        models,
        response_metric="pct_change",
        model_name="pct_magnetic_intensity",
        term="magnetic_intensity_rank",
    )
    coef_log, p_log = _coef_termino(
        models,
        response_metric="log_ratio_centered",
        model_name="log_ratio_centered_magnetic_intensity",
        term="magnetic_intensity_rank",
    )
    _, p_group_delta = _coef_termino(
        models,
        response_metric="delta_weight",
        model_name="delta_magnetic_vs_control",
        term="C(treatment_role)[T.magnetic]",
    )
    _, p_group_pct = _coef_termino(
        models,
        response_metric="pct_change",
        model_name="pct_magnetic_vs_control",
        term="C(treatment_role)[T.magnetic]",
    )
    _, p_group_log = _coef_termino(
        models,
        response_metric="log_ratio_centered",
        model_name="log_ratio_centered_magnetic_vs_control",
        term="C(treatment_role)[T.magnetic]",
    )
    p_floor_delta = _valor_p_term(
        models,
        "C(floor)",
        "delta_weight",
        model_name="delta_magnetic_intensity",
    )
    p_position_delta = _valor_p_term(
        models,
        "C(position_factor)",
        "delta_weight",
        model_name="delta_magnetic_intensity",
    )

    total = len(enriched)
    valid_delta = int(enriched["valid_delta"].sum())
    valid_pct = int(enriched["valid_pct_change"].sum())
    valid_log = int(enriched["valid_log_ratio"].sum())
    magnetic_rows = int(enriched["treatment_role"].eq("magnetic").sum())
    control_rows = int(enriched["treatment_role"].eq("control").sum())
    has_active_magnet = bool(enriched["has_active_magnet"].any())
    control_delta_mean = pd.to_numeric(
        enriched["control_delta_mean"], errors="coerce"
    ).dropna()
    control_log_ratio_mean = pd.to_numeric(
        enriched["control_log_ratio_mean"], errors="coerce"
    ).dropna()

    return {
        "exp_id": str(enriched["exp_id"].iloc[0]),
        "total_rows": total,
        "valid_delta_rows": valid_delta,
        "excluded_delta_rows": total - valid_delta,
        "valid_pct_rows": valid_pct,
        "excluded_pct_rows": total - valid_pct,
        "valid_log_rows": valid_log,
        "excluded_log_rows": total - valid_log,
        "has_active_magnet": has_active_magnet,
        "magnetic_rows": magnetic_rows,
        "control_rows": control_rows,
        "magnetic_intensity_delta_coef": coef_delta,
        "magnetic_intensity_delta_p_value": p_delta,
        "magnetic_intensity_pct_coef": coef_pct,
        "magnetic_intensity_pct_p_value": p_pct,
        "magnetic_intensity_log_coef": coef_log,
        "magnetic_intensity_log_p_value": p_log,
        "magnetic_vs_control_delta_p_value": p_group_delta,
        "magnetic_vs_control_pct_p_value": p_group_pct,
        "magnetic_vs_control_log_p_value": p_group_log,
        "treatment_delta_coef": coef_delta,
        "treatment_delta_p_value": p_delta,
        "treatment_pct_coef": coef_pct,
        "treatment_pct_p_value": p_pct,
        "treatment_log_coef": coef_log,
        "treatment_log_p_value": p_log,
        "control_delta_mean": (
            float(control_delta_mean.iloc[0]) if not control_delta_mean.empty else np.nan
        ),
        "control_delta_count": int(pd.to_numeric(enriched["control_delta_count"], errors="coerce").max())
        if "control_delta_count" in enriched
        else 0,
        "control_log_ratio_mean": (
            float(control_log_ratio_mean.iloc[0])
            if not control_log_ratio_mean.empty
            else np.nan
        ),
        "control_log_ratio_count": int(
            pd.to_numeric(enriched["control_log_ratio_count"], errors="coerce").max()
        )
        if "control_log_ratio_count" in enriched
        else 0,
        "floor_delta_p_value": p_floor_delta,
        "position_delta_p_value": p_position_delta,
        "has_complete_metadata": bool(enriched["has_metadata"].all()),
        "exp_dir": str(exp_dir),
        "processed_dir": str(exp_dir / "processed"),
        "plots_dir": str(exp_dir / "plots"),
    }


def imprimir_resumen(
    resumenes_experimentos: list[dict[str, Any]],
    advertencias: list[str],
) -> None:
    print(f"Experimentos analizados: {len(resumenes_experimentos)}")
    print()

    for resumen in resumenes_experimentos:
        print(f"Experimento {resumen['exp_id']}")
        print(f"  Filas totales: {resumen['total_rows']}")
        print(
            f"  Tratamientos magnetic/control: {resumen['magnetic_rows']} / {resumen['control_rows']}"
        )
        print(
            "  Delta valido: "
            f"{resumen['valid_delta_rows']} | excluidas: {resumen['excluded_delta_rows']}"
        )
        print(
            "  Pct_change valido: "
            f"{resumen['valid_pct_rows']} | excluidas: {resumen['excluded_pct_rows']}"
        )
        print(
            "  Log ratio valido: "
            f"{resumen['valid_log_rows']} | excluidas: {resumen['excluded_log_rows']}"
        )
        print(
            "  Normalizacion por control: "
            f"mean(delta_control)={resumen['control_delta_mean']:.6f} "
            f"(n={resumen['control_delta_count']})"
            if not np.isnan(resumen["control_delta_mean"])
            else "  Normalizacion por control: no disponible"
        )
        print(
            "  Centrado log por control: "
            f"mean(log_control)={resumen['control_log_ratio_mean']:.6f} "
            f"(n={resumen['control_log_ratio_count']})"
            if not np.isnan(resumen["control_log_ratio_mean"])
            else "  Centrado log por control: no disponible"
        )
        print(
            "  Intensidad magnetica en delta_weight: "
            f"coef={resumen['magnetic_intensity_delta_coef']:.6f} | "
            f"p-value={resumen['magnetic_intensity_delta_p_value']:.6g}"
            if not np.isnan(resumen["magnetic_intensity_delta_coef"])
            else "  Intensidad magnetica en delta_weight: no disponible"
        )
        print(
            "  Intensidad magnetica en pct_change: "
            f"coef={resumen['magnetic_intensity_pct_coef']:.6f} | "
            f"p-value={resumen['magnetic_intensity_pct_p_value']:.6g}"
            if not np.isnan(resumen["magnetic_intensity_pct_coef"])
            else "  Intensidad magnetica en pct_change: no disponible"
        )
        print(
            "  Intensidad magnetica en log_ratio_centered: "
            f"coef={resumen['magnetic_intensity_log_coef']:.6f} | "
            f"p-value={resumen['magnetic_intensity_log_p_value']:.6g}"
            if not np.isnan(resumen["magnetic_intensity_log_coef"])
            else "  Intensidad magnetica en log_ratio_centered: no disponible"
        )
        print(
            "  Magnetic vs control: "
            f"delta p={resumen['magnetic_vs_control_delta_p_value']:.6g}, "
            f"pct_change p={resumen['magnetic_vs_control_pct_p_value']:.6g}, "
            f"log_ratio_centered p={resumen['magnetic_vs_control_log_p_value']:.6g}"
            if not np.isnan(resumen["magnetic_vs_control_delta_p_value"])
            or not np.isnan(resumen["magnetic_vs_control_pct_p_value"])
            or not np.isnan(resumen["magnetic_vs_control_log_p_value"])
            else "  Magnetic vs control: no disponible"
        )
        print(
            "  Efecto espacial (delta_weight): "
            f"floor p={resumen['floor_delta_p_value']:.6g}, position p={resumen['position_delta_p_value']:.6g}"
            if not np.isnan(resumen["floor_delta_p_value"])
            or not np.isnan(resumen["position_delta_p_value"])
            else "  Efecto espacial (delta_weight): no disponible"
        )
        print()

    if advertencias:
        print("Advertencias:")
        for advertencia in sorted(set(advertencias)):
            print(f"  - {advertencia}")


def run_analysis(exp_id: str | None = None, verbose: bool = True) -> dict[str, Any]:
    if sm is None or smf is None:
        raise RuntimeError(
            "No se encontro 'statsmodels'. Instala dependencias con "
            "'pip install -r requirements.txt' antes de ejecutar este analisis."
        )

    metadata, advertencias = leer_metadata_global()
    trays_paths = descubrir_trays(exp_id)

    experimentos: list[pd.DataFrame] = []
    modelos: list[pd.DataFrame] = []
    comparaciones: list[pd.DataFrame] = []
    resumenes_experimentos: list[dict[str, Any]] = []

    for trays_path in trays_paths:
        enriched, warnings_exp = cargar_experimento(trays_path, metadata)
        summary = resumen_experimento(enriched)
        models = modelos_experimento(enriched)
        comparisons = comparaciones_tratamiento_vs_control_experimento(enriched)

        guardar_experimento(enriched, summary, models, comparisons)

        experimentos.append(enriched)
        modelos.append(models)
        comparaciones.append(comparisons)
        resumenes_experimentos.append(construir_resumen_experimento(enriched, models))
        advertencias.extend(warnings_exp)

    enriched_all = pd.concat(experimentos, ignore_index=True)
    summary_all = resumen_global(enriched_all)
    models_all = pd.concat(modelos, ignore_index=True)
    comparisons_all = pd.concat(comparaciones, ignore_index=True)
    guardar_global(enriched_all, summary_all, models_all, comparisons_all)

    advertencias_unicas = sorted(set(advertencias))
    if verbose:
        imprimir_resumen(resumenes_experimentos, advertencias_unicas)

    return {
        "experiment_data": experimentos,
        "experiment_models": modelos,
        "experiment_treatment_vs_control": comparaciones,
        "experiment_summaries": resumenes_experimentos,
        "global_enriched": enriched_all,
        "global_summary": summary_all,
        "global_models": models_all,
        "global_treatment_vs_control": comparisons_all,
        "warnings": advertencias_unicas,
        "global_analysis_dir": str(GLOBAL_ANALYSIS_DIR),
    }


def analizar(args: argparse.Namespace) -> int:
    run_analysis(args.exp_id, verbose=True)
    return 0


def main() -> int:
    parser = construir_parser()
    args = parser.parse_args()

    try:
        return analizar(args)
    except RuntimeError as exc:
        print(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
