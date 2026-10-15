from __future__ import annotations

import argparse
import os
import re
import tempfile
import unicodedata
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
METADATA_COLUMNS = ["exp_id", "experiment_type", "magnet_pole", "water_ml"]
# water_ml queda fuera de las obligatorias: no se exige para no romper la lectura
# de metadata viejo, pero se arrastra si esta presente.
REQUIRED_METADATA_COLUMNS = {"exp_id", "experiment_type", "magnet_pole"}
SUMMARY_METRICS = (
    "pre_weight",
    "post_weight",
    "delta_weight",
    "pct_change",
    "log_ratio_centered",
)
FLOOR_ORDER = ["A", "B", "C"]
POSITION_LEFT_CENTER_RIGHT_ORDER = ["izquierda", "centro", "derecha"]
POSITION_FRONT_MIDDLE_BACK_ORDER = ["frente", "medio", "atras"]
TRAY_CELL_ORDER = [f"{floor}{position}" for floor in FLOOR_ORDER for position in range(1, 10)]
MAGNET_NONE_VALUE = "NONE"
NEGATIVE_CONTROL_EXPERIMENT_TYPES = {"CTRL_NEGATIVO", MAGNET_NONE_VALUE}
NON_MAGNETIC_EXPERIMENT_TYPES = {
    "CTRL_NEGATIVO",
    "CTRL_POSITIVO",
    MAGNET_NONE_VALUE,
}
MAGNETIC_TREATMENT_FLOORS = tuple(range(1, 9))
CONTROL_TREATMENT_FLOOR = 9
METRIC_YLABELS = {
    "delta_weight": "(post_weight - pre_weight) / mean(control_delta)",
    "pct_change": "(post_weight - pre_weight) / pre_weight",
    "log_ratio_centered": "log(post/pre) centrado por control",
}
METRIC_FOREST_TITLES = {
    "delta_weight": "Tratamiento vs Control (delta_weight normalizado)",
    "pct_change": "Tratamiento vs Control (pct_change)",
    "log_ratio_centered": "Tratamiento vs Control (log_ratio centrado por control)",
}
TREATMENT_VS_CONTROL_COLUMNS = [
    "analysis_scope",
    "source_exp_id",
    "response_metric",
    "treatment_label",
    "control_label",
    "analysis_type",
    "method",
    "estimate_diff_vs_control",
    "std_error",
    "statistic",
    "statistic_name",
    "p_value",
    "p_value_adjusted",
    "reject_null_fdr_bh",
    "conf_int_low",
    "conf_int_high",
    "n_treatment",
    "n_control",
    "mean_treatment",
    "mean_control",
    "notes",
]

plt: Any | None = None
sns: Any | None = None


def cargar_dependencias_plotting() -> bool:
    global plt, sns

    if plt is not None and sns is not None:
        return True

    try:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            str(Path(tempfile.gettempdir()) / "magseed-matplotlib-cache"),
        )

        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as matplotlib_pyplot
        import seaborn as seaborn_module
    except ImportError as exc:
        warnings.warn(
            "No se pudieron cargar matplotlib/seaborn; se omitira la generacion "
            f"de plots. Detalle: {exc}",
            RuntimeWarning,
        )
        return False

    plt = matplotlib_pyplot
    sns = seaborn_module
    return True


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


def normalizar_experiment_type(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value).strip().upper())
    value = value.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"_+", "_", re.sub(r"\s+", "_", value))


def descubrir_trays(
    exp_id: str | None = None,
    exp_ids: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    if exp_id is not None and exp_ids is not None:
        raise ValueError("Usa exp_id o exp_ids, no ambos.")

    if exp_ids is not None:
        rutas: list[Path] = []
        for valor in exp_ids:
            exp_id_validado = validar_exp_id(valor)
            ruta = EXPERIMENTS_DIR / exp_id_validado / TRAYS_FILENAME
            if not ruta.exists():
                raise FileNotFoundError(f"No existe '{ruta}'.")
            rutas.append(ruta)
        return rutas

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
    if "experiment_type" not in metadata.columns and "magnet" in metadata.columns:
        metadata["experiment_type"] = metadata["magnet"]

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
    metadata["experiment_type"] = (
        metadata["experiment_type"]
        .astype(str)
        .map(normalizar_experiment_type)
    )
    metadata["magnet"] = metadata["experiment_type"]
    metadata["magnet_pole"] = (
        metadata["magnet_pole"].astype(str).str.strip().str.upper()
    )

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
    return pd.to_numeric(
        serie.astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )


def construir_etiqueta_tratamiento(fila: pd.Series) -> str:
    treatment_floor = fila["treatment_floor"]
    if pd.isna(treatment_floor):
        return ""

    if (
        fila["has_metadata"]
        and fila["experiment_type"] in NEGATIVE_CONTROL_EXPERIMENT_TYPES
    ):
        return "Control negativo"

    if fila["has_active_magnet"] and treatment_floor in MAGNETIC_TREATMENT_FLOORS:
        return f"Mag {int(treatment_floor)}"

    if fila["has_active_magnet"] and treatment_floor == CONTROL_TREATMENT_FLOOR:
        return "Control"

    return f"Floor {int(treatment_floor)}"


def enriquecer_tratamientos(enriched: pd.DataFrame) -> pd.DataFrame:
    enriched["has_active_magnet"] = enriched["has_metadata"] & ~enriched[
        "experiment_type"
    ].isin(NON_MAGNETIC_EXPERIMENT_TYPES)

    es_control_negativo = enriched["has_metadata"] & enriched["experiment_type"].isin(
        NEGATIVE_CONTROL_EXPERIMENT_TYPES
    )
    es_control_positivo = enriched["has_metadata"] & enriched["experiment_type"].eq(
        "CTRL_POSITIVO"
    )
    es_tratamiento_magnetico = enriched["has_active_magnet"] & enriched[
        "treatment_floor"
    ].isin(MAGNETIC_TREATMENT_FLOORS)
    es_control_interno = enriched["has_active_magnet"] & enriched["treatment_floor"].eq(
        CONTROL_TREATMENT_FLOOR
    )

    treatment_role = pd.Series(pd.NA, index=enriched.index, dtype="string")
    treatment_role.loc[es_tratamiento_magnetico] = "magnetic"
    treatment_role.loc[
        es_control_interno | es_control_negativo | es_control_positivo
    ] = "control"
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
    magnetic_intensity_rank.loc[es_tratamiento_magnetico] = enriched.loc[
        es_tratamiento_magnetico, "treatment_floor"
    ].astype("Int64")
    enriched["magnetic_intensity_rank"] = magnetic_intensity_rank
    enriched["treatment_label"] = enriched.apply(
        construir_etiqueta_tratamiento, axis=1
    ).astype("string")
    return enriched


def es_experimento_control_negativo(data: pd.DataFrame) -> bool:
    return bool(data["has_metadata"].all()) and bool(
        data["experiment_type"].isin(NEGATIVE_CONTROL_EXPERIMENT_TYPES).all()
    )


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
    es_control_negativo = es_experimento_control_negativo(enriched)
    if es_control_negativo:
        mask_control = enriched["has_complete_weights"]
        expected_count = int(mask_control.sum())
        control_desc = "todas las bandejas del control negativo"
    else:
        mask_control = (
            enriched["treatment_floor"].eq(CONTROL_TREATMENT_FLOOR)
            & enriched["has_complete_weights"]
        )
        expected_count = 3
        control_desc = f"treatment_floor={CONTROL_TREATMENT_FLOOR}"
    deltas_control = (
        enriched.loc[mask_control, "post_weight"]
        - enriched.loc[mask_control, "pre_weight"]
    ).dropna()
    cantidad_controles = int(len(deltas_control))

    if cantidad_controles == 0:
        advertencias.append(
            f"El experimento {exp_name} no tiene controles validos "
            f"({control_desc}) para normalizar pesos."
        )
        return np.nan, cantidad_controles, advertencias

    if cantidad_controles != expected_count:
        advertencias.append(
            f"El experimento {exp_name} esperaba {expected_count} deltas de control para normalizar, "
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
    es_control_negativo = es_experimento_control_negativo(enriched)
    if es_control_negativo:
        mask_control = (
            enriched["has_complete_weights"]
            & (enriched["raw_pre_weight"] > 0)
            & (enriched["raw_post_weight"] > 0)
        )
        expected_count = int(mask_control.sum())
        control_desc = "todas las bandejas del control negativo"
    else:
        mask_control = (
            enriched["treatment_floor"].eq(CONTROL_TREATMENT_FLOOR)
            & enriched["has_complete_weights"]
            & (enriched["raw_pre_weight"] > 0)
            & (enriched["raw_post_weight"] > 0)
        )
        expected_count = 3
        control_desc = f"treatment_floor={CONTROL_TREATMENT_FLOOR}"
    log_ratios_control = np.log(
        enriched.loc[mask_control, "raw_post_weight"]
        / enriched.loc[mask_control, "raw_pre_weight"]
    )
    log_ratios_control = log_ratios_control.replace([np.inf, -np.inf], np.nan).dropna()
    cantidad_controles = int(len(log_ratios_control))

    if cantidad_controles == 0:
        advertencias.append(
            f"El experimento {exp_name} no tiene controles validos "
            f"({control_desc}) para centrar log(post/pre)."
        )
        return np.nan, cantidad_controles, advertencias

    if cantidad_controles != expected_count:
        advertencias.append(
            f"El experimento {exp_name} esperaba {expected_count} controles para centrar log(post/pre), "
            f"pero encontro {cantidad_controles}. Se uso el promedio disponible."
        )

    return float(log_ratios_control.mean()), cantidad_controles, advertencias


def cargar_dataset_completo(exp_ids: list[str] = None) -> pd.DataFrame:
    """
    Función utilitaria para Notebooks. 
    Carga uno o varios experimentos y retorna un solo DataFrame limpio y enriquecido.
    """
    metadata, _ = leer_metadata_global()
    paths = descubrir_trays(exp_ids=exp_ids)
    
    all_data = []
    for p in paths:
        enriched, _ = cargar_experimento(p, metadata)
        all_data.append(enriched)
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def cargar_experimento(
    trays_path: Path, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
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
    enriched["experiment_type"] = (
        enriched["experiment_type"]
        .fillna("")
        .astype(str)
        .map(normalizar_experiment_type)
    )
    enriched["magnet"] = enriched["experiment_type"]
    enriched["magnet_pole"] = (
        enriched["magnet_pole"].fillna("").astype(str).str.strip().str.upper()
    )

    enriched["has_metadata"] = enriched["experiment_type"].ne("") & enriched[
        "magnet_pole"
    ].ne("")
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
    enriched["valid_pct_change"] = enriched["valid_delta"] & (
        enriched["pre_weight"] > 0
    )
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
        enriched["valid_log_ratio"],
        enriched["raw_log_ratio"],
        np.nan,
    )

    # Si es un experimento de tratamiento y hay una media de control, se centra.
    # Si es un control negativo, también se centra con su propia media.
    # El ajuste por la media global de controles negativos se hará después.
    if np.isfinite(control_log_ratio_mean):
        enriched["log_ratio_centered"] -= control_log_ratio_mean

    # El ajuste final se hace en la función `run_analysis` o `run_negative_control_analysis`
    # que tiene el contexto de todos los experimentos.
    # Aquí solo preparamos los datos base.

    # Actualizamos la etiqueta del eje Y para ser más genérica, ya que el centrado
    # final puede ser complejo.
    if "log_ratio_centered" in METRIC_YLABELS:
        METRIC_YLABELS["log_ratio_centered"] = (
            "log(post/pre) centrado por control"
            if es_experimento_control_negativo(enriched)
            else "log(post/pre) ajustado por control interno y control negativo global"
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
    enriched["floor"] = pd.Categorical(
        enriched["floor"], categories=FLOOR_ORDER, ordered=True
    )
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
                "experiment_type": pd.NA,
                "magnet": pd.NA,
                "magnet_pole": pd.NA,
                "treatment_floor": pd.NA,
                "floor": pd.NA,
                "position": pd.NA,
                "n_total": int(fila["n_total"]),
                "n_valid": int(fila["n_valid"]) if not pd.isna(fila["n_valid"]) else 0,
                "n_missing": int(
                    fila["n_total"]
                    - (fila["n_valid"] if not pd.isna(fila["n_valid"]) else 0)
                ),
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
            group_by="experiment_type",
            group_columns=["experiment_type"],
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
            group_by="experiment_type_and_pole",
            group_columns=["experiment_type", "magnet_pole"],
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
                    "El experimento no tiene tratamiento magnetico activo, "
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

        if (
            subset["magnetic_intensity_rank"].nunique() < 2
            or subset[metrica].nunique() < 2
        ):
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
    es_control_negativo = es_experimento_control_negativo(data)

    intensity_data = data.dropna(subset=["magnetic_intensity_rank"]).copy()
    intensity_formula_delta = "delta_weight ~ pre_weight + magnetic_intensity_rank + C(floor) + C(position_factor)"
    intensity_formula_pct = (
        "pct_change ~ magnetic_intensity_rank + C(floor) + C(position_factor)"
    )
    intensity_formula_log = (
        "log_ratio_centered ~ magnetic_intensity_rank + C(floor) + C(position_factor)"
    )

    if (
        bool(data["has_active_magnet"].any())
        and intensity_data["magnetic_intensity_rank"].nunique() >= 2
    ):
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
            "El experimento no tiene tratamiento magnetico activo, asi que no hay "
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

    spatial_formula_delta = (
        "delta_weight ~ pre_weight + C(treatment_label) + C(floor) + C(position_factor)"
    )
    spatial_formula_pct = (
        "pct_change ~ C(treatment_label) + C(floor) + C(position_factor)"
    )
    spatial_formula_log = (
        "log_ratio_centered ~ C(treatment_label) + C(floor) + C(position_factor)"
    )
    spatial_data = data.dropna(subset=["treatment_label"]).copy()
    modelos.extend(
        [
            ajustar_modelo(
                spatial_data,
                exp_id=exp_id,
                response_metric="delta_weight",
                model_name="delta_spatial_effect",
                formula=spatial_formula_delta,
            ),
            ajustar_modelo(
                spatial_data,
                exp_id=exp_id,
                response_metric="pct_change",
                model_name="pct_spatial_effect",
                formula=spatial_formula_pct,
            ),
            ajustar_modelo(
                spatial_data,
                exp_id=exp_id,
                response_metric="log_ratio_centered",
                model_name="log_ratio_centered_spatial_effect",
                formula=spatial_formula_log,
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
    elif not es_control_negativo:
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


def comparaciones_tratamiento_vs_control_experimento(
    data: pd.DataFrame,
) -> pd.DataFrame:
    exp_id = str(data["exp_id"].iloc[0])
    if es_experimento_control_negativo(data):
        return pd.DataFrame(columns=TREATMENT_VS_CONTROL_COLUMNS)

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
            ],
            columns=TREATMENT_VS_CONTROL_COLUMNS,
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

    return pd.DataFrame(filas, columns=TREATMENT_VS_CONTROL_COLUMNS)


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
    pivot = pivot.reindex(
        sorted([col for col in pivot.columns if pd.notna(col)]), axis=1
    )

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
    if not cargar_dependencias_plotting():
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    es_control_negativo = es_experimento_control_negativo(data)

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
            etiqueta
            for etiqueta in etiquetas_presentes
            if etiqueta not in orden_treatment
        )
    )
    orden_intensidad = sorted(
        data["magnetic_intensity_rank"].dropna().unique().tolist()
    )
    orden_position = sorted(data["position"].dropna().unique().tolist())
    orden_floor = [
        floor for floor in FLOOR_ORDER if floor in data["floor"].astype(str).unique()
    ]

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
    if len(orden_role) >= 2 and not es_control_negativo:
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
            output_path=plots_dir
            / "trays_log_ratio_centered_by_magnetic_intensity.png",
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
            output_path=plots_dir
            / "trays_forest_log_ratio_centered_treatment_vs_control.png",
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
    comparisons.to_csv(processed_dir / "trays_treatment_vs_control.csv", index=False)
    guardar_plots_experimento(enriched, plots_dir, comparisons)


def guardar_global(
    enriched: pd.DataFrame,
    summary: pd.DataFrame,
    models: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    GLOBAL_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(
        GLOBAL_ANALYSIS_DIR / "trays_analysis_enriched_all.csv", index=False
    )
    summary.to_csv(GLOBAL_ANALYSIS_DIR / "trays_analysis_summary_all.csv", index=False)
    models.to_csv(GLOBAL_ANALYSIS_DIR / "trays_analysis_models_all.csv", index=False)
    comparisons.to_csv(
        GLOBAL_ANALYSIS_DIR / "trays_treatment_vs_control_all.csv", index=False
    )


def _ajustar_modelo_control_negativo_combinado(
    data: pd.DataFrame,
    *,
    response_metric: str,
) -> pd.DataFrame:
    filas: list[dict[str, Any]] = []
    model_data = data.copy()
    model_data[response_metric] = pd.to_numeric(
        model_data[response_metric], errors="coerce"
    )
    model_data = model_data.dropna(subset=[response_metric]).copy()
    model_data["exp_id"] = model_data["exp_id"].astype(str)
    model_data["floor"] = model_data["floor"].astype(str)
    model_data["position_factor"] = model_data["position"].astype("string")
    formula = f"{response_metric} ~ C(exp_id) + C(floor) + C(position_factor)"

    base_row = {
        "analysis_scope": "negative_control_combined",
        "source_exp_id": ",".join(sorted(data["exp_id"].astype(str).unique())),
        "response_metric": response_metric,
        "analysis_type": "term_test",
        "model_name": "negative_control_day_spatial_effect",
        "formula": formula,
        "method": "anova_typ2",
        "term": pd.NA,
        "estimate": np.nan,
        "std_error": np.nan,
        "statistic": np.nan,
        "statistic_name": "F",
        "p_value": np.nan,
        "conf_int_low": np.nan,
        "conf_int_high": np.nan,
        "sum_sq": np.nan,
        "df_term": np.nan,
        "n_obs": len(model_data),
        "rsquared": np.nan,
        "adj_rsquared": np.nan,
        "notes": pd.NA,
    }

    if len(model_data) < 4:
        fila = base_row.copy()
        fila["analysis_type"] = "model_not_run"
        fila["method"] = "insufficient_data"
        fila["notes"] = "No alcanza para ajustar el modelo combinado."
        return pd.DataFrame([fila])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resultado = smf.ols(formula=formula, data=model_data).fit()
            anova = sm.stats.anova_lm(resultado, typ=2).reset_index()
    except Exception as exc:
        fila = base_row.copy()
        fila["analysis_type"] = "model_error"
        fila["method"] = "ols/anova_typ2"
        fila["notes"] = str(exc)
        return pd.DataFrame([fila])

    anova.rename(columns={"index": "term", "PR(>F)": "p_value"}, inplace=True)
    anova = anova[anova["term"] != "Residual"]
    for _, fila_anova in anova.iterrows():
        fila = base_row.copy()
        fila.update(
            {
                "term": fila_anova["term"],
                "statistic": fila_anova.get("F", np.nan),
                "p_value": fila_anova.get("p_value", np.nan),
                "sum_sq": fila_anova.get("sum_sq", np.nan),
                "df_term": fila_anova.get("df", np.nan),
                "n_obs": int(resultado.nobs),
                "rsquared": resultado.rsquared,
                "adj_rsquared": resultado.rsquared_adj,
            }
        )
        filas.append(fila)

    return pd.DataFrame(filas)


def _one_sample_tests_control_negativo(data: pd.DataFrame) -> pd.DataFrame:
    filas: list[dict[str, Any]] = []
    source_exp_id = ",".join(sorted(data["exp_id"].astype(str).unique()))
    tests = {
        "raw_delta_weight": 0.0,
        "pct_change": 0.0,
        "raw_log_ratio": 0.0,
        "log_ratio_centered": 0.0,
    }

    for metrica, valor_esperado in tests.items():
        serie = pd.to_numeric(data[metrica], errors="coerce").dropna()
        statistic = np.nan
        p_value = np.nan
        notes = pd.NA
        if len(serie) < 2:
            notes = "No alcanza para t-test de una muestra."
        elif serie.nunique() < 2:
            notes = "No se pudo calcular por falta de variacion."
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resultado = stats.ttest_1samp(serie, popmean=valor_esperado)
            statistic, p_value = _extraer_stat_pvalue(resultado)

        filas.append(
            {
                "analysis_scope": "negative_control_combined",
                "source_exp_id": source_exp_id,
                "response_metric": metrica,
                "analysis_type": "one_sample_ttest",
                "model_name": "negative_control_mean_vs_expected",
                "formula": pd.NA,
                "method": "ttest_1samp",
                "term": f"mean = {valor_esperado:g}",
                "estimate": float(serie.mean()) if not serie.empty else np.nan,
                "std_error": (
                    float(serie.std(ddof=1) / np.sqrt(len(serie)))
                    if len(serie) > 1
                    else np.nan
                ),
                "statistic": statistic,
                "statistic_name": "t",
                "p_value": p_value,
                "conf_int_low": np.nan,
                "conf_int_high": np.nan,
                "sum_sq": np.nan,
                "df_term": len(serie) - 1 if len(serie) > 0 else np.nan,
                "n_obs": len(serie),
                "rsquared": np.nan,
                "adj_rsquared": np.nan,
                "notes": notes,
            }
        )

    return pd.DataFrame(filas)


def diagnosticos_supuestos_control_negativo(data: pd.DataFrame) -> pd.DataFrame:
    filas: list[dict[str, Any]] = []
    source_exp_id = ",".join(sorted(data["exp_id"].astype(str).unique()))
    variables = ("raw_delta_weight", "pct_change", "raw_log_ratio")

    for metrica in variables:
        model_data = data[["exp_id", "floor", "position", metrica]].copy()
        model_data[metrica] = pd.to_numeric(model_data[metrica], errors="coerce")
        model_data = model_data.dropna(subset=["exp_id", "floor", "position", metrica])
        model_data["exp_id"] = model_data["exp_id"].astype(str)
        model_data["floor"] = model_data["floor"].astype(str)
        model_data["position_factor"] = model_data["position"].astype("string")
        formula = f"{metrica} ~ C(exp_id) + C(floor) + C(position_factor)"

        shapiro_w = np.nan
        shapiro_p = np.nan
        brown_f = np.nan
        brown_p = np.nan
        residual_df = np.nan
        notes = pd.NA

        if len(model_data) < 4:
            notes = "No alcanza para ajustar el modelo de diagnostico."
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    resultado = smf.ols(formula=formula, data=model_data).fit()
                    shapiro = stats.shapiro(resultado.resid)
                shapiro_w, shapiro_p = _extraer_stat_pvalue(shapiro)
                residual_df = int(resultado.df_resid)
            except Exception as exc:
                notes = f"No se pudo calcular Shapiro sobre residuos: {exc}"

        grupos = [
            grupo[metrica].dropna().to_numpy(dtype=float)
            for _, grupo in model_data.groupby("exp_id", sort=True)
        ]
        grupos = [grupo for grupo in grupos if len(grupo) > 0]
        if len(grupos) >= 2 and sum(len(grupo) for grupo in grupos) >= 3:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    brown = stats.levene(*grupos, center="median")
                brown_f, brown_p = _extraer_stat_pvalue(brown)
            except Exception as exc:
                if pd.isna(notes):
                    notes = f"No se pudo calcular Brown-Forsythe: {exc}"
                else:
                    notes = f"{notes} | No se pudo calcular Brown-Forsythe: {exc}"
        elif pd.isna(notes):
            notes = "No alcanza para Brown-Forsythe entre dias."

        filas.append(
            {
                "analysis_scope": "negative_control_combined",
                "source_exp_id": source_exp_id,
                "response_metric": metrica,
                "model_formula": formula,
                "n_obs": len(model_data),
                "residual_df": residual_df,
                "shapiro_w": shapiro_w,
                "shapiro_p_value": shapiro_p,
                "brown_forsythe_statistic": brown_f,
                "brown_forsythe_p_value": brown_p,
                "normality_alpha_0_05_ok": (
                    bool(shapiro_p >= 0.05) if not pd.isna(shapiro_p) else pd.NA
                ),
                "homoscedasticity_alpha_0_05_ok": (
                    bool(brown_p >= 0.05) if not pd.isna(brown_p) else pd.NA
                ),
                "notes": notes,
            }
        )

    return pd.DataFrame(filas)


def agregar_agrupaciones_posicion(data: pd.DataFrame) -> pd.DataFrame:
    enriched = data.copy()
    position = pd.to_numeric(enriched["position"], errors="coerce")
    floor = enriched["floor"].astype(str)
    position_label = position.astype("Int64").astype(str)
    enriched["tray_cell"] = pd.Categorical(
        np.where(position.notna(), floor + position_label, pd.NA),
        categories=TRAY_CELL_ORDER,
        ordered=True,
    )

    enriched["position_left_center_right"] = pd.Categorical(
        np.select(
            [
                position.isin([1, 4, 7]),
                position.isin([2, 5, 8]),
                position.isin([3, 6, 9]),
            ],
            POSITION_LEFT_CENTER_RIGHT_ORDER,
            default=pd.NA,
        ),
        categories=POSITION_LEFT_CENTER_RIGHT_ORDER,
        ordered=True,
    )
    enriched["position_front_middle_back"] = pd.Categorical(
        np.select(
            [
                position.isin([7, 8, 9]),
                position.isin([4, 5, 6]),
                position.isin([1, 2, 3]),
            ],
            POSITION_FRONT_MIDDLE_BACK_ORDER,
            default=pd.NA,
        ),
        categories=POSITION_FRONT_MIDDLE_BACK_ORDER,
        ordered=True,
    )
    return enriched


def comparar_grupo_control_negativo(
    data: pd.DataFrame,
    *,
    group_column: str,
    group_order: list[str],
    comparison: str,
    notes_label: str,
) -> pd.DataFrame:
    filas: list[dict[str, Any]] = []
    source_exp_id = ",".join(sorted(data["exp_id"].astype(str).unique()))
    variables = ("raw_delta_weight", "pct_change", "raw_log_ratio")

    for metrica in variables:
        model_data = data[["exp_id", group_column, metrica]].copy()
        model_data[metrica] = pd.to_numeric(model_data[metrica], errors="coerce")
        model_data = model_data.dropna(subset=["exp_id", group_column, metrica])
        model_data["exp_id"] = model_data["exp_id"].astype(str)
        model_data[group_column] = pd.Categorical(
            model_data[group_column].astype(str),
            categories=group_order,
            ordered=True,
        )
        model_data = model_data.dropna(subset=[group_column])
        formula = f"{metrica} ~ C(exp_id) + C({group_column})"
        term = f"C({group_column})"

        base_row = {
            "analysis_scope": "negative_control_combined",
            "source_exp_id": source_exp_id,
            "response_metric": metrica,
            "comparison": comparison,
            "group_column": group_column,
            "formula": formula,
            "method": "ols_wald_hc3",
            "term": term,
            "statistic": np.nan,
            "statistic_name": "F",
            "p_value": np.nan,
            "df_num": np.nan,
            "df_denom": np.nan,
            "n_obs": len(model_data),
            "rsquared": np.nan,
            "adj_rsquared": np.nan,
            "notes": pd.NA,
        }

        if len(model_data) < 4 or model_data[group_column].nunique() < 2:
            fila = base_row.copy()
            fila["method"] = "skipped"
            fila["notes"] = f"No alcanza para comparar {notes_label}."
            filas.append(fila)
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resultado = smf.ols(formula=formula, data=model_data).fit(
                    cov_type="HC3",
                    use_t=True,
                )
                terms = resultado.wald_test_terms(skip_single=False)
                test_row = terms.table.loc[term]

            fila = base_row.copy()
            fila.update(
                {
                    "statistic": float(np.asarray(test_row["statistic"]).squeeze()),
                    "p_value": float(test_row["pvalue"]),
                    "df_num": float(np.asarray(test_row["df_constraint"]).squeeze()),
                    "df_denom": float(test_row.get("df_denom", resultado.df_resid)),
                    "n_obs": int(resultado.nobs),
                    "rsquared": float(resultado.rsquared),
                    "adj_rsquared": float(resultado.rsquared_adj),
                }
            )
        except Exception as exc:
            fila = base_row.copy()
            fila["method"] = "model_error"
            fila["notes"] = str(exc)

        filas.append(fila)

    return pd.DataFrame(filas)


def comparar_pisos_control_negativo(data: pd.DataFrame) -> pd.DataFrame:
    return comparar_grupo_control_negativo(
        data,
        group_column="floor",
        group_order=FLOOR_ORDER,
        comparison="floor_A_B_C_adjusted_by_day",
        notes_label="pisos",
    )


def comparar_agrupaciones_posicion_control_negativo(data: pd.DataFrame) -> pd.DataFrame:
    partes = [
        comparar_grupo_control_negativo(
            data,
            group_column="position_left_center_right",
            group_order=POSITION_LEFT_CENTER_RIGHT_ORDER,
            comparison="position_left_center_right_adjusted_by_day",
            notes_label="izquierda/centro/derecha",
        ),
        comparar_grupo_control_negativo(
            data,
            group_column="position_front_middle_back",
            group_order=POSITION_FRONT_MIDDLE_BACK_ORDER,
            comparison="position_front_middle_back_adjusted_by_day",
            notes_label="frente/medio/atras",
        ),
    ]
    return pd.concat(partes, ignore_index=True)


def comparar_celdas_control_negativo(data: pd.DataFrame) -> pd.DataFrame:
    return comparar_grupo_control_negativo(
        data,
        group_column="tray_cell",
        group_order=TRAY_CELL_ORDER,
        comparison="tray_cell_A1_to_C9_adjusted_by_day",
        notes_label="celdas A1-C9",
    )


def resumen_control_negativo_combinado(
    data: pd.DataFrame, models: pd.DataFrame
) -> dict[str, Any]:
    exp_ids = sorted(data["exp_id"].astype(str).unique().tolist())
    total = len(data)
    valid_delta = int(data["raw_delta_weight"].notna().sum())
    valid_pct = int(data["pct_change"].notna().sum())
    valid_log = int(data["raw_log_ratio"].notna().sum())

    def p_term(metrica: str, termino: str) -> float:
        subset = models[
            (models["analysis_type"] == "term_test")
            & (models["response_metric"] == metrica)
            & (models["term"] == termino)
        ]
        if subset.empty:
            return np.nan
        return float(subset["p_value"].iloc[0])

    def mean_test(metrica: str) -> tuple[float, float]:
        subset = models[
            (models["analysis_type"] == "one_sample_ttest")
            & (models["response_metric"] == metrica)
        ]
        if subset.empty:
            return np.nan, np.nan
        fila = subset.iloc[0]
        return float(fila["estimate"]), float(fila["p_value"])

    raw_delta_mean, raw_delta_p = mean_test("raw_delta_weight")
    pct_mean, pct_p = mean_test("pct_change")
    raw_log_mean, raw_log_p = mean_test("raw_log_ratio")

    return {
        "exp_ids": exp_ids,
        "total_rows": total,
        "valid_delta_rows": valid_delta,
        "valid_pct_rows": valid_pct,
        "valid_log_rows": valid_log,
        "raw_delta_mean": raw_delta_mean,
        "raw_delta_mean_p_value": raw_delta_p,
        "pct_change_mean": pct_mean,
        "pct_change_mean_p_value": pct_p,
        "raw_log_ratio_mean": raw_log_mean,
        "raw_log_ratio_mean_p_value": raw_log_p,
        "day_raw_delta_p_value": p_term("raw_delta_weight", "C(exp_id)"),
        "floor_raw_delta_p_value": p_term("raw_delta_weight", "C(floor)"),
        "position_raw_delta_p_value": p_term("raw_delta_weight", "C(position_factor)"),
        "day_pct_p_value": p_term("pct_change", "C(exp_id)"),
        "floor_pct_p_value": p_term("pct_change", "C(floor)"),
        "position_pct_p_value": p_term("pct_change", "C(position_factor)"),
        "day_log_p_value": p_term("raw_log_ratio", "C(exp_id)"),
        "floor_log_p_value": p_term("raw_log_ratio", "C(floor)"),
        "position_log_p_value": p_term("raw_log_ratio", "C(position_factor)"),
        "output_dir": str(GLOBAL_ANALYSIS_DIR / "negative_control_combined"),
    }


def _formatear_p_plot(valor: float) -> str:
    if pd.isna(valor):
        return "n/d"
    if valor < 0.001:
        return "<0.001"
    return f"{valor:.3f}"


def calcular_tests_facetas_control_negativo(data: pd.DataFrame) -> pd.DataFrame:
    filas: list[dict[str, Any]] = []
    specs = [
        ("floor", FLOOR_ORDER),
        ("position", list(range(1, 10))),
    ]

    for group_column, group_order in specs:
        plot_data = data.dropna(subset=[group_column, "pct_change", "exp_id"]).copy()
        if plot_data.empty:
            continue

        plot_data[group_column] = plot_data[group_column].astype(str)
        plot_data["exp_id"] = plot_data["exp_id"].astype(str)
        valores_presentes = plot_data[group_column].unique().tolist()
        valores = [
            str(valor) for valor in group_order if str(valor) in valores_presentes
        ]
        valores.extend(
            sorted(valor for valor in valores_presentes if valor not in valores)
        )

        for group_value in valores:
            subset = plot_data[plot_data[group_column] == group_value].copy()
            groups = [
                pd.to_numeric(grupo["pct_change"], errors="coerce").dropna().to_numpy()
                for _, grupo in subset.groupby("exp_id", sort=True)
            ]
            groups = [grupo for grupo in groups if len(grupo) > 0]
            values = np.concatenate(groups) if groups else np.array([])
            statistic = np.nan
            p_value = np.nan
            notes = pd.NA

            if len(groups) < 2:
                notes = "No alcanza para comparar fechas."
            elif len(values) < 2:
                notes = "No alcanza para comparar fechas."
            elif len(np.unique(values)) < 2:
                statistic = 0.0
                p_value = 1.0
                notes = "Sin variacion entre valores."
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        resultado = stats.kruskal(*groups)
                    statistic, p_value = _extraer_stat_pvalue(resultado)
                except Exception as exc:
                    notes = str(exc)

            filas.append(
                {
                    "analysis_scope": "negative_control_combined",
                    "response_metric": "pct_change",
                    "group_column": group_column,
                    "group_value": group_value,
                    "method": "kruskal",
                    "statistic": statistic,
                    "p_value": p_value,
                    "q_value_fdr_bh": np.nan,
                    "reject_null_fdr_bh": pd.NA,
                    "n_dates": len(groups),
                    "n_obs": int(sum(len(grupo) for grupo in groups)),
                    "notes": notes,
                }
            )

    tests = pd.DataFrame(filas)
    if tests.empty:
        return tests

    valid_mask = tests["p_value"].notna()
    if valid_mask.any() and multipletests is not None:
        rechazo, q_values, _, _ = multipletests(
            tests.loc[valid_mask, "p_value"].to_numpy(dtype=float),
            method="fdr_bh",
        )
        tests.loc[valid_mask, "q_value_fdr_bh"] = q_values
        tests.loc[valid_mask, "reject_null_fdr_bh"] = rechazo
    elif valid_mask.any():
        tests.loc[valid_mask, "q_value_fdr_bh"] = tests.loc[valid_mask, "p_value"]
        tests.loc[valid_mask, "reject_null_fdr_bh"] = (
            tests.loc[valid_mask, "p_value"] < 0.05
        )

    return tests


def guardar_control_negativo_combinado(
    enriched: pd.DataFrame,
    summary: pd.DataFrame,
    models: pd.DataFrame,
    assumptions: pd.DataFrame,
    floor_tests: pd.DataFrame,
    position_group_tests: pd.DataFrame,
    tray_cell_tests: pd.DataFrame,
) -> tuple[Path, dict[str, Path]]:
    output_dir = GLOBAL_ANALYSIS_DIR / "negative_control_combined"
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_dir / "negative_control_combined_enriched.csv", index=False)
    summary.to_csv(output_dir / "negative_control_combined_summary.csv", index=False)
    models.to_csv(output_dir / "negative_control_combined_models.csv", index=False)
    assumptions.to_csv(
        output_dir / "negative_control_combined_assumption_tests.csv",
        index=False,
    )
    floor_tests.to_csv(
        output_dir / "negative_control_combined_floor_tests.csv",
        index=False,
    )
    position_group_tests.to_csv(
        output_dir / "negative_control_combined_position_group_tests.csv",
        index=False,
    )
    tray_cell_tests.to_csv(
        output_dir / "negative_control_combined_tray_cell_tests.csv",
        index=False,
    )
    facet_tests = calcular_tests_facetas_control_negativo(enriched)
    facet_tests.to_csv(
        output_dir / "negative_control_combined_facet_tests.csv",
        index=False,
    )
    plot_paths = {
        "floor_day": guardar_plot_control_negativo_combinado(enriched, output_dir),
        "barplot_exp": guardar_barplot_control_negativo_combinado(enriched, output_dir),
        "floor_barplot_raw_delta": guardar_barplot_pisos_control_negativo(
            enriched,
            output_dir,
            response_metric="raw_delta_weight",
        ),
        "floor_barplot_pct": guardar_barplot_pisos_control_negativo(
            enriched,
            output_dir,
            response_metric="pct_change",
        ),
        "floor_barplot_log": guardar_barplot_pisos_control_negativo(
            enriched,
            output_dir,
            response_metric="raw_log_ratio",
        ),
        "left_center_right_barplot_raw_delta": guardar_barplot_pisos_control_negativo(
            enriched,
            output_dir,
            response_metric="raw_delta_weight",
            group_column="position_left_center_right",
            group_order=POSITION_LEFT_CENTER_RIGHT_ORDER,
            group_xlabel="Ubicacion horizontal",
            group_title="izquierda, centro y derecha",
            output_suffix="left_center_right",
        ),
        "left_center_right_barplot_pct": guardar_barplot_pisos_control_negativo(
            enriched,
            output_dir,
            response_metric="pct_change",
            group_column="position_left_center_right",
            group_order=POSITION_LEFT_CENTER_RIGHT_ORDER,
            group_xlabel="Ubicacion horizontal",
            group_title="izquierda, centro y derecha",
            output_suffix="left_center_right",
        ),
        "left_center_right_barplot_log": guardar_barplot_pisos_control_negativo(
            enriched,
            output_dir,
            response_metric="raw_log_ratio",
            group_column="position_left_center_right",
            group_order=POSITION_LEFT_CENTER_RIGHT_ORDER,
            group_xlabel="Ubicacion horizontal",
            group_title="izquierda, centro y derecha",
            output_suffix="left_center_right",
        ),
        "front_middle_back_barplot_raw_delta": guardar_barplot_pisos_control_negativo(
            enriched,
            output_dir,
            response_metric="raw_delta_weight",
            group_column="position_front_middle_back",
            group_order=POSITION_FRONT_MIDDLE_BACK_ORDER,
            group_xlabel="Ubicacion frente-atras",
            group_title="frente, medio y atras",
            output_suffix="front_middle_back",
        ),
        "front_middle_back_barplot_pct": guardar_barplot_pisos_control_negativo(
            enriched,
            output_dir,
            response_metric="pct_change",
            group_column="position_front_middle_back",
            group_order=POSITION_FRONT_MIDDLE_BACK_ORDER,
            group_xlabel="Ubicacion frente-atras",
            group_title="frente, medio y atras",
            output_suffix="front_middle_back",
        ),
        "front_middle_back_barplot_log": guardar_barplot_pisos_control_negativo(
            enriched,
            output_dir,
            response_metric="raw_log_ratio",
            group_column="position_front_middle_back",
            group_order=POSITION_FRONT_MIDDLE_BACK_ORDER,
            group_xlabel="Ubicacion frente-atras",
            group_title="frente, medio y atras",
            output_suffix="front_middle_back",
        ),
        "tray_cell_barplot_raw_delta": guardar_barplot_pisos_control_negativo(
            enriched,
            output_dir,
            response_metric="raw_delta_weight",
            group_column="tray_cell",
            group_order=TRAY_CELL_ORDER,
            group_xlabel="Celda",
            group_title="celda A1-C9",
            output_suffix="tray_cell",
            figsize=(16, 6),
        ),
        "tray_cell_barplot_pct": guardar_barplot_pisos_control_negativo(
            enriched,
            output_dir,
            response_metric="pct_change",
            group_column="tray_cell",
            group_order=TRAY_CELL_ORDER,
            group_xlabel="Celda",
            group_title="celda A1-C9",
            output_suffix="tray_cell",
            figsize=(16, 6),
        ),
        "tray_cell_barplot_log": guardar_barplot_pisos_control_negativo(
            enriched,
            output_dir,
            response_metric="raw_log_ratio",
            group_column="tray_cell",
            group_order=TRAY_CELL_ORDER,
            group_xlabel="Celda",
            group_title="celda A1-C9",
            output_suffix="tray_cell",
            figsize=(16, 6),
        ),
        "floor_facets": guardar_plot_control_negativo_facetas(
            enriched,
            output_dir,
            group_column="floor",
            group_order=FLOOR_ORDER,
            group_label="Piso",
            titulo="Control negativo combinado: pisos A, B y C por fecha",
            output_filename="negative_control_combined_pct_change_floor_facets_by_day.png",
            ncols=3,
            facet_tests=facet_tests,
        ),
        "position_facets": guardar_plot_control_negativo_facetas(
            enriched,
            output_dir,
            group_column="position",
            group_order=list(range(1, 10)),
            group_label="Posicion",
            titulo="Control negativo combinado: posiciones 1 a 9 por fecha",
            output_filename="negative_control_combined_pct_change_position_facets_by_day.png",
            ncols=3,
            facet_tests=facet_tests,
        ),
    }
    plot_paths = {
        nombre: ruta for nombre, ruta in plot_paths.items() if ruta is not None
    }
    return output_dir, plot_paths


def guardar_barplot_control_negativo_combinado(
    data: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    if not cargar_dependencias_plotting():
        return None

    plot_data = data.dropna(subset=["pct_change", "exp_id"]).copy()
    if plot_data.empty:
        return None

    plot_data["exp_id"] = plot_data["exp_id"].astype(str)
    exp_order = sorted(plot_data["exp_id"].unique().tolist())

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        sns.barplot(
            data=plot_data,
            x="exp_id",
            y="pct_change",
            order=exp_order,
            errorbar="sd",
            color="#b8d8f8",
            capsize=0.1,
            ax=ax,
        )
    except TypeError:
        # Fallback para versiones de seaborn más antiguas
        sns.barplot(
            data=plot_data,
            x="exp_id",
            y="pct_change",
            order=exp_order,
            ci="sd",
            color="#b8d8f8",
            capsize=0.1,
            ax=ax,
        )

    sns.stripplot(
        data=plot_data,
        x="exp_id",
        y="pct_change",
        order=exp_order,
        color="#0b3954",
        alpha=0.6,
        size=5,
        jitter=0.2,
        ax=ax,
    )

    y_min = float(plot_data["pct_change"].min())
    y_max = float(plot_data["pct_change"].max())
    y_range = y_max - y_min
    y_margin = y_range * 0.1 if y_range > 0 else max(abs(y_min) * 0.1, 0.01)
    ax.set_ylim(min(y_min - y_margin, 0), y_max + y_margin)

    ax.set_title("Control negativo combinado: Cambio relativo por fecha (Media ± SD)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Cambio relativo ((post_weight - pre_weight) / pre_weight)")
    ax.axhline(0.0, color="#c0392b", linestyle="--", linewidth=1.5)
    ax.tick_params(axis="x", labelrotation=45)

    fig.tight_layout()
    output_path = output_dir / "negative_control_combined_pct_change_barplot.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def guardar_barplot_pisos_control_negativo(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    response_metric: str,
    group_column: str = "floor",
    group_order: list[str] | None = None,
    group_xlabel: str = "Piso",
    group_title: str = "piso",
    output_suffix: str = "floor",
    figsize: tuple[float, float] = (9, 6),
) -> Path | None:
    if not cargar_dependencias_plotting():
        return None

    labels = {
        "raw_delta_weight": "Delta crudo (post_weight - pre_weight)",
        "pct_change": "Cambio relativo ((post_weight - pre_weight) / pre_weight)",
        "raw_log_ratio": "log(post_weight / pre_weight)",
    }
    plot_data = data.dropna(subset=[group_column, "exp_id", response_metric]).copy()
    if plot_data.empty:
        return None

    plot_data[response_metric] = pd.to_numeric(
        plot_data[response_metric], errors="coerce"
    )
    plot_data = plot_data.dropna(subset=[response_metric])
    if plot_data.empty:
        return None

    plot_data[group_column] = plot_data[group_column].astype(str)
    plot_data["exp_id"] = plot_data["exp_id"].astype(str)
    if group_order is None:
        group_order = FLOOR_ORDER
    order = [
        value for value in group_order if value in plot_data[group_column].unique().tolist()
    ]
    if not order:
        order = sorted(plot_data[group_column].unique().tolist())
    exp_order = sorted(plot_data["exp_id"].unique().tolist())
    palette = dict(
        zip(exp_order, sns.color_palette("Set2", n_colors=max(len(exp_order), 3)))
    )

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=figsize)

    try:
        sns.barplot(
            data=plot_data,
            x=group_column,
            y=response_metric,
            order=order,
            errorbar="sd",
            color="#c9d6df",
            capsize=0.12,
            ax=ax,
        )
    except TypeError:
        sns.barplot(
            data=plot_data,
            x=group_column,
            y=response_metric,
            order=order,
            ci="sd",
            color="#c9d6df",
            capsize=0.12,
            ax=ax,
        )

    sns.stripplot(
        data=plot_data,
        x=group_column,
        y=response_metric,
        hue="exp_id",
        order=order,
        hue_order=exp_order,
        palette=palette,
        dodge=True,
        jitter=0.18,
        alpha=0.85,
        size=6,
        ax=ax,
    )

    y_min = float(plot_data[response_metric].min())
    y_max = float(plot_data[response_metric].max())
    y_range = y_max - y_min
    y_margin = y_range * 0.1 if y_range > 0 else max(abs(y_min) * 0.1, 0.01)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_title(
        "Control negativo combinado: "
        f"{labels.get(response_metric, response_metric)} por {group_title}"
    )
    ax.set_xlabel(group_xlabel)
    ax.set_ylabel(labels.get(response_metric, response_metric))
    ax.legend(title="Dia", frameon=True, loc="best")
    ax.tick_params(axis="x", labelrotation=45 if len(order) > 9 else 0)

    fig.tight_layout()
    output_path = (
        output_dir
        / f"negative_control_combined_{response_metric}_{output_suffix}_barplot.png"
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def guardar_plot_control_negativo_combinado(
    data: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    if not cargar_dependencias_plotting():
        return None

    plot_data = data.dropna(subset=["floor", "pct_change", "exp_id"]).copy()
    if plot_data.empty:
        return None

    plot_data["floor"] = plot_data["floor"].astype(str)
    plot_data["exp_id"] = plot_data["exp_id"].astype(str)
    orden_floor = [
        floor for floor in FLOOR_ORDER if floor in plot_data["floor"].unique().tolist()
    ]
    if not orden_floor:
        orden_floor = sorted(plot_data["floor"].unique().tolist())

    exp_order = sorted(plot_data["exp_id"].unique().tolist())
    palette = dict(
        zip(exp_order, sns.color_palette("Set2", n_colors=max(len(exp_order), 3)))
    )

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=plot_data,
        x="floor",
        y="pct_change",
        order=orden_floor,
        showfliers=False,
        color="#d8dee2",
        ax=ax,
    )
    sns.stripplot(
        data=plot_data,
        x="floor",
        y="pct_change",
        hue="exp_id",
        order=orden_floor,
        hue_order=exp_order,
        palette=palette,
        dodge=True,
        jitter=0.18,
        alpha=0.85,
        size=7,
        ax=ax,
    )
    y_min = float(plot_data["pct_change"].min())
    y_max = float(plot_data["pct_change"].max())
    y_range = y_max - y_min
    y_margin = y_range * 0.08 if y_range > 0 else max(abs(y_min) * 0.08, 0.01)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_title("Control negativo combinado: cambio relativo por piso")
    ax.set_xlabel("Piso")
    ax.set_ylabel("Cambio relativo ((post_weight - pre_weight) / pre_weight)")
    ax.legend(title="Día", frameon=True, loc="best")
    fig.tight_layout()
    output_path = output_dir / "negative_control_combined_pct_change_by_floor_day.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def guardar_plot_control_negativo_facetas(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    group_column: str,
    group_order: list[Any],
    group_label: str,
    titulo: str,
    output_filename: str,
    ncols: int,
    facet_tests: pd.DataFrame | None = None,
) -> Path | None:
    if not cargar_dependencias_plotting():
        return None

    plot_data = data.dropna(subset=[group_column, "pct_change", "exp_id"]).copy()
    if plot_data.empty:
        return None

    plot_data[group_column] = plot_data[group_column].astype(str)
    plot_data["exp_id"] = plot_data["exp_id"].astype(str)
    group_order = [
        str(valor)
        for valor in group_order
        if str(valor) in plot_data[group_column].unique().tolist()
    ]
    if not group_order:
        group_order = sorted(plot_data[group_column].unique().tolist())

    exp_order = sorted(plot_data["exp_id"].unique().tolist())
    test_by_group: dict[str, dict[str, Any]] = {}
    if facet_tests is not None and not facet_tests.empty:
        test_rows = facet_tests[facet_tests["group_column"].eq(group_column)].copy()
        test_by_group = {
            str(fila["group_value"]): fila.to_dict() for _, fila in test_rows.iterrows()
        }

    nrows = int(np.ceil(len(group_order) / ncols))
    fig_width = max(12, min(24, 1.25 * len(exp_order) * ncols))
    fig_height = max(4, 3.7 * nrows)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    y_min = float(plot_data["pct_change"].min())
    y_max = float(plot_data["pct_change"].max())
    y_range = y_max - y_min
    y_margin = y_range * 0.08 if y_range > 0 else max(abs(y_min) * 0.08, 0.01)

    for index, group_value in enumerate(group_order):
        ax = axes_flat[index]
        subset = plot_data[plot_data[group_column] == group_value].copy()
        test_result = test_by_group.get(group_value, {})
        q_value = test_result.get("q_value_fdr_bh", np.nan)
        reject_null = test_result.get("reject_null_fdr_bh", False)
        significant = False if pd.isna(reject_null) else bool(reject_null)

        sns.boxplot(
            data=subset,
            x="exp_id",
            y="pct_change",
            order=exp_order,
            showfliers=False,
            color="#d8dee2",
            ax=ax,
        )
        sns.stripplot(
            data=subset,
            x="exp_id",
            y="pct_change",
            order=exp_order,
            color="#0b3954",
            jitter=0.18,
            alpha=0.85,
            size=6,
            ax=ax,
        )
        summary_rows = []
        for exp_index, exp_id in enumerate(exp_order):
            valores = pd.to_numeric(
                subset.loc[subset["exp_id"].eq(exp_id), "pct_change"],
                errors="coerce",
            ).dropna()
            if valores.empty:
                continue

            mean = float(valores.mean())
            if len(valores) > 1:
                sem = float(stats.sem(valores))
                if np.isfinite(sem):
                    ci_delta = float(stats.t.ppf(0.975, len(valores) - 1) * sem)
                else:
                    ci_delta = 0.0
            else:
                ci_delta = 0.0

            summary_rows.append(
                {
                    "x": exp_index,
                    "mean": mean,
                    "ci_low": mean - ci_delta,
                    "ci_high": mean + ci_delta,
                }
            )

        if summary_rows:
            summary_data = pd.DataFrame(summary_rows)
            means = summary_data["mean"].to_numpy(dtype=float)
            yerr = np.vstack(
                [
                    means - summary_data["ci_low"].to_numpy(dtype=float),
                    summary_data["ci_high"].to_numpy(dtype=float) - means,
                ]
            )
            ax.errorbar(
                summary_data["x"].to_numpy(dtype=float),
                means,
                yerr=yerr,
                fmt="D",
                color="#1f2a25",
                ecolor="#1f2a25",
                elinewidth=1.5,
                capsize=4,
                markersize=5,
                zorder=10,
            )

        ax.axhline(0.0, color="#a94442", linestyle="--", linewidth=1)
        title_suffix = f" | q={_formatear_p_plot(q_value)}"
        if significant:
            title_suffix += " *"
            ax.set_facecolor("#fff5f5")
            ax.text(
                0.02,
                0.96,
                "diferencia entre fechas",
                transform=ax.transAxes,
                va="top",
                ha="left",
                color="#a94442",
                fontsize=9,
                fontweight="bold",
            )
        ax.set_title(
            f"{group_label} {group_value}{title_suffix}",
            color="#a94442" if significant else "#1f2a25",
        )
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Cambio relativo" if index % ncols == 0 else "")
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.tick_params(axis="x", labelrotation=45)

    for ax in axes_flat[len(group_order) :]:
        ax.set_visible(False)

    fig.suptitle(titulo, y=1.02)
    fig.tight_layout()
    output_path = output_dir / output_filename
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_negative_control_analysis(
    exp_ids: list[str] | tuple[str, ...],
    verbose: bool = True,
) -> dict[str, Any]:
    if sm is None or smf is None:
        raise RuntimeError(
            "No se encontro 'statsmodels'. Instala dependencias con "
            "'pip install -r requirements.txt' antes de ejecutar este analisis."
        )
    if len(exp_ids) < 2:
        raise ValueError(
            "Selecciona al menos dos experimentos para combinar controles negativos."
        )

    metadata, advertencias = leer_metadata_global()
    trays_paths = descubrir_trays(exp_ids=exp_ids)
    experimentos: list[pd.DataFrame] = []
    resumenes_experimentos: list[dict[str, Any]] = []

    for trays_path in trays_paths:
        enriched, warnings_exp = cargar_experimento(trays_path, metadata)
        if not es_experimento_control_negativo(enriched):
            raise ValueError(
                f"El experimento {trays_path.parent.name} no es control negativo "
                "(metadata experiment_type debe ser CTRL_NEGATIVO)."
            )
        models = modelos_experimento(enriched)
        summary = resumen_experimento(enriched)
        comparisons = comparaciones_tratamiento_vs_control_experimento(enriched)
        guardar_experimento(enriched, summary, models, comparisons)
        experimentos.append(enriched)
        resumenes_experimentos.append(construir_resumen_experimento(enriched, models))
        advertencias.extend(warnings_exp)

    combined = agregar_agrupaciones_posicion(pd.concat(experimentos, ignore_index=True))
    summary = resumen_global(combined)
    model_parts = [_one_sample_tests_control_negativo(combined)]
    for metrica in (
        "raw_delta_weight",
        "pct_change",
        "raw_log_ratio",
        "log_ratio_centered",
    ):
        model_parts.append(
            _ajustar_modelo_control_negativo_combinado(
                combined,
                response_metric=metrica,
            )
        )
    models = pd.concat(model_parts, ignore_index=True)
    assumptions = diagnosticos_supuestos_control_negativo(combined)
    floor_tests = comparar_pisos_control_negativo(combined)
    position_group_tests = comparar_agrupaciones_posicion_control_negativo(combined)
    tray_cell_tests = comparar_celdas_control_negativo(combined)
    output_dir, plot_paths = guardar_control_negativo_combinado(
        combined,
        summary,
        models,
        assumptions,
        floor_tests,
        position_group_tests,
        tray_cell_tests,
    )
    plot_path = output_dir / "negative_control_combined_pct_change_by_floor_day.png"
    combined_summary = resumen_control_negativo_combinado(combined, models)

    advertencias_unicas = sorted(set(advertencias))
    if verbose:
        imprimir_resumen(resumenes_experimentos, advertencias_unicas)
        print()
        print("=" * 72)
        print("Control negativo combinado")
        print("=" * 72)
        print(f"Experimentos: {', '.join(combined_summary['exp_ids'])}")
        print(
            "Datos: "
            f"{combined_summary['total_rows']} trays | "
            f"delta {combined_summary['valid_delta_rows']}/{combined_summary['total_rows']} | "
            f"pct {combined_summary['valid_pct_rows']}/{combined_summary['total_rows']} | "
            f"log {combined_summary['valid_log_rows']}/{combined_summary['total_rows']}"
        )
        print(f"Archivos: {output_dir}")
        for ruta_plot in plot_paths.values():
            if ruta_plot.exists():
                print(f"Plot: {ruta_plot}")
        print(
            "Supuestos: "
            f"{output_dir / 'negative_control_combined_assumption_tests.csv'}"
        )
        print(
            "Comparacion A/B/C: "
            f"{output_dir / 'negative_control_combined_floor_tests.csv'}"
        )
        print(
            "Comparacion posiciones agrupadas: "
            f"{output_dir / 'negative_control_combined_position_group_tests.csv'}"
        )
        print(
            "Comparacion celdas A1-C9: "
            f"{output_dir / 'negative_control_combined_tray_cell_tests.csv'}"
        )

    return {
        "experiment_data": experimentos,
        "experiment_summaries": resumenes_experimentos,
        "combined_enriched": combined,
        "combined_summary_table": summary,
        "combined_models": models,
        "combined_assumption_tests": assumptions,
        "combined_floor_tests": floor_tests,
        "combined_position_group_tests": position_group_tests,
        "combined_tray_cell_tests": tray_cell_tests,
        "combined_summary": combined_summary,
        "warnings": advertencias_unicas,
        "output_dir": str(output_dir),
        "plot_path": str(plot_path) if plot_path.exists() else "",
        "facet_tests_path": str(
            output_dir / "negative_control_combined_facet_tests.csv"
        ),
        "assumption_tests_path": str(
            output_dir / "negative_control_combined_assumption_tests.csv"
        ),
        "floor_tests_path": str(
            output_dir / "negative_control_combined_floor_tests.csv"
        ),
        "position_group_tests_path": str(
            output_dir / "negative_control_combined_position_group_tests.csv"
        ),
        "tray_cell_tests_path": str(
            output_dir / "negative_control_combined_tray_cell_tests.csv"
        ),
        "plot_paths": {
            nombre: str(ruta) for nombre, ruta in plot_paths.items() if ruta.exists()
        },
        "global_analysis_dir": str(GLOBAL_ANALYSIS_DIR),
    }


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
    experiment_type = str(enriched["experiment_type"].iloc[0]).strip()
    magnet_pole = str(enriched["magnet_pole"].iloc[0]).strip()
    has_complete_metadata = bool(enriched["has_metadata"].all())
    has_active_magnet = bool(enriched["has_active_magnet"].any())
    is_negative_control = es_experimento_control_negativo(enriched)

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
        model_name="delta_spatial_effect",
    )
    p_position_delta = _valor_p_term(
        models,
        "C(position_factor)",
        "delta_weight",
        model_name="delta_spatial_effect",
    )

    total = len(enriched)
    valid_delta = int(enriched["valid_delta"].sum())
    valid_pct = int(enriched["valid_pct_change"].sum())
    valid_log = int(enriched["valid_log_ratio"].sum())
    magnetic_rows = int(enriched["treatment_role"].eq("magnetic").sum())
    control_rows = int(enriched["treatment_role"].eq("control").sum())
    control_delta_mean = pd.to_numeric(
        enriched["control_delta_mean"], errors="coerce"
    ).dropna()
    control_log_ratio_mean = pd.to_numeric(
        enriched["control_log_ratio_mean"], errors="coerce"
    ).dropna()

    if is_negative_control:
        experiment_kind_label = "Control negativo"
    elif has_active_magnet:
        experiment_kind_label = f"Tratamiento magnetico ({experiment_type}/{magnet_pole})"
    elif has_complete_metadata:
        experiment_kind_label = (
            f"Experimento sin tratamiento magnetico activo ({experiment_type}/{magnet_pole})"
        )
    else:
        experiment_kind_label = "Metadata incompleta"

    return {
        "exp_id": str(enriched["exp_id"].iloc[0]),
        "experiment_type": experiment_type,
        "magnet": experiment_type,
        "magnet_pole": magnet_pole,
        "is_negative_control": is_negative_control,
        "experiment_kind_label": experiment_kind_label,
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
            float(control_delta_mean.iloc[0])
            if not control_delta_mean.empty
            else np.nan
        ),
        "control_delta_count": (
            int(pd.to_numeric(enriched["control_delta_count"], errors="coerce").max())
            if "control_delta_count" in enriched
            else 0
        ),
        "control_log_ratio_mean": (
            float(control_log_ratio_mean.iloc[0])
            if not control_log_ratio_mean.empty
            else np.nan
        ),
        "control_log_ratio_count": (
            int(
                pd.to_numeric(
                    enriched["control_log_ratio_count"], errors="coerce"
                ).max()
            )
            if "control_log_ratio_count" in enriched
            else 0
        ),
        "floor_delta_p_value": p_floor_delta,
        "position_delta_p_value": p_position_delta,
        "has_complete_metadata": has_complete_metadata,
        "exp_dir": str(exp_dir),
        "processed_dir": str(exp_dir / "processed"),
        "plots_dir": str(exp_dir / "plots"),
    }


def _valor_disponible(valor: float) -> bool:
    return not np.isnan(valor)


def _formatear_valor(valor: float, *, decimales: int = 3) -> str:
    if np.isnan(valor):
        return "n/d"
    return f"{valor:.{decimales}f}"


def _formatear_p_value(valor: float) -> str:
    if np.isnan(valor):
        return "n/d"
    if valor < 0.001:
        return "<0.001"
    return f"{valor:.3f}"


def _interpretar_efecto_espacial(p_floor: float, p_position: float) -> str:
    disponibles = [valor for valor in (p_floor, p_position) if not np.isnan(valor)]
    if not disponibles:
        return "sin test disponible"

    minimo = min(disponibles)
    if minimo < 0.05:
        return "hay señal espacial"
    if minimo < 0.10:
        return "hay una señal espacial débil"
    return "sin señal espacial clara"


def _resumir_coef_y_p(coef: float, p_value: float) -> str:
    if np.isnan(coef) or np.isnan(p_value):
        return "no disponible"
    return f"coef={coef:.3f} | p={_formatear_p_value(p_value)}"


def imprimir_resumen(
    resumenes_experimentos: list[dict[str, Any]],
    advertencias: list[str],
) -> None:
    print("=" * 72)
    print(f"Resumen de análisis | Experimentos: {len(resumenes_experimentos)}")
    print("=" * 72)

    for resumen in resumenes_experimentos:
        print()
        print(f"{resumen['exp_id']} | {resumen['experiment_kind_label']}")
        print("-" * 72)
        print(
            "Datos: "
            f"{resumen['total_rows']} trays | "
            f"delta {resumen['valid_delta_rows']}/{resumen['total_rows']} | "
            f"pct {resumen['valid_pct_rows']}/{resumen['total_rows']} | "
            f"log {resumen['valid_log_rows']}/{resumen['total_rows']}"
        )
        print(
            "Grupos: "
            f"magnetic={resumen['magnetic_rows']} | control={resumen['control_rows']} | "
            f"metadata={'completa' if resumen['has_complete_metadata'] else 'incompleta'}"
        )
        print(
            "Referencia: "
            f"delta_control={_formatear_valor(resumen['control_delta_mean'])} "
            f"(n={resumen['control_delta_count']}) | "
            f"log_control={_formatear_valor(resumen['control_log_ratio_mean'])} "
            f"(n={resumen['control_log_ratio_count']})"
        )
        print(
            "Intensidad magnética: "
            f"delta { _resumir_coef_y_p(resumen['magnetic_intensity_delta_coef'], resumen['magnetic_intensity_delta_p_value']) } | "
            f"pct { _resumir_coef_y_p(resumen['magnetic_intensity_pct_coef'], resumen['magnetic_intensity_pct_p_value']) } | "
            f"log { _resumir_coef_y_p(resumen['magnetic_intensity_log_coef'], resumen['magnetic_intensity_log_p_value']) }"
        )

        comparacion_disponible = any(
            _valor_disponible(valor)
            for valor in (
                resumen["magnetic_vs_control_delta_p_value"],
                resumen["magnetic_vs_control_pct_p_value"],
                resumen["magnetic_vs_control_log_p_value"],
            )
        )
        if comparacion_disponible:
            print(
                "Magnetic vs control: "
                f"delta p={_formatear_p_value(resumen['magnetic_vs_control_delta_p_value'])} | "
                f"pct p={_formatear_p_value(resumen['magnetic_vs_control_pct_p_value'])} | "
                f"log p={_formatear_p_value(resumen['magnetic_vs_control_log_p_value'])}"
            )
        else:
            print("Magnetic vs control: no disponible")

        print(
            "Efecto espacial (delta): "
            f"floor p={_formatear_p_value(resumen['floor_delta_p_value'])} | "
            f"position p={_formatear_p_value(resumen['position_delta_p_value'])} | "
            f"{_interpretar_efecto_espacial(resumen['floor_delta_p_value'], resumen['position_delta_p_value'])}"
        )
        print(
            f"Archivos: processed={resumen['processed_dir']} | plots={resumen['plots_dir']}"
        )

    if advertencias:
        print()
        print("Advertencias")
        print("-" * 72)
        for advertencia in sorted(set(advertencias)):
            print(f"- {advertencia}")


def _get_negative_control_global_mean(metadata: pd.DataFrame) -> float:
    """Calcula la media de raw_log_ratio de todos los controles negativos."""
    neg_control_exp_ids = metadata[
        metadata["experiment_type"].isin(NEGATIVE_CONTROL_EXPERIMENT_TYPES)
    ]["exp_id"].tolist()

    if not neg_control_exp_ids:
        return np.nan

    try:
        paths = descubrir_trays(exp_ids=neg_control_exp_ids)
    except FileNotFoundError:
        return np.nan

    all_log_ratios = []
    for path in paths:
        df, _ = cargar_experimento(path, metadata)
        valid_log_mask = (
            df["has_complete_weights"]
            & (df["raw_pre_weight"] > 0)
            & (df["raw_post_weight"] > 0)
        )
        if valid_log_mask.any():
            log_ratios = np.log(
                df.loc[valid_log_mask, "raw_post_weight"]
                / df.loc[valid_log_mask, "raw_pre_weight"]
            )
            all_log_ratios.append(log_ratios.dropna())

    if not all_log_ratios:
        return np.nan

    return float(pd.concat(all_log_ratios, ignore_index=True).mean())


def run_analysis(
    exp_id: str | None = None,
    verbose: bool = True,
    exp_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if sm is None or smf is None:
        raise RuntimeError(
            "No se encontro 'statsmodels'. Instala dependencias con "
            "'pip install -r requirements.txt' antes de ejecutar este analisis."
        )

    metadata, advertencias = leer_metadata_global()
    neg_control_global_mean = _get_negative_control_global_mean(metadata)

    trays_paths = descubrir_trays(exp_id=exp_id, exp_ids=exp_ids)

    experimentos: list[pd.DataFrame] = []
    modelos: list[pd.DataFrame] = []
    comparaciones: list[pd.DataFrame] = []
    resumenes_experimentos: list[dict[str, Any]] = []

    for trays_path in trays_paths:
        enriched, warnings_exp = cargar_experimento(trays_path, metadata)
        if not es_experimento_control_negativo(enriched) and np.isfinite(neg_control_global_mean) and np.isfinite(enriched["control_log_ratio_mean"].iloc[0]):
            offset = enriched["control_log_ratio_mean"].iloc[0] - neg_control_global_mean
            enriched["log_ratio_centered"] = enriched["raw_log_ratio"] - offset

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
