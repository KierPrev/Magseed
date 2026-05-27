from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REQUIRED_COLUMNS = {"pre_weight", "post_weight", "water_ml"}


def load_absorption_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Ensure weight columns exist
    weight_cols = {"pre_weight", "post_weight"}
    missing_weights = weight_cols - set(df.columns)
    if missing_weights:
        raise ValueError(
            f"Faltan columnas de peso requeridas en {csv_path}: {', '.join(sorted(missing_weights))}"
        )

    result = df.copy()
    for column in weight_cols:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    # If water_ml is missing, try to obtain it from a treatment.csv file
    if "water_ml" not in result.columns:
        treatment_path = csv_path.parent / "treatment.csv"
        if not treatment_path.exists():
            raise ValueError(
                f"No hay columna 'water_ml' en {csv_path} y no existe {treatment_path} para mapearla."
            )

        tdf = pd.read_csv(treatment_path)
        needed = {"treatment_id", "treatment_value"}
        if not needed <= set(tdf.columns):
            raise ValueError(
                f"{treatment_path} debe contener las columnas: {', '.join(sorted(needed))}"
            )

        if "treatment_floor" not in result.columns:
            raise ValueError(
                f"Para mapear agua desde {treatment_path} se necesita la columna 'treatment_floor' en {csv_path}."
            )

        map_df = tdf[["treatment_id", "treatment_value"]].copy()
        map_df["treatment_value"] = pd.to_numeric(
            map_df["treatment_value"], errors="coerce"
        )

        merged = result.merge(
            map_df,
            left_on="treatment_floor",
            right_on="treatment_id",
            how="left",
            suffixes=(None, "_t"),
        )
        # Prefer any existing water_ml, otherwise use treatment_value
        if "water_ml" in merged.columns:
            merged["water_ml"] = pd.to_numeric(merged["water_ml"], errors="coerce")
            merged["water_ml"] = merged["water_ml"].fillna(merged["treatment_value"])
        else:
            merged["water_ml"] = merged["treatment_value"]

        result = merged
    else:
        result["water_ml"] = pd.to_numeric(result["water_ml"], errors="coerce")

    # Keep only rows with numeric weights and water_ml
    required_now = ["pre_weight", "post_weight", "water_ml"]
    result = result.dropna(subset=required_now).copy()
    if result.empty:
        raise ValueError(f"No hay filas numericas validas en {csv_path}.")

    result["absolute_absorption"] = result["post_weight"] - result["pre_weight"]
    result["relative_absorption_pct"] = (
        result["absolute_absorption"] / result["pre_weight"]
    ) * 100

    return result


def default_output_dir(csv_path: Path) -> Path:
    return csv_path.parent / "plots"


def plot_water_vs_relative_absorption(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(
        data=df,
        x="water_ml",
        y="relative_absorption_pct",
        hue="water_ml",
        palette="viridis",
        s=95,
        edgecolor="black",
        linewidth=0.7,
        legend=False,
        ax=ax,
    )
    sns.regplot(
        data=df,
        x="water_ml",
        y="relative_absorption_pct",
        scatter=False,
        color="#2f3e46",
        line_kws={"linewidth": 1.8},
        ax=ax,
    )

    ax.set_title("Hidratacion segun volumen de agua disponible")
    ax.set_xlabel("Agua disponible (ml)")
    ax.set_ylabel("Absorcion relativa (%)")
    ax.margins(x=0.08, y=0.12)
    fig.tight_layout()
    return fig


def plot_relative_absorption_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    water_order = sorted(df["water_ml"].dropna().unique())

    sns.boxplot(
        data=df,
        x="water_ml",
        y="relative_absorption_pct",
        order=water_order,
        color="#a8dadc",
        width=0.5,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="water_ml",
        y="relative_absorption_pct",
        order=water_order,
        color="#1d3557",
        size=7,
        alpha=0.85,
        jitter=0.14,
        ax=ax,
    )

    ax.set_title("Distribucion de absorcion relativa por nivel de agua")
    ax.set_xlabel("Agua disponible (ml)")
    ax.set_ylabel("Absorcion relativa (%)")
    fig.tight_layout()
    return fig


def plot_pre_weight_vs_absolute_absorption(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(
        data=df,
        x="pre_weight",
        y="absolute_absorption",
        hue="water_ml",
        palette="mako",
        s=95,
        edgecolor="black",
        linewidth=0.7,
        ax=ax,
    )
    sns.regplot(
        data=df,
        x="pre_weight",
        y="absolute_absorption",
        scatter=False,
        color="#2f3e46",
        line_kws={"linewidth": 1.8},
        ax=ax,
    )

    ax.set_title("Control de sesgo por masa inicial")
    ax.set_xlabel("Peso inicial seco (g)")
    ax.set_ylabel("Absorcion absoluta (g)")
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Agua (ml)")
    fig.tight_layout()
    return fig


def build_plots(csv_path: Path, output_dir: Path | None = None) -> list[Path]:
    df = load_absorption_data(csv_path)
    if output_dir is None:
        output_dir = default_output_dir(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")
    plots = [
        (
            "water_vs_relative_absorption.png",
            plot_water_vs_relative_absorption(df),
        ),
        (
            "relative_absorption_by_water_boxplot.png",
            plot_relative_absorption_distribution(df),
        ),
        (
            "pre_weight_vs_absolute_absorption.png",
            plot_pre_weight_vs_absolute_absorption(df),
        ),
    ]

    saved_paths: list[Path] = []
    for filename, fig in plots:
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=200)
        saved_paths.append(output_path)

    plt.show()
    return saved_paths


def build_plots_from_df(df: pd.DataFrame, output_dir: Path | None = None) -> list[Path]:
    if output_dir is None:
        raise ValueError("output_dir must be provided when plotting from a DataFrame")
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")
    plots = [
        ("water_vs_relative_absorption.png", plot_water_vs_relative_absorption(df)),
        (
            "relative_absorption_by_water_boxplot.png",
            plot_relative_absorption_distribution(df),
        ),
        (
            "pre_weight_vs_absolute_absorption.png",
            plot_pre_weight_vs_absolute_absorption(df),
        ),
    ]

    saved_paths: list[Path] = []
    for filename, fig in plots:
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=200)
        saved_paths.append(output_path)

    plt.show()
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera tres graficos de absorcion para trays.csv."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Ruta al CSV de trays. Si no se especifica, usar --date.",
    )
    parser.add_argument(
        "--date",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Una o varias fechas del experimento en formato YYYYMMDD. "
            "Ejemplo: --date 20260522 20260518. Usado para localizar "
            "data/experiments/YYYYMMDD/trays.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Carpeta de salida. Si se omite, usa la subcarpeta plots del experimento.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv
    if csv_path is None:
        if args.date is None:
            raise SystemExit("Es necesario pasar --csv o --date (YYYYMMDD).")

        # Validate and collect CSV paths for all provided dates
        dates = args.date
        for d in dates:
            if len(d) != 8 or not d.isdigit():
                raise SystemExit(
                    f"--date '{d}' debe tener formato YYYYMMDD, por ejemplo 20260522."
                )

        csv_paths = [Path(f"data/experiments/{d}/trays.csv") for d in dates]
        for p in csv_paths:
            if not p.exists():
                raise SystemExit(f"No existe el archivo esperado: {p}")

        # Load and combine data from all dates
        dfs = [load_absorption_data(p) for p in csv_paths]
        combined_df = pd.concat(dfs, ignore_index=True, sort=False)

        # Determine output dir (use first date's default if none provided)
        if args.output_dir is None:
            output_dir = default_output_dir(csv_paths[0])
        else:
            output_dir = args.output_dir

        saved_paths = build_plots_from_df(combined_df, output_dir)
    else:
        saved_paths = build_plots(csv_path, args.output_dir)
    for output_path in saved_paths:
        print(f"[OK] Plot guardado en: {output_path}")


if __name__ == "__main__":
    main()
