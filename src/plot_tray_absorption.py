from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REQUIRED_COLUMNS = {"pre_weight", "post_weight", "water_ml"}


def load_absorption_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas en {csv_path}: {', '.join(sorted(missing))}"
        )

    result = df.copy()
    for column in REQUIRED_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(subset=list(REQUIRED_COLUMNS)).copy()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera tres graficos de absorcion para trays.csv."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/experiments/20260518/trays.csv"),
        help="Ruta al CSV de trays.",
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
    saved_paths = build_plots(args.csv, args.output_dir)
    for output_path in saved_paths:
        print(f"[OK] Plot guardado en: {output_path}")


if __name__ == "__main__":
    main()
