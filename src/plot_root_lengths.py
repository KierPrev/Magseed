from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_tray_order(csv_path: Path, trays_csv_path: Path | None) -> list[str]:
    if trays_csv_path is not None and trays_csv_path.exists():
        trays_df = pd.read_csv(trays_csv_path)
        if {"floor", "position"}.issubset(trays_df.columns):
            ordered = (
                trays_df.copy()
                .assign(position_num=pd.to_numeric(trays_df["position"], errors="coerce"))
                .sort_values(["floor", "position_num"], na_position="last")
            )
            return [
                f"{floor}{int(position)}"
                for floor, position in zip(
                    ordered["floor"].astype(str),
                    ordered["position_num"],
                )
                if pd.notna(position)
            ]

    df = pd.read_csv(csv_path, usecols=["tray_id"])
    groups = (
        df["tray_id"]
        .astype(str)
        .str.extract(r"^[^_]+_([^_]+)")[0]
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(groups, key=lambda value: (value[:1], int(value[1:])))


def build_plot(
    csv_path: Path,
    output_path: Path,
    trays: list[str] | None = None,
    trays_csv_path: Path | None = None,
) -> None:
    df = pd.read_csv(csv_path)

    required_columns = {"tray_id", "length_mm"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas en {csv_path}: {', '.join(sorted(missing))}"
        )

    df = df.copy()
    df["tray_group"] = df["tray_id"].astype(str).str.extract(r"^[^_]+_([^_]+)")
    df["length_mm"] = pd.to_numeric(df["length_mm"], errors="coerce")
    df = df.dropna(subset=["tray_group", "length_mm"])

    if trays is None:
        trays = load_tray_order(csv_path, trays_csv_path)

    plot_df = df[df["tray_group"].isin(trays)].copy()
    if plot_df.empty:
        raise ValueError(
            "No hay datos para los trays pedidos: " + ", ".join(trays)
        )

    sns.set_theme(style="whitegrid", context="notebook")
    fig_width = max(10, len(trays) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    sns.boxplot(
        data=plot_df,
        x="tray_group",
        y="length_mm",
        order=trays,
        showfliers=False,
        color="#b8d8f8",
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="tray_group",
        y="length_mm",
        order=trays,
        color="#0b3954",
        alpha=0.75,
        size=4,
        jitter=0.22,
        ax=ax,
    )

    ax.set_title("Distribucion de length_mm por tray")
    ax.set_xlabel("Tray")
    ax.set_ylabel("length_mm")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un boxplot de length_mm por tray."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/experiments/20260331/processed/root_lengths.csv"),
        help="Ruta al CSV de entrada.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/20260331/plots/root_lengths_boxplot.png"),
        help="Ruta del PNG de salida.",
    )
    parser.add_argument(
        "--trays-csv",
        type=Path,
        default=Path("data/experiments/20260331/trays.csv"),
        help="CSV con el orden de bandejas. Si existe, se usa para ordenar el eje X.",
    )
    parser.add_argument(
        "--trays",
        nargs="+",
        default=None,
        help="Lista de trays a incluir en el eje X. Si se omite, usa todos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_plot(args.csv, args.output, args.trays, args.trays_csv)
    print(f"[OK] Plot guardado en: {args.output}")


if __name__ == "__main__":
    main()
