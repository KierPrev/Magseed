from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MIN_VALID_TEMP_C = -40.0
MAX_VALID_TEMP_C = 80.0
TRIM_EDGE_MINUTES = 10.0


def parse_temps_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_columns = {"Timestamp", "Mode", "Data"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas en {csv_path}: {', '.join(sorted(missing))}"
        )

    parsed = df["Data"].astype(str).str.split(",", n=1, expand=True)
    if parsed.shape[1] != 2:
        raise ValueError(
            f"La columna Data de {csv_path} no tiene el formato esperado 'elapsed_ms,temp_c'."
        )

    result = df.copy()
    result["elapsed_ms"] = pd.to_numeric(parsed[0], errors="coerce")
    result["temp_c"] = pd.to_numeric(parsed[1], errors="coerce")
    result = result.dropna(subset=["elapsed_ms", "temp_c"]).copy()
    filtered = result[
        result["temp_c"].between(MIN_VALID_TEMP_C, MAX_VALID_TEMP_C)
    ].copy()

    if filtered.empty:
        raise ValueError(f"No hay datos numericos validos en {csv_path}.")

    min_elapsed_ms = filtered["elapsed_ms"].min()
    max_elapsed_ms = filtered["elapsed_ms"].max()
    trim_ms = TRIM_EDGE_MINUTES * 60_000.0

    filtered = filtered[
        filtered["elapsed_ms"].between(
            min_elapsed_ms + trim_ms,
            max_elapsed_ms - trim_ms,
        )
    ].copy()

    if filtered.empty:
        raise ValueError(
            f"No quedan datos validos en {csv_path} despues de recortar el primer y ultimo minuto."
        )

    filtered = filtered.sort_values("elapsed_ms").reset_index(drop=True)
    filtered["elapsed_min"] = (
        filtered["elapsed_ms"] - filtered["elapsed_ms"].min()
    ) / 60_000.0

    filtered.attrs["dropped_rows"] = len(result) - len(filtered)
    return filtered.sort_values("elapsed_ms").reset_index(drop=True)


def default_output_path(csv_path: Path) -> Path:
    experiment_dir = csv_path.parent.parent
    return experiment_dir / "plots" / "temps_over_time.png"


def build_plot(csv_path: Path, output_path: Path) -> None:
    df = parse_temps_csv(csv_path)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="elapsed_min",
        y="temp_c",
        hue="Mode",
        linewidth=1.8,
        palette="deep",
        ax=ax,
    )

    experiment_id = csv_path.parent.parent.name
    ax.set_title(f"Temperatura vs tiempo ({experiment_id})")
    ax.set_xlabel("Tiempo transcurrido (min)")
    ax.set_ylabel("Temperatura (C)")
    ax.margins(x=0.01)

    temp_min = df["temp_c"].min()
    temp_max = df["temp_c"].max()
    duration_min = df["elapsed_min"].max()
    dropped_rows = int(df.attrs.get("dropped_rows", 0))
    summary = (
        f"n = {len(df)}\n"
        f"Duracion = {duration_min:.1f} min\n"
        f"Rango = {temp_min:.2f} a {temp_max:.2f} C"
    )
    if dropped_rows:
        summary += f"\nDescartadas = {dropped_rows}"
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )

    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Mode")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def discover_csvs(root: Path) -> list[Path]:
    return sorted(root.glob("*/raw/temps.csv"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera plots de temperatura vs tiempo para archivos temps.csv."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/experiments"),
        help="Directorio raiz donde buscar experiments/*/raw/temps.csv.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        default=None,
        help="Ruta(s) especifica(s) a temps.csv. Si se omite, busca todos bajo --root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = args.csv if args.csv else discover_csvs(args.root)

    if not csv_paths:
        raise SystemExit("No se encontraron archivos temps.csv para graficar.")

    generated: list[Path] = []
    for csv_path in csv_paths:
        output_path = default_output_path(csv_path)
        build_plot(csv_path, output_path)
        generated.append(output_path)
        print(f"[OK] Plot guardado en: {output_path}")

    print(f"[OK] Total de plots generados: {len(generated)}")


if __name__ == "__main__":
    main()
