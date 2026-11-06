from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

# Dimensiones de cada iman en mm, con el eje de magnetizacion (el mas corto)
# ya puesto en la tercera posicion (z).
MAGNET_DIMENSIONS_MM: dict[str, tuple[float, float, float]] = {
    "ceramica": (100.0, 80.0, 30.5),
    "neodimio": (50.8, 50.8, 25.3),
}

# Puntos de medicion en el plano de la cara, relativos al centro (mm).
CORNER_OFFSET_MM = 30.0
MEASUREMENT_POINTS: dict[str, tuple[float, float]] = {
    "Centro": (0.0, 0.0),
    "Esq A": (CORNER_OFFSET_MM, CORNER_OFFSET_MM),
    "Esq B": (CORNER_OFFSET_MM, -CORNER_OFFSET_MM),
    "Esq C": (-CORNER_OFFSET_MM, CORNER_OFFSET_MM),
    "Esq D": (-CORNER_OFFSET_MM, -CORNER_OFFSET_MM),
}
CORNER_LABELS = [label for label in MEASUREMENT_POINTS if label != "Centro"]

CSV_COLUMN_BY_POINT = {
    "Centro": "Centro",
    "Esq A": "Esq A",
    "Esq B": "Esq B",
    "Esq C": "Esq C",
    "Esq D": "Esq D",
}

# Los 3 imanes caracterizados. Por defecto el script corre los 3 de una.
DEFAULT_JOBS = [
    {
        "magnet": "ceramica",
        "csv": Path("data/extra/Caracterización imanes - Ceramica N.csv"),
        "output": Path("data/extra/plots/ceramica_n_bz_fit.png"),
        "title": "Ceramica N: Bz medido vs. simulado",
    },
    {
        "magnet": "ceramica",
        "csv": Path("data/extra/Caracterización imanes - Ceramica S.csv"),
        "output": Path("data/extra/plots/ceramica_s_bz_fit.png"),
        "title": "Ceramica S: Bz medido vs. simulado",
    },
    {
        "magnet": "neodimio",
        "csv": Path("data/extra/Caracterización imanes - Neodimio N.csv"),
        "output": Path("data/extra/plots/neodimio_n_bz_fit.png"),
        "title": "Neodimio N: Bz medido vs. simulado",
    },
]



def guardar_figura(fig, output_path, **kwargs) -> None:
    """PNG para mirar y PDF para LaTeX.

    El PDF es el unico de los dos que queda versionado, porque el
    .gitignore bloquea *.png. Sin el, el documento no compila en un clon
    limpio del repositorio.
    """
    destino = Path(output_path)
    fig.savefig(destino.with_suffix(".png"), dpi=200, **kwargs)
    fig.savefig(destino.with_suffix(".pdf"), **kwargs)


def load_measurements(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, decimal=",")
    required = {"Tick Distancia", "Distancia real", *CSV_COLUMN_BY_POINT.values()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas en {csv_path}: {', '.join(sorted(missing))}"
        )

    # La sonda tiene un offset de carcasa fijo que "Distancia real" ya debe
    # incluir (una sola vez) por sobre la posicion cruda del carro (Tick
    # Distancia). Si el offset no fuera constante en todas las filas, seria
    # señal de doble conteo o de un dato mal cargado.
    offset = df["Distancia real"] - df["Tick Distancia"]
    if offset.max() - offset.min() > 0.05:
        raise ValueError(
            f"El offset de la sonda no es constante en {csv_path} "
            f"(rango {offset.min():.2f}-{offset.max():.2f} mm); revisar doble conteo."
        )

    return df.sort_values("Distancia real").reset_index(drop=True)


def simulate_bz_unit(
    dimension_mm: tuple[float, float, float],
    distances_mm: np.ndarray,
    xy_offset: tuple[float, float],
) -> np.ndarray:
    """Bz (mT) para cada distancia, con remanencia unitaria (Br = 1 mT)."""
    thickness = dimension_mm[2]
    cube = magpy.magnet.Cuboid(polarization=(0.0, 0.0, 1.0), dimension=dimension_mm)
    z = thickness / 2 + np.asarray(distances_mm, dtype=float)
    x, y = xy_offset
    observers = np.column_stack([np.full_like(z, x), np.full_like(z, y), z])
    b = cube.getB(observers)
    return b[:, 2]


def _r2_rmse(measured: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    residuals = measured - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((measured - measured.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return r2, rmse


def fit_remanence_centro(
    measured_centro: np.ndarray, unit_centro: np.ndarray
) -> tuple[float, float, float]:
    """Ajusta Br (mT) por minimos cuadrados usando solo el Centro.

    El Centro es el punto geometricamente menos ambiguo (sobre el eje, campo
    mas uniforme). Las esquinas quedan afuera del ajuste porque para el
    neodimio caen fuera de la cara del iman: ahi el campo de retorno se
    curva y Bz es muy sensible a la posicion exacta y al promediado espacial
    de la sonda, asi que si entraran al fit contaminarian Br con ese ruido
    geometrico en vez de reflejar el material.

    Bz es lineal en Br (superposicion), asi que el ajuste tiene solucion
    cerrada: Br = sum(medido * unit) / sum(unit^2).
    """
    br_fit = float(np.sum(measured_centro * unit_centro) / np.sum(unit_centro**2))
    predicted = br_fit * unit_centro
    r2, rmse = _r2_rmse(measured_centro, predicted)
    return br_fit, r2, rmse


def validate_corners(
    measured_by_point: dict[str, np.ndarray],
    unit_by_point: dict[str, np.ndarray],
    br_fit: float,
) -> tuple[float, float]:
    """R² y RMSE de las 4 esquinas frente al Br ajustado con el Centro.

    No se usa para ajustar nada: es una metrica de validacion independiente,
    para ver si el modelo geometrico predice bien tambien fuera del eje.
    """
    measured = np.concatenate([measured_by_point[label] for label in CORNER_LABELS])
    unit = np.concatenate([unit_by_point[label] for label in CORNER_LABELS])
    predicted = br_fit * unit
    return _r2_rmse(measured, predicted)


def sweep_corner_radius(
    dimension_mm: tuple[float, float, float],
    distances_mm: np.ndarray,
    measured_corner_mean: np.ndarray,
    br_fit: float,
    radii_mm: np.ndarray,
) -> pd.DataFrame:
    """Barre la posicion (r, r) asumida para las esquinas y evalua el ajuste.

    Sirve para distinguir dos hipotesis frente a un desacuerdo de signo en
    las esquinas: (1) las esquinas estan donde se asumio (fuera de la cara)
    y la discrepancia es del promediado espacial de la sonda cerca del
    borde, o (2) las esquinas estan mas adentro de lo asumido y el modelo
    esta evaluando en el lugar equivocado. Si el mejor ajuste (o el punto
    donde el modelo deja de predecir signo negativo) cae cerca del offset
    nominal (30 mm), la hipotesis 1 es la mas consistente; si cae bien
    adentro, hay que revisar la geometria de medicion.
    """
    rows = []
    for r in radii_mm:
        unit = simulate_bz_unit(dimension_mm, distances_mm, (r, r))
        predicted = br_fit * unit
        _, rmse = _r2_rmse(measured_corner_mean, predicted)
        rows.append(
            {
                "radio_mm": float(r),
                "rmse_mT": rmse,
                "todo_positivo": bool(np.all(predicted > 0)),
                "bz_mas_cercano_mT": float(predicted[0]),
            }
        )
    return pd.DataFrame(rows)


def sweep_z_offset(
    dimension_mm: tuple[float, float, float],
    distances_mm: np.ndarray,
    measured_centro: np.ndarray,
    offsets_mm: np.ndarray,
) -> pd.DataFrame:
    """Barre un offset extra de distancia sobre el Centro y refitea Br.

    Diagnostico para el caso en que Br sale fuera del rango fisico del
    material: si un pequeño error sistematico en la distancia real (por
    ejemplo, un offset de sonda mal calibrado) explica el desvio, entonces
    debería existir un valor de offset que simultaneamente (a) mejora el
    RMSE del ajuste y (b) lleva Br a un valor plausible. Si ningun offset
    razonable hace ambas cosas a la vez, la causa no es un error de
    distancia sino otra cosa (material, dimensiones asumidas, calibracion
    del gaussimetro).
    """
    rows = []
    for extra_z in offsets_mm:
        unit = simulate_bz_unit(dimension_mm, distances_mm + extra_z, (0.0, 0.0))
        br, r2, rmse = fit_remanence_centro(measured_centro, unit)
        rows.append(
            {
                "offset_extra_mm": float(extra_z),
                "br_mT": br,
                "r2": r2,
                "rmse_mT": rmse,
            }
        )
    return pd.DataFrame(rows)


def plot_z_offset_sweep(
    sweep: pd.DataFrame, best_offset: float, title: str, output_path: Path
) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sweep["offset_extra_mm"], sweep["rmse_mT"], color="#0b3954", label="RMSE")
    ax1.set_xlabel("Offset extra de distancia asumido (mm)")
    ax1.set_ylabel("RMSE del Centro (mT)", color="#0b3954")
    ax1.axvline(best_offset, color="crimson", linestyle=":", label=f"Mejor ajuste ({best_offset:.2f} mm)")
    ax1.axvline(0.0, color="gray", linestyle="--", label="Offset asumido (0 mm)")

    ax2 = ax1.twinx()
    ax2.plot(sweep["offset_extra_mm"], sweep["br_mT"], color="#d1495b", alpha=0.6)
    ax2.set_ylabel("Br ajustado (mT)", color="#d1495b")

    ax1.set_title(title)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    guardar_figura(fig, output_path)
    plt.close(fig)


def plot_comparison(
    distances_mm: np.ndarray,
    measured_by_point: dict[str, np.ndarray],
    predicted_by_point: dict[str, np.ndarray],
    br_fit: float,
    centro_r2: float,
    centro_rmse: float,
    corner_r2: float,
    corner_rmse: float,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(measured_by_point)))

    for color, (label, measured) in zip(colors, measured_by_point.items()):
        ax.plot(
            distances_mm,
            measured,
            marker="o",
            linestyle="none",
            color=color,
            label=f"{label} (medido)",
        )
        ax.plot(
            distances_mm,
            predicted_by_point[label],
            linestyle="-",
            color=color,
            alpha=0.8,
            label=f"{label} (simulado)",
        )

    ax.set_xlabel("Distancia a la cara (mm)")
    ax.set_ylabel("Bz (mT)")
    ax.set_title(
        f"{title}\n"
        f"Br ajustado (Centro) = {br_fit:.1f} mT\n"
        f"Centro: R² = {centro_r2:.4f}, RMSE = {centro_rmse:.2f} mT   |   "
        f"Esquinas (validación): R² = {corner_r2:.4f}, RMSE = {corner_rmse:.2f} mT",
        fontsize=11,
    )
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    guardar_figura(fig, output_path)
    plt.close(fig)


def plot_corner_radius_sweep(
    sweep: pd.DataFrame, best_radius: float, sign_flip_radius: float | None, title: str, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep["radio_mm"], sweep["rmse_mT"], color="#0b3954")
    ax.axvline(CORNER_OFFSET_MM, color="gray", linestyle="--", label=f"Asumido ({CORNER_OFFSET_MM:.0f} mm)")
    ax.axvline(best_radius, color="crimson", linestyle=":", label=f"Mejor ajuste ({best_radius:.1f} mm)")
    if sign_flip_radius is not None:
        ax.axvline(
            sign_flip_radius,
            color="darkorange",
            linestyle="-.",
            label=f"Transición de signo ({sign_flip_radius:.1f} mm)",
        )
    ax.set_xlabel("Radio asumido de la esquina, r en (r, r) (mm)")
    ax.set_ylabel("RMSE vs. esquinas medidas (mT)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    guardar_figura(fig, output_path)
    plt.close(fig)


def plot_field_map(
    dimension_mm: tuple[float, float, float],
    br_fit: float,
    measured_by_point: dict[str, np.ndarray],
    distances_mm: np.ndarray,
    x_range_mm: float,
    z_below_margin_mm: float,
    z_above_mm: float,
    resolution: int,
    title: str,
    output_path: Path,
) -> None:
    """Mapa 2D (corte xz por y=0) de Bz, con lineas de campo y puntos medidos.

    Diagnostico de esquinas: el campo de retorno de un iman finito se curva
    y cambia de signo fuera de la cara. Si las esquinas medidas caen en la
    zona donde el modelo predice Bz < 0, un desacuerdo de signo ahi no
    implica que el modelo este mal, sino que la sonda esta midiendo en la
    region de campo de retorno.
    """
    fig, ax = plt.subplots(figsize=(11, 9))
    mesh = dibujar_mapa_campo(
        ax,
        dimension_mm,
        br_fit,
        measured_by_point,
        distances_mm,
        x_range_mm,
        z_below_margin_mm,
        z_above_mm,
        resolution,
    )
    fig.colorbar(mesh, ax=ax, label="Bz (mT)")
    ax.set_title(
        f"{title}\nCorte xz (y=0) de Bz — Br ajustado = {br_fit:.1f} mT, "
        f"contorno negro = Bz = 0",
        fontsize=11,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    guardar_figura(fig, output_path)
    plt.close(fig)


def dibujar_mapa_campo(
    ax,
    dimension_mm: tuple[float, float, float],
    br_fit: float,
    measured_by_point: dict[str, np.ndarray],
    distances_mm: np.ndarray,
    x_range_mm: float,
    z_below_margin_mm: float,
    z_above_mm: float,
    resolution: int,
    leyenda: bool = True,
):
    """Dibuja el mapa de plot_field_map sobre un eje dado y devuelve el mesh.

    Separado para que plot_field_maps_combinados.py arme una figura con los
    dos imanes lado a lado reusando exactamente el mismo dibujo.
    """
    width_mm, _, thickness_mm = dimension_mm
    z_face = thickness_mm / 2

    x = np.linspace(-x_range_mm, x_range_mm, resolution)
    z = np.linspace(-z_face - z_below_margin_mm, z_face + z_above_mm, resolution)
    X, Z = np.meshgrid(x, z)

    cube = magpy.magnet.Cuboid(polarization=(0.0, 0.0, br_fit), dimension=dimension_mm)
    observers = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
    b = cube.getB(observers)
    Bx = b[:, 0].reshape(X.shape)
    Bz = b[:, 2].reshape(X.shape)

    vmax = float(np.nanpercentile(np.abs(Bz), 99))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    mesh = ax.pcolormesh(X, Z, Bz, cmap="seismic", norm=norm, shading="auto")

    ax.contour(X, Z, Bz, levels=[0.0], colors="black", linewidths=2.5)

    ax.streamplot(x, z, Bx, Bz, color="black", linewidth=0.6, density=1.3, arrowsize=0.8)

    ax.add_patch(
        Rectangle(
            (-width_mm / 2, -z_face),
            width_mm,
            thickness_mm,
            facecolor="none",
            edgecolor="lime",
            linewidth=2.0,
            label="Contorno del imán",
        )
    )

    corner_radius_mm = float(np.hypot(CORNER_OFFSET_MM, CORNER_OFFSET_MM))
    point_colors = dict(zip(MEASUREMENT_POINTS, plt.cm.tab10(np.linspace(0, 1, len(MEASUREMENT_POINTS)))))
    jitter_by_label = {label: 0.0 for label in MEASUREMENT_POINTS}
    jitter_step = x_range_mm * 0.01
    for i, label in enumerate(CORNER_LABELS):
        jitter_by_label[label] = (i - (len(CORNER_LABELS) - 1) / 2) * jitter_step

    for label in MEASUREMENT_POINTS:
        x_proj = 0.0 if label == "Centro" else corner_radius_mm
        z_points = z_face + distances_mm
        measured = measured_by_point[label]
        edge_colors = np.where(measured < 0, "red", "black")
        ax.scatter(
            np.full_like(z_points, x_proj) + jitter_by_label[label],
            z_points,
            s=45,
            marker="o" if label == "Centro" else "X",
            color=point_colors[label],
            edgecolor=edge_colors,
            linewidth=1.1,
            label=f"{label} (medido; borde rojo = Bz medido < 0)" if label != "Centro" else label,
            zorder=5,
        )

    ax.axvline(width_mm / 2, color="gray", linestyle=":", linewidth=1.0)
    ax.text(
        width_mm / 2,
        z.max(),
        f"borde cara x={width_mm / 2:.1f} mm",
        rotation=90,
        va="top",
        ha="right",
        fontsize=7,
        color="gray",
    )
    ax.text(
        corner_radius_mm,
        z.max(),
        f"esquinas r={corner_radius_mm:.1f} mm",
        rotation=90,
        va="top",
        ha="right",
        fontsize=7,
        color="dimgray",
    )

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_xlim(-x_range_mm, x_range_mm)
    ax.set_ylim(z.min(), z.max())
    ax.set_aspect("equal")
    if leyenda:
        ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
    return mesh


def run(
    magnet: str,
    csv_path: Path,
    output_path: Path,
    title: str,
    diagnose_corners: bool,
    corner_radius_min: float,
    corner_radius_max: float,
    corner_radius_step: float,
    diagnose_offset: bool,
    offset_min: float,
    offset_max: float,
    offset_step: float,
    diagnose_map: bool,
    map_x_range: float,
    map_z_below_margin: float,
    map_z_above: float,
    map_resolution: int,
) -> None:
    dimension_mm = MAGNET_DIMENSIONS_MM[magnet]
    df = load_measurements(csv_path)
    distances_mm = df["Distancia real"].to_numpy()

    measured_by_point = {
        label: df[column].to_numpy() for label, column in CSV_COLUMN_BY_POINT.items()
    }
    unit_by_point = {
        label: simulate_bz_unit(dimension_mm, distances_mm, xy_offset)
        for label, xy_offset in MEASUREMENT_POINTS.items()
    }

    br_fit, centro_r2, centro_rmse = fit_remanence_centro(
        measured_by_point["Centro"], unit_by_point["Centro"]
    )
    corner_r2, corner_rmse = validate_corners(measured_by_point, unit_by_point, br_fit)
    predicted_by_point = {label: br_fit * unit for label, unit in unit_by_point.items()}

    plot_comparison(
        distances_mm,
        measured_by_point,
        predicted_by_point,
        br_fit,
        centro_r2,
        centro_rmse,
        corner_r2,
        corner_rmse,
        title,
        output_path,
    )

    print(
        f"[OK] Br ajustado (solo Centro): {br_fit:.2f} mT "
        f"(R² = {centro_r2:.4f}, RMSE = {centro_rmse:.2f} mT)"
    )
    print(
        f"[OK] Validación esquinas (no entran al fit): "
        f"R² = {corner_r2:.4f}, RMSE = {corner_rmse:.2f} mT"
    )
    print(f"[OK] Plot guardado en: {output_path}")

    if diagnose_corners:
        measured_corner_mean = np.mean(
            [measured_by_point[label] for label in CORNER_LABELS], axis=0
        )
        radii_mm = np.arange(corner_radius_min, corner_radius_max + 1e-9, corner_radius_step)
        sweep = sweep_corner_radius(
            dimension_mm, distances_mm, measured_corner_mean, br_fit, radii_mm
        )

        best_row = sweep.loc[sweep["rmse_mT"].idxmin()]
        best_radius = float(best_row["radio_mm"])

        all_positive = sweep[sweep["todo_positivo"]]
        sign_flip_radius = (
            float(all_positive["radio_mm"].max()) if not all_positive.empty else None
        )

        sweep_output = output_path.with_name(output_path.stem + "_corner_radius_sweep.png")
        plot_corner_radius_sweep(
            sweep,
            best_radius,
            sign_flip_radius,
            f"{title}\nBarrido de posición asumida de esquina",
            sweep_output,
        )

        print(
            f"[OK] Barrido radial de esquinas: mejor ajuste en r = {best_radius:.1f} mm "
            f"(RMSE = {best_row['rmse_mT']:.2f} mT)"
        )
        if sign_flip_radius is not None:
            print(
                f"[OK] El modelo predice Bz > 0 en todo el barrido de distancias "
                f"para r <= {sign_flip_radius:.1f} mm (offset asumido: {CORNER_OFFSET_MM:.0f} mm)"
            )
        else:
            print(
                "[OK] El modelo predice Bz < 0 en algún punto para todo el rango de r evaluado."
            )
        print(f"[OK] Plot de diagnóstico guardado en: {sweep_output}")

    if diagnose_offset:
        offsets_mm = np.arange(offset_min, offset_max + 1e-9, offset_step)
        sweep = sweep_z_offset(dimension_mm, distances_mm, measured_by_point["Centro"], offsets_mm)
        best_row = sweep.loc[sweep["rmse_mT"].idxmin()]
        best_offset = float(best_row["offset_extra_mm"])

        offset_output = output_path.with_name(output_path.stem + "_z_offset_sweep.png")
        plot_z_offset_sweep(
            sweep,
            best_offset,
            f"{title}\nBarrido de offset extra de distancia (Centro)",
            offset_output,
        )

        print(
            f"[OK] Barrido de offset de distancia: mejor ajuste con "
            f"{best_offset:+.2f} mm extra (RMSE = {best_row['rmse_mT']:.2f} mT, "
            f"Br = {best_row['br_mT']:.1f} mT, R² = {best_row['r2']:.4f})"
        )
        print(f"[OK] Plot de diagnóstico guardado en: {offset_output}")

    if diagnose_map:
        map_output = output_path.with_name(output_path.stem + "_field_map.png")
        plot_field_map(
            dimension_mm,
            br_fit,
            measured_by_point,
            distances_mm,
            map_x_range,
            map_z_below_margin,
            map_z_above,
            map_resolution,
            title,
            map_output,
        )
        print(f"[OK] Mapa de campo (corte xz) guardado en: {map_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simula el campo Bz de un iman permanente (magpylib), ajusta la "
            "remanencia Br al Centro medido y valida contra las esquinas."
        )
    )
    parser.add_argument(
        "--magnet",
        choices=sorted(MAGNET_DIMENSIONS_MM),
        default=None,
        help=(
            "Tipo de iman a simular. Si se omite junto con --csv, corre los "
            "3 imanes caracterizados (ceramica N, ceramica S, neodimio N)."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "CSV con las mediciones (Distancia real, Centro, Esq A-D en mT). "
            "Si se omite, corre los 3 imanes caracterizados por defecto."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta del PNG de salida (solo aplica si se especifica --csv).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Titulo de la figura (solo aplica si se especifica --csv).",
    )
    parser.add_argument(
        "--diagnose-corners",
        action="store_true",
        help=(
            "Ademas del ajuste, barre la posicion (r, r) asumida de las "
            "esquinas para distinguir si la discrepancia de signo viene de "
            "una geometria de medicion distinta a la asumida o del "
            "promediado espacial de la sonda cerca del borde."
        ),
    )
    parser.add_argument("--corner-radius-min", type=float, default=15.0)
    parser.add_argument("--corner-radius-max", type=float, default=45.0)
    parser.add_argument("--corner-radius-step", type=float, default=0.5)
    parser.add_argument(
        "--diagnose-offset",
        action="store_true",
        help=(
            "Barre un offset extra de distancia sobre el Centro y refitea Br, "
            "para chequear si un error de calibracion de distancia (no el "
            "material) explica un Br fuera de rango fisico."
        ),
    )
    parser.add_argument("--offset-min", type=float, default=-3.0)
    parser.add_argument("--offset-max", type=float, default=15.0)
    parser.add_argument("--offset-step", type=float, default=0.25)
    parser.add_argument(
        "--diagnose-map",
        action="store_true",
        help=(
            "Genera un mapa 2D del corte xz (y=0) de Bz, con contorno Bz=0, "
            "lineas de campo y los puntos de medicion superpuestos, para ver "
            "en que region cae cada punto (en particular las esquinas)."
        ),
    )
    parser.add_argument(
        "--map-x-range",
        type=float,
        default=100.0,
        help="Semirrango en x del mapa, en mm (por defecto x en [-100, 100]).",
    )
    parser.add_argument(
        "--map-z-below-margin",
        type=float,
        default=20.0,
        help="Margen del mapa por debajo de la cara trasera del iman, en mm.",
    )
    parser.add_argument(
        "--map-z-above",
        type=float,
        default=100.0,
        help="Extension del mapa por encima de la cara del iman, en mm.",
    )
    parser.add_argument("--map-resolution", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.csv is not None:
        jobs = [
            {
                "magnet": args.magnet or "ceramica",
                "csv": args.csv,
                "output": args.output or Path("data/extra/plots/bz_fit.png"),
                "title": args.title or "Bz medido vs. simulado",
            }
        ]
    else:
        jobs = DEFAULT_JOBS

    for job in jobs:
        print(f"=== {job['title']} ===")
        run(
            job["magnet"],
            job["csv"],
            job["output"],
            job["title"],
            args.diagnose_corners,
            args.corner_radius_min,
            args.corner_radius_max,
            args.corner_radius_step,
            args.diagnose_offset,
            args.offset_min,
            args.offset_max,
            args.offset_step,
            args.diagnose_map,
            args.map_x_range,
            args.map_z_below_margin,
            args.map_z_above,
            args.map_resolution,
        )


if __name__ == "__main__":
    main()
