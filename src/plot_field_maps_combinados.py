"""
Figura de la tesis con los mapas de campo de los dos imanes lado a lado:
(a) cerámica N y (b) neodimio N, corte xz (y=0) de Bz.

Reemplaza a las dos figuras sueltas *_bz_fit_field_map que genera
simulate_magnet_field.py --diagnose-map, a pedido del director: una sola
figura con paneles a) y b) y barras de color más chicas. El dibujo de cada
panel es exactamente el de simulate_magnet_field.dibujar_mapa_campo, con la
misma remanencia ajustada sobre el Centro.

Uso (desde la raíz del repo):
    .venv/bin/python src/plot_field_maps_combinados.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulate_magnet_field import (
    CSV_COLUMN_BY_POINT,
    MAGNET_DIMENSIONS_MM,
    MEASUREMENT_POINTS,
    dibujar_mapa_campo,
    fit_remanence_centro,
    guardar_figura,
    load_measurements,
    simulate_bz_unit,
)

PANELES = [
    ("ceramica", Path("data/extra/Caracterización imanes - Ceramica N.csv"), "(a) Cerámica N"),
    ("neodimio", Path("data/extra/Caracterización imanes - Neodimio N.csv"), "(b) Neodimio N"),
]
SALIDA = Path("data/extra/plots/field_maps_ceramica_neodimio_n.png")

# Mismos valores por defecto que simulate_magnet_field.py --diagnose-map.
X_RANGE_MM = 100.0
Z_BELOW_MARGIN_MM = 20.0
Z_ABOVE_MM = 100.0
RESOLUCION = 220


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    for ax, (magnet, csv_path, titulo) in zip(axes, PANELES):
        dimension_mm = MAGNET_DIMENSIONS_MM[magnet]
        df = load_measurements(csv_path)
        distances_mm = df["Distancia real"].to_numpy()
        measured_by_point = {
            label: df[column].to_numpy() for label, column in CSV_COLUMN_BY_POINT.items()
        }
        unit_centro = simulate_bz_unit(
            dimension_mm, distances_mm, MEASUREMENT_POINTS["Centro"]
        )
        br_fit, _, _ = fit_remanence_centro(measured_by_point["Centro"], unit_centro)

        mesh = dibujar_mapa_campo(
            ax,
            dimension_mm,
            br_fit,
            measured_by_point,
            distances_mm,
            X_RANGE_MM,
            Z_BELOW_MARGIN_MM,
            Z_ABOVE_MM,
            RESOLUCION,
            leyenda=False,
        )
        fig.colorbar(mesh, ax=ax, label="$B_z$ (mT)", shrink=0.6, aspect=25, pad=0.02)
        ax.set_title(titulo, fontsize=12)
        print(f"[OK] {titulo}: Br ajustado = {br_fit:.1f} mT")

    # Una sola leyenda abajo, común a los dos paneles, para no tapar el mapa.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    guardar_figura(fig, SALIDA, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figura guardada en: {SALIDA.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
