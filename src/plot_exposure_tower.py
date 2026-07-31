"""Esquema a escala de la torre de exposicion con el campo de cada nivel.

Dibuja, para cada iman, un corte lateral del montaje: el iman en la base y los
ocho cajones apilados encima, todo en escala real (mm). Cada nivel se anota con
el campo sobre el eje --donde Bz = |B|-- y con el intervalo que abarca |B| a lo
largo de la extension horizontal de los compartimentos.

Dibujarlo a escala es deliberado: hace evidente que la huella del cajon
(65.56 mm) excede la cara del iman de neodimio (50.8 mm) pero no la del
ceramico (100 mm), que es el origen geometrico de la diferencia de
homogeneidad entre ambos.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from compute_treatment_doses import (
    BASE_GAP_MM,
    CAJON_FLOOR_MM,
    CAJON_FOOTPRINT_MM,
    CAJON_PITCH_MM,
    GRAIN_H_MAX_MM,
    GRAIN_H_MIN_MM,
    MAGNETS,
    N_LEVELS,
)

GRAIN_MM = (GRAIN_H_MIN_MM + GRAIN_H_MAX_MM) / 2.0
CMAP = plt.cm.plasma


def draw_panel(ax, name: str, df: pd.DataFrame, norm) -> None:
    dim, br = MAGNETS[name]["dim"], MAGNETS[name]["br"]
    mag_w, _, mag_t = dim
    cajon_half = CAJON_FOOTPRINT_MM / 2.0

    # --- iman ---
    ax.add_patch(Rectangle((-mag_w / 2, -mag_t), mag_w, mag_t,
                           facecolor="#c8c8c8", edgecolor="black", linewidth=1.4))
    ax.text(0, -mag_t / 2, f"imán\n{mag_w:.1f} mm", ha="center", va="center", fontsize=7.5)

    # --- cajones ---
    for _, r in df.iterrows():
        lvl = int(r["nivel"])
        z0 = BASE_GAP_MM + (lvl - 1) * CAJON_PITCH_MM
        color = CMAP(norm(r["Bmod_centro_mT"]))

        # cuerpo del cajon
        ax.add_patch(Rectangle((-cajon_half, z0), CAJON_FOOTPRINT_MM, CAJON_PITCH_MM,
                               facecolor="white", edgecolor="black", linewidth=0.9))
        # capa de semillas
        ax.add_patch(Rectangle((-cajon_half, z0 + CAJON_FLOOR_MM), CAJON_FOOTPRINT_MM, GRAIN_MM,
                               facecolor=color, edgecolor="none", alpha=0.95))

        zc = z0 + CAJON_PITCH_MM / 2
        ax.text(-cajon_half - 2.5, zc, f"{lvl}", ha="right", va="center",
                fontsize=8, fontweight="bold")
        ax.text(cajon_half + 3, zc,
                f"{r['Bmod_centro_mT']:.1f}  ({r['Bmod_lat_min_mT']:.0f}–{r['Bmod_lat_max_mT']:.0f})",
                ha="left", va="center", fontsize=7.5)

    # --- borde del cajon ---
    top = BASE_GAP_MM + N_LEVELS * CAJON_PITCH_MM
    for x in (-cajon_half, cajon_half):
        ax.plot([x, x], [0, top + 4], ls="--", lw=0.8, color="#0b3954", zorder=0)

    ax.text(cajon_half + 3, top + 1.5, r"$|\vec{B}|$ (mT)", ha="left", va="bottom",
            fontsize=7, style="italic", color="#333333")

    display = {"Ceramica N": "Cerámica, polo norte", "Neodimio N": "Neodimio, polo norte"}
    ax.set_title(f"{display.get(name, name)}   ($B_r$ = {br:.0f} mT)", fontsize=8.5, pad=6)
    ax.set_xlabel("x (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    ax.set_xlim(-56, 92)
    ax.set_ylim(-mag_t - 4, top + 9)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    dfs = {n: pd.read_csv(f"data/extra/dosis_{n.lower().replace(' ', '_')}.csv")
           for n in MAGNETS}
    allv = np.concatenate([d["Bmod_centro_mT"].values for d in dfs.values()])
    norm = LogNorm(vmin=allv.min(), vmax=allv.max())

    # Se dibuja al tamaño en que sera impresa (ancho de linea de la tesis), de
    # modo que LaTeX no la reduzca y los cuerpos tipograficos se mantengan
    # legibles. El titulo y la explicacion van en el epigrafe del documento.
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.5), sharey=True)
    for ax, (name, df) in zip(axes, dfs.items()):
        draw_panel(ax, name, df, norm)
    axes[0].set_ylabel("Distancia a la cara del imán (mm)", fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.012, shrink=0.95)
    cb.set_label(r"$|\vec{B}|$ sobre el eje (mT)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    out = Path("data/extra/plots/torre_exposicion_campos.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[OK] {out}")
    print(f"[OK] {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
