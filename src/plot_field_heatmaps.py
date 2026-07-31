"""Mapas transversales de |B| en el plano de las semillas, por nivel de tratamiento.

Para cada nivel de la torre de exposicion dibuja un corte horizontal (plano xy)
a la altura del centro de la capa de semillas, con la huella del cajon y sus
cuatro compartimentos superpuestos.

La figura tiene que responder dos preguntas a la vez --cuanto campo hay y cuan
homogeneo es dentro de una bandeja-- y para eso usa los dos canales visuales por
separado: el color codifica |B| absoluto en una escala compartida por los 16
paneles, y las curvas de nivel codifican el cociente respecto del valor sobre el
eje del propio nivel. Asi la intensidad se compara entre paneles y la estructura
interna sigue siendo legible incluso donde el color queda comprimido.

Salidas en data/extra/plots/heatmaps/: 16 mapas individuales (8 por iman) y dos
grillas 4x2, una por iman, en PNG y PDF.

Uso:
    python3 src/plot_field_heatmaps.py --magnet neodimio --level 1
    python3 src/plot_field_heatmaps.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from compute_treatment_doses import (
    CAJON_FOOTPRINT_MM,
    COMPARTMENT_MM,
    MAGNETS,
    N_LEVELS,
    bmod,
    seed_layer_z,
)

# Se usan los dos canales visuales para no tener que elegir entre comparar
# intensidades y ver la estructura interna:
#
#   COLOR  = |B| absoluto, en escala logaritmica compartida por los 16 paneles.
#            Un mismo color significa el mismo campo en cualquier panel, y la
#            escala coincide con la del esquema de la torre.
#   CURVAS = cociente respecto del valor sobre el eje del propio nivel. La
#            homogeneidad se lee de las lineas, de modo que sigue siendo legible
#            en los niveles altos, donde el color queda comprimido.
#
# Una primera version normalizaba el color al centro de cada panel. Se descarto:
# volvia los paneles incomparables en intensidad, y como el rojo se lee siempre
# como "intenso", ceramica nivel 1 (76 mT) aparentaba mas campo que neodimio
# nivel 1 (253 mT).
CMAP = "plasma"
# Lista deliberadamente rala: con la ventana ampliada, un paso de 0.10 generaba
# hasta once curvas por panel y las etiquetas se amontonaban. Se conserva mas
# detalle cerca de 1, que es la zona que ocupa el cajon.
CONTOUR_CANDIDATES = [0.30, 0.50, 0.70, 0.85, 0.95, 1.10, 1.25]
HALF = CAJON_FOOTPRINT_MM / 2.0
WALL_OUT = (CAJON_FOOTPRINT_MM - 2 * COMPARTMENT_MM) / 4.0
# Semirrango del mapa. Se elige para que entre la cara del iman mas grande (el
# ceramico, 100x80 mm, o sea +-50 x +-40), de modo que ambas figuras puedan
# mostrar la silueta del iman a una misma escala espacial. Es justamente la
# comparacion que interesa: la cara del ceramico contiene al cajon, la del
# neodimio queda dentro de el.
EXTENT = 54.0
RES = 300
OUTDIR = Path("data/extra/plots/heatmaps")


def draw_map(ax, magnet_key: str, level: int, norm, annotate_axes: bool = True,
             face_note: bool = True):
    dim, br = MAGNETS[magnet_key]["dim"], MAGNETS[magnet_key]["br"]
    _, z_mid, _ = seed_layer_z(level)

    g = np.linspace(-EXTENT, EXTENT, RES)
    X, Y = np.meshgrid(g, g)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_mid)])
    B = bmod(dim, br, pts).reshape(X.shape)
    b_center = float(bmod(dim, br, [[0.0, 0.0, z_mid]])[0])
    R = B / b_center

    # Color: campo absoluto, escala compartida.
    #
    # rasterized=True es importante: sin eso el PDF guarda las 260x260 celdas
    # como cuadrilateros vectoriales y cada grilla pesa ~10 MB. Rasterizando solo
    # la malla, el texto y las curvas de nivel siguen siendo vectoriales y el
    # archivo baja dos ordenes de magnitud.
    mesh = ax.pcolormesh(X, Y, B, cmap=CMAP, norm=norm, shading="auto",
                         rasterized=True)

    # Solo las curvas que caen dentro del rango de este panel, para no rotular
    # niveles inexistentes. Las etiquetas llevan fondo blanco: sin el se pierden
    # sobre las zonas oscuras del mapa y sobre las lineas del cajon.
    lo, hi = R.min(), R.max()
    levels = [c for c in CONTOUR_CANDIDATES if lo + 0.015 < c < hi - 0.005]
    if levels:
        cs = ax.contour(X, Y, R, levels=levels, colors="white",
                        linewidths=0.9, alpha=0.95)
        labels = ax.clabel(cs, inline=True, fontsize=6.5, fmt="%.2f",
                           inline_spacing=6, colors="black")
        for t in labels:
            t.set_bbox(dict(boxstyle="round,pad=0.12", facecolor="white",
                            edgecolor="none", alpha=0.9))

    # Cara del iman: solo se dibuja si entra en el area graficada. En el ceramico
    # (100x80 mm) queda por fuera, y en ese caso se deja constancia con una nota
    # en lugar de una linea invisible con leyenda enganosa.
    mag_w, mag_d, _ = dim
    face_visible = mag_w / 2 <= EXTENT and mag_d / 2 <= EXTENT
    if face_visible:
        ax.add_patch(Rectangle((-mag_w / 2, -mag_d / 2), mag_w, mag_d, fill=False,
                               edgecolor="#1d9e75", linewidth=2.0,
                               label="cara del imán"))
    elif face_note:
        # En la grilla la nota se pone una sola vez a nivel de figura: repetirla
        # en los ocho paneles la hace chocar con las etiquetas de nivel.
        ax.text(0.5, 0.015,
                f"la cara del imán ({mag_w:.0f}$\\times${mag_d:.0f} mm) excede el área mostrada",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=5.5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.85))

    # cajon y compartimentos
    ax.add_patch(Rectangle((-HALF, -HALF), CAJON_FOOTPRINT_MM, CAJON_FOOTPRINT_MM,
                           fill=False, edgecolor="black", linewidth=1.6))
    for sx in (-1, 1):
        for sy in (-1, 1):
            x0 = sx * (WALL_OUT + COMPARTMENT_MM) if sx < 0 else sx * WALL_OUT
            y0 = sy * (WALL_OUT + COMPARTMENT_MM) if sy < 0 else sy * WALL_OUT
            ax.add_patch(Rectangle((x0, y0), COMPARTMENT_MM, COMPARTMENT_MM,
                                   fill=False, edgecolor="black", linewidth=1.0,
                                   linestyle="--"))

    m = np.abs(g) <= HALF
    inside = R[np.ix_(m, m)]
    # Titulo en una sola linea: en la grilla, dos lineas por panel consumen alto
    # que se traduce directamente en paneles mas chicos al escalar a la pagina.
    ax.set_title(f"Nivel {level}  ·  {b_center:.1f} mT  ·  "
                 f"{inside.min():.2f}–{inside.max():.2f}", fontsize=8.5)
    ax.set_aspect("equal")
    ax.set_xlim(-EXTENT, EXTENT)
    ax.set_ylim(-EXTENT, EXTENT)
    ax.tick_params(labelsize=7)
    if annotate_axes:
        ax.set_xlabel("x (mm)", fontsize=8)
        ax.set_ylabel("y (mm)", fontsize=8)
    return mesh, R


def shared_norm() -> LogNorm:
    """Escala de color comun a los 16 paneles, sobre |B| dentro del cajon."""
    g = np.linspace(-HALF, HALF, 81)
    X, Y = np.meshgrid(g, g)
    lo, hi = np.inf, 0.0
    for m in MAGNETS.values():
        for lvl in range(1, N_LEVELS + 1):
            _, z, _ = seed_layer_z(lvl)
            B = bmod(m["dim"], m["br"],
                     np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)]))
            lo, hi = min(lo, B.min()), max(hi, B.max())
    return LogNorm(vmin=lo, vmax=hi)


def single(magnet_key: str, level: int, norm) -> None:
    fig, ax = plt.subplots(figsize=(4.9, 4.6))
    mesh, R = draw_map(ax, magnet_key, level, norm)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$|\vec{B}|$ (mT)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    # Solo si hay algo que rotular: en el ceramico la cara no entra en el area
    # graficada y no se dibuja, de modo que una leyenda seria enganosa.
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=5, loc="lower right", framealpha=0.92,
                  handlelength=1.2, borderpad=0.3, handletextpad=0.4)

    slug = magnet_key.lower().replace(" ", "_")
    out = OUTDIR / f"heatmap_{slug}_nivel{level}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    m = np.abs(np.linspace(-EXTENT, EXTENT, RES)) <= HALF
    inside = R[np.ix_(m, m)]
    print(f"[OK] {out.name:36s} cociente en el cajón: "
          f"{inside.min():.3f} – {inside.max():.3f}")


def grid(magnet_key: str, norm) -> None:
    """Los 8 niveles de un iman en una grilla 4x2, con barra de color comun."""
    # Proporcion deliberadamente menos esbelta que la natural de 2 columnas x 4
    # filas: al escalar la figura a la altura de la pagina, todo alto sobrante se
    # paga en tamano de panel. Con titulos de una linea y filas mas juntas, la
    # misma pagina admite paneles ~30% mayores.
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 11.6),
                             gridspec_kw={"hspace": 0.16, "wspace": 0.10})
    for i, lvl in enumerate(range(1, N_LEVELS + 1)):
        row, col = i // 2, i % 2
        ax = axes[row][col]
        mesh, _ = draw_map(ax, magnet_key, lvl, norm, annotate_axes=False,
                           face_note=False)
        # Se suprimen las etiquetas de los ticks interiores: si no, los numeros
        # del eje x de una fila se superponen con el titulo de la fila siguiente.
        last_row, first_col = row == 3, col == 0
        ax.tick_params(labelbottom=last_row, labelleft=first_col)
        if last_row:
            ax.set_xlabel("x (mm)", fontsize=8)
        if first_col:
            ax.set_ylabel("y (mm)", fontsize=8)

    cb = fig.colorbar(mesh, ax=axes, fraction=0.022, pad=0.02, shrink=0.6)
    cb.set_label(r"$|\vec{B}|$ (mT)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # La aclaracion de que la cara del iman excede el area mostrada va en el
    # epigrafe del documento, no dentro de la figura: cualquier posicion aqui
    # choca con los rotulos de los ejes de la ultima fila.

    slug = magnet_key.lower().replace(" ", "_")
    out = OUTDIR / f"grilla_heatmaps_{slug}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--magnet", default="neodimio")
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--all", action="store_true",
                   help="genera los 16 mapas individuales y las dos grillas 4x2")
    a = p.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    norm = shared_norm()
    print(f"escala de color compartida: {norm.vmin:.2f} – {norm.vmax:.2f} mT\n")

    if a.all:
        for key in MAGNETS:
            for lvl in range(1, N_LEVELS + 1):
                single(key, lvl, norm)
            grid(key, norm)
    else:
        key = next(k for k in MAGNETS if k.lower().startswith(a.magnet.lower()))
        single(key, a.level, norm)


if __name__ == "__main__":
    main()
