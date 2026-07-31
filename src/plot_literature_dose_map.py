"""Plano intensidad x tiempo de exposicion: literatura en Zea mays vs. este trabajo.

Cada estudio de maiz se ubica en el punto (intensidad, tiempo) de su tratamiento
optimo reportado, con el signo del efecto codificado por color y forma. Las
condiciones de este trabajo se superponen como una transecta vertical, dado que
la exposicion es fija en 1 h y solo varia la intensidad segun el nivel.

Criterios de armado:
  - Solo Zea mays. Los estudios en otras especies y los de campo debil (marco
    teorico) quedan afuera.
  - Un punto por optimo reportado, no el rango barrido. Cuando un estudio
    informa dos optimos (Anand 2012) se incluyen ambos.
  - El efecto se codifica solo por direccion. Las magnitudes no son comparables
    entre estudios: unos reportan germinacion en %, otros peso fresco,
    rendimiento de grano, area foliar o PIABS.
  - Coordenadas repetidas se agrupan en un marcador cuyo area es proporcional al
    numero de estudios.

Fuente de los datos: resumenes/ResumenTotal.md (tablas 2, 3.1 y 3.2) y los
resumenes individuales. La columna `fuente` de cada entrada permite rastrearlo.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from compute_treatment_doses import MAGNETS, N_LEVELS, bmod, seed_layer_z

EXPOSICION_PROPIA_MIN = 60.0

# (etiqueta, mT, minutos, direccion)
#   direccion: "+" estimula, "-" inhibe
LITERATURA = [
    ("Torres 2019",            50.0,     1.0, "+"),
    ("Aladjadjiyan 2002",     150.0,    10.0, "+"),
    ("Siyami 2018",            40.0,    10.0, "+"),
    ("Torres 2018",           200.0,    10.0, "+"),
    ("Zepeda-Bautista 2014",  480.0,    12.0, "+"),
    ("Domínguez-Pacheco 2010",560.0,    30.0, "+"),
    ("Vashisth 2010",         200.0,    60.0, "+"),
    ("Shine 2012",            200.0,    60.0, "+"),
    ("Kataria 2017",          200.0,    60.0, "+"),
    ("Shine 2017",            200.0,    60.0, "+"),
    ("Vashisth 2017",         200.0,    60.0, "+"),
    ("Anand 2012",            200.0,    60.0, "+"),
    ("Anand 2012",            100.0,   120.0, "+"),
    ("Vashisth 2009",         100.0,   120.0, "+"),
    ("Sharma 2017",           100.0,   120.0, "+"),
    ("Flórez 2007",           125.0,  1440.0, "+"),
    ("Martínez 2017",         125.0,  1440.0, "+"),
    ("Răcuciu 2006",          250.0, 15840.0, "+"),
    ("Torres 2019",           250.0,     7.0, "-"),   # >150 mT: régimen inhibitorio
]


def agrupar(direccion: str):
    """Agrupa entradas con la misma coordenada; devuelve (x, y, n, etiquetas)."""
    d: dict[tuple[float, float], list[str]] = {}
    for lab, mt, mi, sig in LITERATURA:
        if sig == direccion:
            d.setdefault((mt, mi), []).append(lab)
    xs = [k[0] for k in d]
    ys = [k[1] for k in d]
    ns = [len(v) for v in d.values()]
    labs = [sorted(set(v)) for v in d.values()]
    return xs, ys, ns, labs


def niveles_propios(magnet_key: str) -> list[float]:
    dim, br = MAGNETS[magnet_key]["dim"], MAGNETS[magnet_key]["br"]
    out = []
    for lvl in range(1, N_LEVELS + 1):
        _, z, _ = seed_layer_z(lvl)
        out.append(float(bmod(dim, br, [[0.0, 0.0, z]])[0]))
    return out


# Desplazamientos propios para las etiquetas que se solapan.
OFFSETS = {"Aladjadjiyan 2002": (-6, -11), "Torres 2018": (7, 4),
           "6 estudios": (10, -13)}

# Intensidad minima explorada por la literatura en maiz (Siyami 2018, 40 mT).
LIT_MIN_MT = 40.0


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    # --- zona sin cobertura de la literatura ---
    ax.axvspan(1, LIT_MIN_MT, color="#d1495b", alpha=0.07, zorder=0)
    ax.text(np.sqrt(9 * LIT_MIN_MT), 0.62, "sin estudios\nen maíz", ha="center",
            va="bottom", fontsize=7, color="#993c1d", style="italic", zorder=2)

    # --- banda de la exposición propia ---
    ax.axhline(EXPOSICION_PROPIA_MIN, color="#0b3954", lw=1.0, ls="--", zorder=1)
    ax.text(0.985, EXPOSICION_PROPIA_MIN * 3.2, "exposición de este trabajo: 1 h",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=7.5, color="#0b3954")

    # --- literatura ---
    for sig, color, marker, lab in [("+", "#1d9e75", "o", "estimula"),
                                    ("-", "#d1495b", "X", "inhibe")]:
        xs, ys, ns, labs = agrupar(sig)
        ax.scatter(xs, ys, s=[36 + 40 * (n - 1) for n in ns], c=color,
                   marker=marker, edgecolor="black", linewidth=0.6, zorder=4,
                   label=f"{lab} (área ∝ n.º de estudios)")
        for x, y, n, ls in zip(xs, ys, ns, labs):
            txt = ls[0] if n == 1 else f"{n} estudios"
            off = OFFSETS.get(txt, (7, 4))
            ax.annotate(txt, (x, y), textcoords="offset points", xytext=off,
                        fontsize=6.2, color="#333333", zorder=5,
                        ha="right" if off[0] < 0 else "left")

    # --- condiciones propias ---
    for key, color, marker in [("Ceramica N", "#7f77dd", "s"),
                               ("Neodimio N", "#ba7517", "D")]:
        b = niveles_propios(key)
        nombre = "Cerámica" if "Cera" in key else "Neodimio"
        ax.scatter(b, [EXPOSICION_PROPIA_MIN] * len(b), s=32, c=color,
                   marker=marker, edgecolor="black", linewidth=0.6, zorder=6,
                   label=f"este trabajo · {nombre}: "
                         f"8 niveles, {min(b):.0f}–{max(b):.0f} mT")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Intensidad del campo aplicado (mT)", fontsize=9)
    ax.set_ylabel("Tiempo de exposición (min)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(9, 900)
    ax.set_ylim(0.5, 40000)
    ax.set_xticks([10, 20, 40, 100, 200, 400, 800])
    ax.set_xticklabels(["10", "20", "40", "100", "200", "400", "800"])
    ax.set_yticks([1, 10, 60, 600, 1440, 15840])
    ax.set_yticklabels(["1 min", "10 min", "1 h", "10 h", "24 h", "11 d"])
    ax.grid(alpha=0.25, which="major", lw=0.5)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.95)
    fig.tight_layout()

    out = Path("data/extra/plots/mapa_dosis_literatura.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[OK] {out}")
    print(f"     {len(LITERATURA)} entradas, "
          f"{len({(m, t) for _, m, t, _ in LITERATURA})} coordenadas distintas")
    for key in MAGNETS:
        b = niveles_propios(key)
        print(f"     {key}: {min(b):.1f}–{max(b):.1f} mT a 60 min")


if __name__ == "__main__":
    main()
