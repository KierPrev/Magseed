"""Campo magnetico efectivo en cada nivel de la torre de exposicion.

Evalua el modelo de magpylib ya validado en simulate_magnet_field.py sobre la
posicion real que ocupan las semillas dentro de la torre, para construir la
tabla de dosis por nivel de tratamiento.

Geometria (medida por Kier, 29/07/26):
  - El iman se apoya en la base de la torre, cara mayor hacia arriba.
  - Entre la cara del iman y la base del primer cajon hay 2.0 mm.
  - Los cajones se apilan; cada uno agrega 9.8 mm de altura.
  - El piso del cajon tiene 0.5 mm de espesor: las semillas apoyan ahi.
  - El grano acostado mide entre 4.9 y 6.4 mm de alto.

El grosor del papel absorbente que va debajo de las semillas se desprecia:
es de decimas de mm, un orden por debajo de la dispersion de altura de grano
(1.5 mm), que ya se propaga al resultado como rango.
"""

from __future__ import annotations

import numpy as np
import magpylib as magpy
import pandas as pd

# --- Geometria de la torre (mm) ---
BASE_GAP_MM = 2.0            # cara del iman -> base del primer cajon
CAJON_PITCH_MM = 9.8         # altura externa del cajon = paso de apilado
CAJON_FLOOR_MM = 0.5         # espesor del piso del cajon
GRAIN_H_MIN_MM = 4.9         # alto del grano acostado
GRAIN_H_MAX_MM = 6.4
CAJON_FOOTPRINT_MM = 65.56   # huella del cajon (cuadrado)
COMPARTMENT_MM = 29.78       # lado de cada compartimento (2x2 dentro de la huella)
N_LEVELS = 8                 # niveles expuestos; el 9no es control fuera de la torre

# --- Imanes: dimensiones y Br ajustado en simulate_magnet_field.py ---
MAGNETS = {
    "Ceramica N": {"dim": (100.0, 80.0, 30.5), "br": 318.95},
    "Neodimio N": {"dim": (50.8, 50.8, 25.3), "br": 926.28},
}


def bvec(dim_mm, br_mt, points_mm):
    """Vector B (mT) en los puntos dados, con z medido desde la cara del iman."""
    cube = magpy.magnet.Cuboid(polarization=(0.0, 0.0, br_mt), dimension=dim_mm)
    pts = np.atleast_2d(np.asarray(points_mm, dtype=float)).copy()
    pts[:, 2] += dim_mm[2] / 2.0          # z=0 en la cara -> coordenada del cubo
    # magpylib devuelve (3,) cuando se le pasa un unico observador; se fuerza (N,3)
    return np.atleast_2d(cube.getB(pts))


def bz(dim_mm, br_mt, points_mm):
    """Bz (mT): componente normal, que es lo que mide la sonda Hall."""
    return bvec(dim_mm, br_mt, points_mm)[:, 2]


def bmod(dim_mm, br_mt, points_mm):
    """|B| (mT): modulo del campo.

    Es la magnitud fisicamente relevante para una muestra sin orientacion
    privilegiada: las semillas quedan dispuestas al azar dentro del
    compartimento, de modo que ninguna componente del sistema de laboratorio
    tiene significado para ellas. Ademas, en la region de campo de retorno
    (fuera de la cara del iman) Bz resulta de la cancelacion parcial de
    contribuciones y es mal condicionado frente a errores de posicion,
    mientras que |B| varia de forma mucho mas suave.
    """
    return np.linalg.norm(bvec(dim_mm, br_mt, points_mm), axis=1)


def seed_layer_z(level: int) -> tuple[float, float, float]:
    """(base, centro, tope) de la capa de semillas del nivel dado, en mm."""
    base = BASE_GAP_MM + (level - 1) * CAJON_PITCH_MM + CAJON_FLOOR_MM
    grain = (GRAIN_H_MIN_MM + GRAIN_H_MAX_MM) / 2.0
    return base, base + grain / 2.0, base + grain


def build_table(name: str) -> pd.DataFrame:
    dim, br = MAGNETS[name]["dim"], MAGNETS[name]["br"]

    # Las semillas solo pueden estar dentro de los compartimentos, no sobre las
    # paredes. Con 2 compartimentos de 29.78 mm en una huella de 65.56 mm, el
    # espesor total de pared es 6.0 mm; asumiendo pared exterior de 1.5 mm, el
    # compartimento va de 1.5 a 31.28 mm en cada semieje.
    wall_out = (CAJON_FOOTPRINT_MM - 2 * COMPARTMENT_MM) / 4.0
    lo, hi = wall_out, wall_out + COMPARTMENT_MM
    s = np.linspace(lo, hi, 21)
    coords = np.concatenate([-s[::-1], s])          # los cuatro compartimentos
    gx, gy = np.meshgrid(coords, coords)
    flat = np.column_stack([gx.ravel(), gy.ravel()])

    rows = []
    for lvl in range(1, N_LEVELS + 1):
        z_base, z_mid, z_top = seed_layer_z(lvl)

        axis_pts = [[0, 0, z_base], [0, 0, z_mid], [0, 0, z_top]]
        b_base, b_mid, b_top = bz(dim, br, axis_pts)
        m_base, m_mid, m_top = bmod(dim, br, axis_pts)

        # gradiente vertical en el centro de la capa (diferencias centradas)
        h = 0.05
        grad_pts = [[0, 0, z_mid - h], [0, 0, z_mid + h]]
        b_lo, b_hi = bz(dim, br, grad_pts)
        m_lo, m_hi = bmod(dim, br, grad_pts)
        grad = (b_hi - b_lo) / (2 * h)
        grad_mod = (m_hi - m_lo) / (2 * h)

        # dispersion lateral a la altura media de la capa
        lat_pts = np.column_stack([flat[:, 0], flat[:, 1], np.full(len(flat), z_mid)])
        lat = bz(dim, br, lat_pts)
        lat_m = bmod(dim, br, lat_pts)

        rows.append({
            "nivel": lvl,
            "z_base_mm": z_base,
            "z_centro_mm": z_mid,
            "z_tope_mm": z_top,
            "Bz_centro_mT": b_mid,
            "Bz_base_mT": b_base,
            "Bz_tope_mT": b_top,
            "grad_Bz_mT_por_mm": grad,
            "Bz_lat_min_mT": lat.min(),
            "Bz_lat_max_mT": lat.max(),
            "Bmod_centro_mT": m_mid,
            "Bmod_base_mT": m_base,
            "Bmod_tope_mT": m_top,
            "grad_Bmod_mT_por_mm": grad_mod,
            "Bmod_lat_min_mT": lat_m.min(),
            "Bmod_lat_max_mT": lat_m.max(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    for name in MAGNETS:
        df = build_table(name)
        print(f"\n=== {name} (Br = {MAGNETS[name]['br']:.1f} mT) ===")
        print(df.round(2).to_string(index=False))
        out = f"data/extra/dosis_{name.lower().replace(' ', '_')}.csv"
        df.to_csv(out, index=False)
        print(f"[OK] guardado en {out}")

    half = CAJON_FOOTPRINT_MM / 2.0
    print(f"\nSemiancho de la huella del cajon: {half:.2f} mm")
    for name, m in MAGNETS.items():
        print(f"  {name}: semiancho del iman = {m['dim'][0] / 2:.2f} x {m['dim'][1] / 2:.2f} mm")


if __name__ == "__main__":
    main()
