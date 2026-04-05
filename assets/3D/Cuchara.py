from pathlib import Path
from math import pi, sqrt
from build123d import *
from ocp_vscode import show, set_port

set_port(3939)

# =========================================================
# CALIBRACIÓN POR CANTIDAD DE SEMILLAS
# =========================================================
SEMILLAS_ACTUALES = 34
SEMILLAS_OBJETIVO = 20

# Ajuste manual fino posterior:
# 1.00 = dejar igual
# 0.97 = 3% menos
# 1.03 = 3% más
FACTOR_AJUSTE_MANUAL = 0.9

# =========================================================
# VOLUMEN BASE
# =========================================================
# Ojo: el código usa 29.78 x 29.78, no 65.56 x 65.56
VOLUMEN_BASE = 9.65 * 29.78 * 29.78  # mm^3

# Escalado empírico según conteo real de semillas
FACTOR_ESCALA_SEMILLAS = SEMILLAS_OBJETIVO / SEMILLAS_ACTUALES

# Volumen interior calibrado
VOLUMEN_INTERIOR = VOLUMEN_BASE * FACTOR_ESCALA_SEMILLAS * FACTOR_AJUSTE_MANUAL

# =========================================================
# PARÁMETROS DEL CILINDRO
# =========================================================
ESPESOR_PISO = 1.5
ESPESOR_PARED = 1.5

# Elegís una altura interior fija y el radio se recalcula solo
ALTURA_INTERIOR = 15.0

RADIO_INTERIOR = sqrt(VOLUMEN_INTERIOR / (pi * ALTURA_INTERIOR))
DIAMETRO_INTERIOR = 2 * RADIO_INTERIOR

RADIO_EXTERIOR = RADIO_INTERIOR + ESPESOR_PARED
DIAMETRO_EXTERIOR = 2 * RADIO_EXTERIOR
ALTURA_EXTERIOR = ALTURA_INTERIOR + ESPESOR_PISO

# =========================================================
# PARÁMETROS DE LA MANIJA TIPO CUCHARA DOSIFICADORA
# =========================================================
LARGO_MANIJA = 25.0
ANCHO_MANIJA = 14.0
ESPESOR_MANIJA = 8.0

# 0 = apoyada desde la base
# ALTURA_EXTERIOR - ESPESOR_MANIJA = alineada arriba del todo
Z_MANIJA = ALTURA_EXTERIOR - ESPESOR_MANIJA

# Pequeña penetración dentro del cuerpo para asegurar unión sólida
SOLAPE_MANIJA = 2.5

# =========================================================
# MODELO
# =========================================================
with BuildPart() as pieza:
    # Cuerpo exterior
    Cylinder(
        radius=RADIO_EXTERIOR,
        height=ALTURA_EXTERIOR,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )

    # Hueco interior dejando piso
    with Locations((0, 0, ESPESOR_PISO)):
        Cylinder(
            radius=RADIO_INTERIOR,
            height=ALTURA_INTERIOR,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
            mode=Mode.SUBTRACT,
        )

    # Manija rectangular lateral, unida en la parte superior
    x_centro_manija = RADIO_EXTERIOR + LARGO_MANIJA / 2 - SOLAPE_MANIJA / 2

    with Locations((x_centro_manija, 0, Z_MANIJA + ESPESOR_MANIJA / 2)):
        Box(
            LARGO_MANIJA + SOLAPE_MANIJA,
            ANCHO_MANIJA,
            ESPESOR_MANIJA,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
            mode=Mode.ADD,
        )

# =========================================================
# MOSTRAR
# =========================================================
show(pieza.part)

# =========================================================
# EXPORTAR STL
# =========================================================
nombre_archivo = (
    f"cuchara_{SEMILLAS_OBJETIVO}sem_"
    f"factorSem_{FACTOR_ESCALA_SEMILLAS:.3f}_"
    f"ajuste_{FACTOR_AJUSTE_MANUAL:.3f}.stl"
)
salida_stl = Path(__file__).with_name(nombre_archivo)

export_stl(pieza.part, str(salida_stl))

# =========================================================
# INFO ÚTIL
# =========================================================
print("=== CALIBRACIÓN ===")
print(f"Semillas actuales: {SEMILLAS_ACTUALES}")
print(f"Semillas objetivo: {SEMILLAS_OBJETIVO}")
print(f"Factor escala semillas: {FACTOR_ESCALA_SEMILLAS:.6f}")
print(f"Factor ajuste manual: {FACTOR_AJUSTE_MANUAL:.3f}")

print("\n=== GEOMETRÍA ===")
print(f"Volumen base: {VOLUMEN_BASE:.2f} mm^3")
print(f"Volumen interior final: {VOLUMEN_INTERIOR:.2f} mm^3")
print(f"Altura interior: {ALTURA_INTERIOR:.2f} mm")
print(f"Radio interior: {RADIO_INTERIOR:.2f} mm")
print(f"Diámetro interior: {DIAMETRO_INTERIOR:.2f} mm")
print(f"Radio exterior: {RADIO_EXTERIOR:.2f} mm")
print(f"Diámetro exterior: {DIAMETRO_EXTERIOR:.2f} mm")
print(f"Altura exterior: {ALTURA_EXTERIOR:.2f} mm")

print("\n=== ARCHIVO ===")
print(f"STL exportado en: {salida_stl}")