from pathlib import Path
from build123d import *
from ocp_vscode import show, set_port

# Forzar el puerto correcto para el viewer
set_port(3939)

# Medidas promedio pensadas para maximizar la chance de que funcione
# con semillas de maíz de distintos tamaños sin que se apilen dos.
ANCHO_PLACA = 80
ALTO_PLACA = 60
ESPESOR_PLACA = 6

PASO_X = 15
PASO_Y = 15
CANT_X = 5
CANT_Y = 4

DIAMETRO_SUPERIOR = 14   # mm
PROFUNDIDAD = 4.5        # mm
TAPER = 45               # grados

# Parámetros de la tapa
ESPESOR_TAPA = 3
HOLGURA_TAPA = 0.7       # juego para que entre sin trabarse
ALTURA_PAREDES_TAPA = 7 # altura total de las paredes que bajan sobre la placa
ESPESOR_PAREDES_TAPA = 2 # grosor de las paredes laterales
SEPARACION_VISTA = 100   # separación entre placa y tapa al mostrar

with BuildPart() as placa:
    # Placa base
    Box(ANCHO_PLACA, ALTO_PLACA, ESPESOR_PLACA)

    # Cara superior de la placa
    cara_superior = placa.faces().sort_by(Axis.Z)[-1]

    # Dibujar la boca de cada cavidad
    with BuildSketch(cara_superior):
        with GridLocations(PASO_X, PASO_Y, CANT_X, CANT_Y):
            Circle(DIAMETRO_SUPERIOR / 2)

    # Cavidades ciegas cónicas, sin perforar el fondo
    extrude(
        amount=-PROFUNDIDAD,
        taper=TAPER,
        mode=Mode.SUBTRACT,
    )

# Tapa tipo capuchón, más grande que la placa y con paredes que calzan por afuera
with BuildPart() as tapa:
    # Tapa superior
    Box(
        ANCHO_PLACA + 2 * (ESPESOR_PAREDES_TAPA + HOLGURA_TAPA),
        ALTO_PLACA + 2 * (ESPESOR_PAREDES_TAPA + HOLGURA_TAPA),
        ESPESOR_TAPA,
    )

    cara_inferior_tapa = tapa.faces().sort_by(Axis.Z)[0]

    with BuildSketch(cara_inferior_tapa):
        # Anillo de paredes que baja alrededor de la placa
        Rectangle(
            ANCHO_PLACA + 2 * (ESPESOR_PAREDES_TAPA + HOLGURA_TAPA),
            ALTO_PLACA + 2 * (ESPESOR_PAREDES_TAPA + HOLGURA_TAPA),
        )

        Rectangle(
            ANCHO_PLACA + 2 * HOLGURA_TAPA,
            ALTO_PLACA + 2 * HOLGURA_TAPA,
            mode=Mode.SUBTRACT,
        )

    extrude(
        amount=-ALTURA_PAREDES_TAPA,
        mode=Mode.ADD,
    )


# Mostrar ambas piezas separadas en el viewer
show(
    placa.part,
    Pos(SEPARACION_VISTA, 0, 0) * tapa.part,
)

# Exportar a STL en la misma carpeta que este script
salida_stl = Path(__file__).with_name("placa_semillas.stl")
salida_tapa_stl = Path(__file__).with_name("tapa_semillas.stl")

export_stl(placa.part, str(salida_stl))
export_stl(tapa.part, str(salida_tapa_stl))

print(f"STL exportado en: {salida_stl}")
print(f"STL exportado en: {salida_tapa_stl}")