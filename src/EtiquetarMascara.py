"""
Herramienta de pintado a mano para generar máscaras de referencia (ground
truth) que después usa OptimizarParametros.py para buscar los parámetros de
Analizar.py que mejor reproducen lo que un humano ve en la foto.

Uso:
    python src/EtiquetarMascara.py data/experiments/20260911/raw/photos/IMG_3887.jpg

Controles:
    1        pintar RAÍZ (verde)
    2        pintar SEMILLA (azul)
    click izq  pintar con el pincel actual
    click der  borrar
    [ / ]    pincel más chico / más grande
    z        deshacer el último trazo
    c        borrar todo (con doble confirmación por teclado)
    h        mostrar/ocultar la ayuda en pantalla
    s        guardar (sin cerrar)
    q / ESC  guardar y salir

La máscara se guarda en <experimento>/raw/labels/<nombre_foto>_mask.png, con
los mismos colores puros que ve el optimizador: fondo negro, raíz verde
(0,255,0 en BGR), semilla azul (255,0,0 en BGR). Se pinta sobre EXACTAMENTE el
mismo recorte que usa el pipeline (redimensionar_para_analisis +
recortar_bordes_blancos una sola vez) para que la máscara quede alineada en
píxeles con lo que Analizar.py realmente procesa — ver la nota en
procesar_imagen(recortar=...) sobre por qué esto importa.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

import Analizar

COLOR_RAIZ = (0, 255, 0)  # BGR
COLOR_SEMILLA = (255, 0, 0)
COLOR_FONDO = (0, 0, 0)

MAX_UNDO = 25
ALPHA_OVERLAY = 0.55
VENTANA = "Etiquetar mascara"
# Altura máxima de la ventana en pantalla. La foto real puede ser mucho más
# grande (2300+ px), y con WINDOW_NORMAL de OpenCV en macOS el mapeo de
# coordenadas del mouse a esa escala interna a veces queda mal (Retina, o el
# usuario redimensiona la ventana a mano) — el click cae en un punto distinto
# de donde se ve, y da la sensación de que "no pasa nada". Para evitarlo
# manejamos el escalado nosotros mismos con una ventana de tamaño fijo
# (WINDOW_AUTOSIZE) y convertimos manualmente coordenadas de pantalla ->
# coordenadas reales de la imagen antes de pintar.
MAX_ALTO_VENTANA = 900


def resolver_ruta_mascara(ruta_foto: Path) -> Path:
    """<exp>/raw/photos/foo.jpg -> <exp>/raw/labels/foo_mask.png.

    Si la foto no está bajo raw/photos (uso suelto de la herramienta), cae a
    una carpeta 'labels' hermana de la carpeta de la foto.
    """
    partes = ruta_foto.resolve().parts
    try:
        idx = len(partes) - 1 - partes[::-1].index("photos")
        if partes[idx - 1] == "raw":
            base = Path(*partes[: idx - 1])
            carpeta_labels = base / "raw" / "labels"
            carpeta_labels.mkdir(parents=True, exist_ok=True)
            return carpeta_labels / f"{ruta_foto.stem}_mask.png"
    except ValueError:
        pass

    carpeta_labels = ruta_foto.resolve().parent / "labels"
    carpeta_labels.mkdir(parents=True, exist_ok=True)
    return carpeta_labels / f"{ruta_foto.stem}_mask.png"


def cargar_imagen_base(ruta_foto: Path) -> np.ndarray:
    img = Analizar.leer_imagen(str(ruta_foto))
    if img is None:
        raise SystemExit(f"[ERROR] No se pudo leer la imagen: {ruta_foto}")
    img = Analizar.redimensionar_para_analisis(img)
    img = Analizar.recortar_bordes_blancos(img)
    if img is None or img.size == 0:
        raise SystemExit(f"[ERROR] No se pudo recortar el borde blanco: {ruta_foto}")
    return img


class EstadoPintado:
    def __init__(self, shape, mascara_inicial=None):
        if mascara_inicial is not None:
            self.mascara = mascara_inicial.copy()
        else:
            self.mascara = np.zeros(shape, dtype=np.uint8)
        self.historial = []
        self.pincel = COLOR_RAIZ
        self.radio = 6
        self.pintando = False
        self.trazo_inicio = None

    def push_historial(self):
        self.historial.append(self.mascara.copy())
        if len(self.historial) > MAX_UNDO:
            self.historial.pop(0)

    def deshacer(self):
        if self.historial:
            self.mascara = self.historial.pop()

    def pintar(self, x, y, color):
        cv2.circle(self.mascara, (x, y), self.radio, color, -1, lineType=cv2.LINE_8)


def construir_vista(img_base, estado: EstadoPintado, mostrar_ayuda: bool, factor: float) -> np.ndarray:
    vista = img_base.copy()
    pintado = estado.mascara.sum(axis=2) > 0
    if pintado.any():
        capa = np.zeros_like(img_base)
        capa[:] = estado.mascara
        vista[pintado] = cv2.addWeighted(
            img_base, 1 - ALPHA_OVERLAY, capa, ALPHA_OVERLAY, 0
        )[pintado]

    if factor < 1.0:
        vista = cv2.resize(
            vista, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA
        )

    nombre_pincel = {
        COLOR_RAIZ: "RAIZ (verde)",
        COLOR_SEMILLA: "SEMILLA (azul)",
    }.get(estado.pincel, "?")
    cv2.putText(
        vista,
        f"Pincel: {nombre_pincel}  |  radio: {estado.radio}px",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    if mostrar_ayuda:
        lineas = [
            "1=raiz  2=semilla  click_izq=pintar  click_der=borrar",
            "[ ]=tamano pincel   z=deshacer   c=borrar todo",
            "s=guardar   q/ESC=guardar y salir   h=ocultar ayuda",
        ]
        for i, linea in enumerate(lineas):
            cv2.putText(
                vista,
                linea,
                (15, 60 + i * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return vista


def guardar_mascara(ruta_mascara: Path, mascara: np.ndarray):
    cv2.imwrite(str(ruta_mascara), mascara)
    raiz_px = int(np.all(mascara == COLOR_RAIZ, axis=2).sum())
    semilla_px = int(np.all(mascara == COLOR_SEMILLA, axis=2).sum())
    print(
        f"[OK] Mascara guardada en {ruta_mascara} "
        f"(raiz: {raiz_px} px, semilla: {semilla_px} px)"
    )


def main():
    if len(sys.argv) < 2:
        print("Uso: python src/EtiquetarMascara.py <ruta_a_la_foto>")
        raise SystemExit(1)

    ruta_foto = Path(sys.argv[1])
    if not ruta_foto.exists():
        raise SystemExit(f"[ERROR] No existe: {ruta_foto}")

    img_base = cargar_imagen_base(ruta_foto)
    ruta_mascara = resolver_ruta_mascara(ruta_foto)

    mascara_inicial = None
    if ruta_mascara.exists():
        previa = cv2.imread(str(ruta_mascara), cv2.IMREAD_COLOR)
        if previa is not None and previa.shape == img_base.shape:
            mascara_inicial = previa
            print(f"[INFO] Retomando mascara existente: {ruta_mascara}")
        else:
            print(
                f"[ALERTA] {ruta_mascara} no coincide en tamaño con la foto actual "
                "(¿cambió la resolución de origen?). Se empieza de cero."
            )

    estado = EstadoPintado(img_base.shape, mascara_inicial)
    mostrar_ayuda = True

    h_full, w_full = img_base.shape[:2]
    factor = min(1.0, MAX_ALTO_VENTANA / h_full)
    print(
        f"[INFO] Imagen real: {w_full}x{h_full}px. Mostrando a escala {factor:.2f} "
        f"({int(w_full*factor)}x{int(h_full*factor)}px en pantalla)."
    )

    confirmar_borrado = False

    def a_coords_reales(x, y):
        """Pantalla -> imagen real. Todo lo que se pinta usa SIEMPRE esta
        coordenada, nunca la de pantalla directamente."""
        xf = int(round(x / factor))
        yf = int(round(y / factor))
        xf = min(max(xf, 0), w_full - 1)
        yf = min(max(yf, 0), h_full - 1)
        return xf, yf

    def on_mouse(event, x, y, flags, param):
        nonlocal confirmar_borrado
        if event == cv2.EVENT_LBUTTONDOWN:
            xf, yf = a_coords_reales(x, y)
            print(f"[click] pantalla=({x},{y}) -> imagen=({xf},{yf})")
            estado.push_historial()
            estado.pintando = True
            estado.pintar(xf, yf, estado.pincel)
        elif event == cv2.EVENT_RBUTTONDOWN:
            xf, yf = a_coords_reales(x, y)
            estado.push_historial()
            estado.pintando = "borrar"
            estado.pintar(xf, yf, COLOR_FONDO)
        elif event == cv2.EVENT_MOUSEMOVE and estado.pintando:
            xf, yf = a_coords_reales(x, y)
            color = COLOR_FONDO if estado.pintando == "borrar" else estado.pincel
            estado.pintar(xf, yf, color)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            estado.pintando = False

    cv2.namedWindow(VENTANA, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(VENTANA, on_mouse)

    print(f"Pintando sobre: {ruta_foto}")
    print(f"Se va a guardar en: {ruta_mascara}")
    print("1=raiz 2=semilla | click_izq=pintar click_der=borrar | []=pincel z=deshacer")
    print("c=borrar todo | s=guardar | q/ESC=guardar y salir")
    print("(cada click imprime sus coordenadas acá abajo — si no ves nada al hacer")
    print(" click, el problema es que la ventana no tiene foco o no está recibiendo")
    print(" eventos, no el pintado en sí)")

    while True:
        vista = construir_vista(img_base, estado, mostrar_ayuda, factor)
        cv2.imshow(VENTANA, vista)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("1"):
            estado.pincel = COLOR_RAIZ
        elif key == ord("2"):
            estado.pincel = COLOR_SEMILLA
        elif key == ord("["):
            estado.radio = max(1, estado.radio - 1)
        elif key == ord("]"):
            estado.radio = min(60, estado.radio + 1)
        elif key == ord("z"):
            estado.deshacer()
            confirmar_borrado = False
        elif key == ord("h"):
            mostrar_ayuda = not mostrar_ayuda
        elif key == ord("c"):
            if confirmar_borrado:
                estado.push_historial()
                estado.mascara[:] = 0
                confirmar_borrado = False
                print("[INFO] Mascara borrada.")
            else:
                confirmar_borrado = True
                print("[INFO] Presioná 'c' de nuevo para confirmar borrar todo.")
        elif key == ord("s"):
            guardar_mascara(ruta_mascara, estado.mascara)
            confirmar_borrado = False
        elif key in (ord("q"), 27):
            break
        else:
            confirmar_borrado = False

        try:
            if cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    guardar_mascara(ruta_mascara, estado.mascara)
    cv2.destroyAllWindows()
    print("\nListo. Ahora corré:")
    exp_id = ruta_mascara.parents[2].name
    print(f"  python src/OptimizarParametros.py {exp_id}")


if __name__ == "__main__":
    main()
