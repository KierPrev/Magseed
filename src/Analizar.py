import cv2
import numpy as np
import pandas as pd
import os
import glob
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from skimage.morphology import skeletonize

try:
    import zxingcpp
except ImportError:
    zxingcpp = None

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DIRECTORIO_RAIZ = str(BASE_DIR / "data" / "experiments")
ARCHIVO_SALIDA = "root_lengths.csv"
ARCHIVO_CONFIG = str(Path(__file__).resolve().with_name("parametros_calibracion.json"))
ARCHIVO_SALIDA_CRUDO = "root_lengths_raw.csv"
ARCHIVO_FOTOS_PROCESADAS = "root_lengths_photos.csv"
# Precisión fija en todos los CSV. Con el repr por defecto, leer el caché y
# volver a escribirlo cambia el último dígito del float y el pipeline deja de ser
# reproducible. 12 cifras significativas sobran para un largo en mm derivado de
# contar píxeles de un esqueleto.
FLOAT_FORMAT_CSV = "%.12g"
GUARDAR_IMAGENES_TESTIGO = True
CARPETA_TESTIGOS = "procesadas_debug"
CARPETA_CACHE_IMAGENES = ".analisis_cache"
EXTENSION_CACHE_IMAGEN = ".png"
HEIC_SUFFIXES = {".heic", ".heif"}

LARGO_REAL_REFERENCIA_CM = 11.09
LARGO_REAL_REFERENCIA_MM = LARGO_REAL_REFERENCIA_CM * 10.0

# Regla blanca agregada a las fotos nuevas. Mide 100 mm exactos y es, de lejos,
# la referencia con menor error relativo (el QR tiene apenas 17 mm de lado).
REGLA_BLANCA_LARGO_MM = 100.0
REGLA_BLANCA_ASPECT_MIN = 6.0
REGLA_BLANCA_AREA_MIN_PX = 1500
REGLA_BLANCA_SOLIDEZ_MIN = 0.8
REGLA_BLANCA_DILATACION_PX = 6
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
A4_WIDTH_PX = 2480
A4_HEIGHT_PX = 3508
ETIQUETA_PAGE_MARGIN_X = 120
ETIQUETA_PAGE_MARGIN_Y = 120
ETIQUETA_COLS = 3
ETIQUETA_ROWS = 9
ETIQUETA_GAP_X = 40
ETIQUETA_GAP_Y = 30


def calcular_medidas_etiqueta_mm():
    usable_width = A4_WIDTH_PX - (2 * ETIQUETA_PAGE_MARGIN_X)
    usable_height = A4_HEIGHT_PX - (2 * ETIQUETA_PAGE_MARGIN_Y)
    cell_w_px = (usable_width - ((ETIQUETA_COLS - 1) * ETIQUETA_GAP_X)) / ETIQUETA_COLS
    cell_h_px = (usable_height - ((ETIQUETA_ROWS - 1) * ETIQUETA_GAP_Y)) / ETIQUETA_ROWS
    cell_w_mm = cell_w_px * (A4_WIDTH_MM / A4_WIDTH_PX)
    cell_h_mm = cell_h_px * (A4_HEIGHT_MM / A4_HEIGHT_PX)
    return cell_w_mm, cell_h_mm


ETIQUETA_ANCHO_MM, ETIQUETA_ALTO_MM = calcular_medidas_etiqueta_mm()
ETIQUETA_LADO_MAYOR_MM = max(ETIQUETA_ANCHO_MM, ETIQUETA_ALTO_MM)
QR_LADO_REAL_MM = 17.274193548387096
MAX_DIM_ANALISIS = 2600
BLACK_BG_GRAY_MAX = 95
BLACK_BG_INSET_PX = 18
QR_BOTTOM_EXCLUSION_START = 0.76
QR_BOTTOM_EXCLUSION_MARGIN_PX = 6
MAX_ROOT_LENGTH_MM = 120.0

# Valores por defecto
PARAMS = {
    "blur_k": 5,
    "clean_k": 1,  # Default bajo
    "seed_glue": 7,
    "seed_h_max": 30,
    "seed_s_min": 75,
    "seed_v_min": 65,
    "root_s_max": 65,
    "root_v_min": 150,
    "min_area_seed": 90,
    "root_bridge": 3,
    "min_area_root": 30,  # Default bajo para raíces pequeñas
    "hsv_h_min": 15,
    "hsv_h_max": 40,
    # Antes hardcodeados en el filtro de contornos de raíces. Se exponen acá para
    # que OptimizarParametros.py también pueda buscar sobre ellos, y para que
    # sean parte del calibracion.json de cada experimento como todo lo demás.
    "root_solidity_max": 0.6,
    "root_aspect_min": 0.12,
    "root_aspect_max": 8.0,
}
# Copia congelada de los valores por defecto. procesar_carpetas() la usa para
# resetear PARAMS antes de cargar el config de cada carpeta: cargar_configuracion
# solo pisa las claves que trae el JSON, así que sin este reset una carpeta sin
# 'blur_k' en su archivo (por ejemplo) heredaría el valor que dejó la carpeta
# anterior en vez de caer al default real.
PARAMS_POR_DEFECTO = dict(PARAMS)

# ==========================================
# GESTIÓN DE CONFIGURACIÓN
# ==========================================


def cargar_configuracion(archivo_config=ARCHIVO_CONFIG):
    global PARAMS
    if os.path.exists(archivo_config):
        try:
            with open(archivo_config, "r") as f:
                datos = json.load(f)
                for k, v in datos.items():
                    if k in PARAMS:
                        PARAMS[k] = v
            print(f"\n[INFO] Configuración cargada desde '{archivo_config}'")
        except Exception as e:
            print(f"\n[ALERTA] Error leyendo config: {e}")
    else:
        print("\n[INFO] Usando valores por defecto.")


def guardar_configuracion(archivo_config=ARCHIVO_CONFIG):
    try:
        with open(archivo_config, "w") as f:
            json.dump(PARAMS, f, indent=4)
        print(f"[INFO] Configuración guardada en '{archivo_config}'")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar config: {e}")


# ==========================================
# FUNCIONES AUXILIARES
# ==========================================


def recortar_bordes_blancos(img):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = img.shape[:2]

    # El fondo útil es la cartulina negra. Detectamos la región oscura principal
    # y recortamos un poco hacia adentro para descartar el exterior claro.
    mask_black = cv2.inRange(gray, 0, BLACK_BG_GRAY_MAX)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > 0.15 * float(h_img * w_img):
            x, y, w, h = cv2.boundingRect(c)
            inset = BLACK_BG_INSET_PX
            x0 = min(max(x + inset, 0), w_img - 1)
            y0 = min(max(y + inset, 0), h_img - 1)
            x1 = max(min(x + w - inset, w_img), x0 + 1)
            y1 = max(min(y + h - inset, h_img), y0 + 1)
            return img[y0:y1, x0:x1]

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    puntos = cv2.findNonZero(thresh)
    if puntos is not None:
        x, y, w, h = cv2.boundingRect(puntos)
        margen = 30
        x = max(0, x - margen)
        y = max(0, y - margen)
        w = min(w_img - x, w + (margen * 2))
        h = min(h_img - y, h + (margen * 2))
        return img[y : y + h, x : x + w]
    return img


def decodificar_imagen_desde_archivo(ruta_archivo):
    try:
        data = np.fromfile(str(ruta_archivo), dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception:
        pass

    return None


def guardar_imagen_cache(ruta_cache, img):
    img_cache = redimensionar_para_analisis(img)
    cv2.imwrite(str(ruta_cache), img_cache)
    return img_cache


def cargar_imagen_cache(ruta_cache):
    img_cache = decodificar_imagen_desde_archivo(ruta_cache)
    if img_cache is None:
        return None

    img_reducida = redimensionar_para_analisis(img_cache)
    if img_reducida.shape[:2] != img_cache.shape[:2]:
        guardar_imagen_cache(ruta_cache, img_reducida)
        return img_reducida

    return img_cache


def obtener_ruta_cache_imagen(ruta_imagen):
    ruta = Path(ruta_imagen)
    return ruta.parent / CARPETA_CACHE_IMAGENES / f"{ruta.stem}{EXTENSION_CACHE_IMAGEN}"


def leer_imagen(ruta_imagen):
    ruta = Path(ruta_imagen)
    cache_ruta = None

    if ruta.suffix.lower() in HEIC_SUFFIXES:
        cache_ruta = obtener_ruta_cache_imagen(ruta)
        try:
            if (
                cache_ruta.exists()
                and cache_ruta.stat().st_mtime >= ruta.stat().st_mtime
            ):
                img_cache = cargar_imagen_cache(cache_ruta)
                if img_cache is not None:
                    return img_cache
        except OSError:
            pass

    img = decodificar_imagen_desde_archivo(ruta)
    if img is not None:
        return img

    if ruta.suffix.lower() in HEIC_SUFFIXES and shutil.which("sips"):
        if cache_ruta is None:
            cache_ruta = obtener_ruta_cache_imagen(ruta)

        cache_ruta.parent.mkdir(parents=True, exist_ok=True)
        temporal = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=EXTENSION_CACHE_IMAGEN, delete=False
            ) as tmp:
                temporal = Path(tmp.name)

            resultado = subprocess.run(
                [
                    "sips",
                    "-s",
                    "format",
                    EXTENSION_CACHE_IMAGEN.lstrip("."),
                    str(ruta),
                    "--out",
                    str(temporal),
                ],
                capture_output=True,
                text=True,
            )
            if resultado.returncode == 0 and temporal.exists():
                temporal.replace(cache_ruta)
                img_cache = cargar_imagen_cache(cache_ruta)
                if img_cache is not None:
                    return img_cache
        except Exception:
            pass
        finally:
            if temporal is not None:
                temporal.unlink(missing_ok=True)

    return None


def redimensionar_para_analisis(img, max_dim=MAX_DIM_ANALISIS):
    if img is None:
        return None

    h, w = img.shape[:2]
    mayor = max(h, w)
    if mayor <= max_dim:
        return img

    escala = max_dim / float(mayor)
    return cv2.resize(
        img, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_AREA
    )


def obtener_largo_esqueleto(img_binaria_radicula):
    bool_image = img_binaria_radicula > 0
    skeleton = skeletonize(bool_image)
    return np.count_nonzero(skeleton)


def obtener_largo_esqueleto_contorno(cnt, shape_img, margen=2):
    x, y, w, h = cv2.boundingRect(cnt)
    x0 = max(x - margen, 0)
    y0 = max(y - margen, 0)
    x1 = min(x + w + margen, shape_img[1])
    y1 = min(y + h + margen, shape_img[0])

    mask_single = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cnt_local = cnt.astype(np.int32, copy=True)
    cnt_local[:, :, 0] -= x0
    cnt_local[:, :, 1] -= y0
    cv2.drawContours(mask_single, [cnt_local], -1, 255, -1)
    return obtener_largo_esqueleto(mask_single)


def detectar_mascara_semillas(hsv, mask_total, params):
    h, s, v = cv2.split(hsv)

    # Las semillas son cálidas y saturadas; las radículas son muy claras y
    # poco saturadas. Aprovechamos esa diferencia para separar ambas clases.
    mask_semillas_color = np.where(
        ((h <= params["seed_h_max"]) | (h >= 170))
        & (s >= params["seed_s_min"])
        & (v >= params["seed_v_min"]),
        255,
        0,
    ).astype(np.uint8)

    mask_radiculas_blancas = np.where(
        (s <= params["root_s_max"]) & (v >= params["root_v_min"]),
        255,
        0,
    ).astype(np.uint8)

    mask_semillas = cv2.bitwise_and(mask_semillas_color, mask_total)
    mask_semillas = cv2.bitwise_and(
        mask_semillas, cv2.bitwise_not(mask_radiculas_blancas)
    )

    k_glue = (
        params["seed_glue"] if params["seed_glue"] % 2 == 1 else params["seed_glue"] + 1
    )
    if k_glue < 1:
        k_glue = 1

    kernel_seed = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_glue, k_glue))
    mask_semillas = cv2.morphologyEx(mask_semillas, cv2.MORPH_CLOSE, kernel_seed)
    mask_semillas = cv2.morphologyEx(
        mask_semillas, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )

    return mask_semillas


def extraer_semillas_desde_mascara(mask_semillas, area_minima):
    detalles_semillas = []
    cnts_sem, _ = cv2.findContours(
        mask_semillas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for i, cnt in enumerate(cnts_sem):
        area = cv2.contourArea(cnt)
        if area < area_minima:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / h if h > 0 else 0
        if not (0.35 < aspect < 3.5):
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue

        solidez = float(area) / hull_area
        if solidez < 0.55:
            continue

        detalles_semillas.append(
            {
                "id": i,
                "area_px": area,
                "bbox": (x, y, w, h),
                "contour": cnt,
            }
        )

    return detalles_semillas


def limite_zona_qr_inferior(alto_img):
    return int(alto_img * QR_BOTTOM_EXCLUSION_START)


def excluir_zona_qr_inferior(mask):
    mask_filtrada = mask.copy()
    y_inicio = limite_zona_qr_inferior(mask.shape[0])
    mask_filtrada[y_inicio:, :] = 0
    return mask_filtrada, y_inicio


def contorno_toca_zona_qr_inferior(y, h, y_limite):
    return (y + h) >= (y_limite - QR_BOTTOM_EXCLUSION_MARGIN_PX)


def rotar_imagen_90(img, rotaciones):
    rotaciones = rotaciones % 4
    if rotaciones == 0:
        return img
    if rotaciones == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotaciones == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def recortar_margen(img, top=0.0, bottom=0.0, left=0.0, right=0.0):
    h, w = img.shape[:2]
    y0 = min(max(int(h * top), 0), h - 1)
    y1 = max(min(int(h * (1.0 - bottom)), h), y0 + 1)
    x0 = min(max(int(w * left), 0), w - 1)
    x1 = max(min(int(w * (1.0 - right)), w), x0 + 1)
    return img[y0:y1, x0:x1]


def normalizar_tamano_qr(img, max_dim=1100):
    h, w = img.shape[:2]
    mayor = max(h, w)
    if mayor <= max_dim:
        return img

    escala = max_dim / float(mayor)
    return cv2.resize(img, (0, 0), fx=escala, fy=escala, interpolation=cv2.INTER_AREA)


def generar_regiones_qr(img):
    bases = [img]
    img_crop = recortar_bordes_blancos(img)
    if img_crop is not None and img_crop.shape != img.shape:
        bases.append(img_crop)

    regiones = []
    for base in bases:
        regiones.extend(
            [
                recortar_margen(base, top=0.70, left=0.20, right=0.20),
                recortar_margen(base, top=0.76, left=0.25, right=0.25),
                recortar_margen(base, top=0.80, left=0.30, right=0.30),
                recortar_margen(base, left=0.78, top=0.16, bottom=0.12),
                recortar_margen(base, left=0.76, top=0.12, bottom=0.10),
                recortar_margen(base, left=0.72, top=0.10, bottom=0.10),
                recortar_margen(base, right=0.78, top=0.16, bottom=0.12),
                recortar_margen(base, right=0.76, top=0.12, bottom=0.10),
                base,
            ]
        )

    return [region for region in regiones if region is not None and region.size > 0]


def generar_variantes_qr(img):
    base = normalizar_tamano_qr(img)
    variantes = [base]

    gris = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    variantes.append(gris)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gris)
    variantes.append(clahe)

    _, otsu = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes.append(otsu)

    adapt = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    variantes.append(adapt)

    nitida = cv2.GaussianBlur(base, (0, 0), 3)
    nitida = cv2.addWeighted(base, 1.6, nitida, -0.6, 0)
    variantes.append(nitida)

    escalas = [1.8]
    variantes_escaladas = []
    for variante in variantes[:3]:
        for escala in escalas:
            variantes_escaladas.append(
                cv2.resize(
                    variante,
                    (0, 0),
                    fx=escala,
                    fy=escala,
                    interpolation=cv2.INTER_CUBIC,
                )
            )

    return variantes + variantes_escaladas


def normalizar_tray_id_qr(texto_qr, exp_id):
    if texto_qr is None:
        return None

    texto = str(texto_qr).strip().upper()
    if not texto:
        return None

    texto = texto.replace("-", "_")
    texto = re.sub(r"\s+", "", texto)

    match_corto = re.fullmatch(r"([A-Z])(\d{1,2})", texto)
    if match_corto:
        fila, posicion = match_corto.groups()
        return f"{exp_id}_{fila}{int(posicion)}"

    match_largo = re.fullmatch(r"(\d{8})_?([A-Z])(\d{1,2})", texto)
    if match_largo:
        exp_qr, fila, posicion = match_largo.groups()
        return f"{exp_qr}_{fila}{int(posicion)}"

    return None


def leer_textos_qr(candidata, detector):
    textos = []

    if zxingcpp is not None:
        try:
            resultados = zxingcpp.read_barcodes(
                candidata, formats=zxingcpp.BarcodeFormat.QRCode
            )
            textos.extend(
                [resultado.text for resultado in resultados if resultado.text]
            )
        except Exception:
            pass

    try:
        detectado, textos_multi, _, _ = detector.detectAndDecodeMulti(candidata)
        if detectado:
            textos.extend(textos_multi)
    except Exception:
        pass

    try:
        texto_simple, _, _ = detector.detectAndDecode(candidata)
        if texto_simple:
            textos.append(texto_simple)
    except Exception:
        pass

    return textos


def detectar_qr_rapido(region, detector):
    preview = normalizar_tamano_qr(region, max_dim=2000)
    escala = region.shape[1] / float(preview.shape[1])
    variantes = [preview, cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)]

    for variante in variantes:
        for rotacion in (0, 1, 3):
            candidata = rotar_imagen_90(variante, rotacion)
            texto_simple = None
            puntos = None
            texto_multi = None
            puntos_multi = None

            try:
                texto_simple, puntos, _ = detector.detectAndDecode(candidata)
            except Exception:
                texto_simple = None
                puntos = None

            try:
                detectado_multi, textos_multi, puntos_multi, _ = (
                    detector.detectAndDecodeMulti(candidata)
                )
                if detectado_multi and textos_multi:
                    for texto_candidato in textos_multi:
                        if texto_candidato:
                            texto_multi = texto_candidato
                            break
            except Exception:
                texto_multi = None
                puntos_multi = None

            textos = []
            if texto_simple:
                textos.append(texto_simple)
            if texto_multi:
                textos.append(texto_multi)
            textos.extend(leer_textos_qr(candidata, detector))

            puntos_validos = None
            if puntos is not None and len(puntos) > 0:
                puntos_validos = puntos[0]
            elif puntos_multi is not None and len(puntos_multi) > 0:
                puntos_validos = puntos_multi[0]

            for texto in textos:
                if texto:
                    return texto, puntos_validos, escala

    return None, None, None


def calcular_lado_promedio_qr(puntos):
    if puntos is None or len(puntos) != 4:
        return None

    lados = []
    for i in range(4):
        p1 = puntos[i]
        p2 = puntos[(i + 1) % 4]
        lados.append(float(np.linalg.norm(p1 - p2)))

    if not lados:
        return None

    return sum(lados) / len(lados)


def detectar_info_qr(img, exp_id=None):
    detector = cv2.QRCodeDetector()
    regiones = generar_regiones_qr(img)
    tray_id = None
    px_por_mm_qr = None

    for region in regiones[:5]:
        texto, puntos, escala = detectar_qr_rapido(region, detector)

        if tray_id is None:
            tray_id = normalizar_tray_id_qr(texto, exp_id) if exp_id else None

        if px_por_mm_qr is None and texto and puntos is not None and escala is not None:
            lado_promedio = calcular_lado_promedio_qr(puntos)
            if lado_promedio is not None and lado_promedio > 0:
                px_por_mm_qr = (lado_promedio * escala) / QR_LADO_REAL_MM

        if tray_id is not None and px_por_mm_qr is not None:
            return tray_id, px_por_mm_qr

    if exp_id is None or tray_id is not None:
        return tray_id, px_por_mm_qr

    for region_idx, region in enumerate(regiones):
        rotaciones = (0, 1, 3) if region_idx < 5 else (0, 1)
        for variante in generar_variantes_qr(region):
            for rotacion in rotaciones:
                candidata = rotar_imagen_90(variante, rotacion)
                for texto in leer_textos_qr(candidata, detector):
                    tray_id = normalizar_tray_id_qr(texto, exp_id)
                    if tray_id:
                        return tray_id, px_por_mm_qr

    return tray_id, px_por_mm_qr


def detectar_tray_id(img, exp_id):
    tray_id, _ = detectar_info_qr(img, exp_id)
    return tray_id


def detectar_referencia_qr(img):
    _, px_por_mm_qr = detectar_info_qr(img)
    return px_por_mm_qr, px_por_mm_qr is not None


def detectar_referencia_etiqueta(img):
    for region in generar_regiones_qr(img)[:5]:
        preview = normalizar_tamano_qr(region, max_dim=1600)
        escala = region.shape[1] / float(preview.shape[1])

        hsv = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 80, 255]))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        mejor_largo_px = 0.0
        mejor_score = 0.0

        alto, ancho = mask.shape[:2]
        area_total = float(alto * ancho)
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            toca_borde = x == 0 or y == 0 or (x + w) >= ancho or (y + h) >= alto
            if toca_borde or area < 5000:
                continue

            cobertura = area / area_total
            if cobertura < 0.005 or cobertura > 0.25:
                continue

            puntos = np.column_stack(np.where(labels == label))
            puntos = np.flip(puntos, axis=1).astype(np.float32)
            rect = cv2.minAreaRect(puntos)
            largo = max(rect[1])
            ancho_rect = min(rect[1])
            if ancho_rect <= 0:
                continue

            aspect_ratio = largo / ancho_rect
            if aspect_ratio < 1.3 or aspect_ratio > 3.0:
                continue

            score = area * aspect_ratio
            if score > mejor_score:
                mejor_score = score
                mejor_largo_px = largo * escala

        if mejor_largo_px > 0:
            return mejor_largo_px / ETIQUETA_LADO_MAYOR_MM, True

    return None, False


def detectar_referencia_regla_blanca(hsv):
    """Busca la regla blanca de 100 mm de las fotos nuevas.

    Devuelve (px_por_mm, encontrada, rect). El rect sirve para después borrar la
    regla de la máscara de radículas, así no se la mide como si fuera una raíz.

    Las fotos viejas no tienen ninguna barra blanca tan alargada, así que acá
    devuelven False y la cascada sigue con el QR igual que antes.
    """
    mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 90, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    alto, ancho = mask.shape[:2]
    mejor_largo_px = 0.0
    mejor_rect = None

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if x == 0 or y == 0 or (x + w) >= ancho or (y + h) >= alto:
            continue
        if area < REGLA_BLANCA_AREA_MIN_PX:
            continue

        puntos = np.column_stack(np.where(labels == label))
        puntos = np.flip(puntos, axis=1).astype(np.float32)
        rect = cv2.minAreaRect(puntos)
        largo = max(rect[1])
        ancho_rect = min(rect[1])
        if ancho_rect <= 0:
            continue
        if (largo / ancho_rect) < REGLA_BLANCA_ASPECT_MIN:
            continue

        # Una regla es una barra recta y llena: ocupa casi todo su minAreaRect.
        # Una maraña de radículas blancas, no.
        area_rect = largo * ancho_rect
        if area_rect <= 0 or (float(area) / area_rect) < REGLA_BLANCA_SOLIDEZ_MIN:
            continue

        if largo > mejor_largo_px:
            mejor_largo_px = largo
            mejor_rect = rect

    if mejor_rect is None or mejor_largo_px <= 0:
        return None, False, None

    return mejor_largo_px / REGLA_BLANCA_LARGO_MM, True, mejor_rect


def dibujar_etiqueta_tray(img_debug, tray_id, inferido):
    """Tray en cyan si salió del QR; en naranja con '(inferido)' si se dedujo
    por el orden de toma, para que en el testigo se note a simple vista."""
    if img_debug is None:
        return
    texto = f"{tray_id} (inferido)" if inferido else tray_id
    color = (0, 165, 255) if inferido else (255, 255, 0)
    cv2.putText(img_debug, texto, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def dibujar_etiqueta_referencia(img_debug, texto):
    if img_debug is None:
        return
    cv2.putText(
        img_debug,
        f"REF: {texto}",
        (20, img_debug.shape[0] - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )


def detectar_referencia(hsv, img_debug, params, img_original=None, px_por_mm_qr=None):
    """Cascada de referencias de escala, de la más precisa a la menos precisa.

    Devuelve (px_por_mm, encontrada, origen, rect_regla).
    """
    px_por_mm_regla, ref_regla, rect_regla = detectar_referencia_regla_blanca(hsv)
    if ref_regla and px_por_mm_regla is not None:
        if img_debug is not None:
            box = np.int64(cv2.boxPoints(rect_regla))
            cv2.drawContours(img_debug, [box], 0, (0, 255, 255), 3)
        dibujar_etiqueta_referencia(img_debug, "REGLA100")
        return px_por_mm_regla, True, "REGLA100", rect_regla

    if px_por_mm_qr is None and img_original is not None:
        px_por_mm_qr, _ = detectar_referencia_qr(img_original)

    if px_por_mm_qr is not None:
        dibujar_etiqueta_referencia(img_debug, "QR")
        return px_por_mm_qr, True, "QR", None

    if img_original is not None:
        px_por_mm_label, ref_label = detectar_referencia_etiqueta(img_original)
        if ref_label and px_por_mm_label is not None:
            dibujar_etiqueta_referencia(img_debug, "LABEL")
            return px_por_mm_label, True, "LABEL", None

    amarillo_min = np.array([params["hsv_h_min"], 80, 80])
    amarillo_max = np.array([params["hsv_h_max"], 255, 255])
    mask_ref_color = cv2.inRange(hsv, amarillo_min, amarillo_max)

    mask_ref_color = cv2.morphologyEx(
        mask_ref_color, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    mask_ref_color = cv2.morphologyEx(
        mask_ref_color, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )

    cnts_ref, _ = cv2.findContours(
        mask_ref_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    mejor_ref = None
    max_area_ref = 0

    for c in cnts_ref:
        area = cv2.contourArea(c)
        if area > 500:
            rect = cv2.minAreaRect(c)
            (center), (w_rect, h_rect), angle = rect
            largo = max(w_rect, h_rect)
            ancho = min(w_rect, h_rect)
            if ancho > 0:
                aspect_ratio = largo / ancho
                if aspect_ratio > 2.5:
                    if area > max_area_ref:
                        max_area_ref = area
                        mejor_ref = c

    if mejor_ref is not None:
        rect = cv2.minAreaRect(mejor_ref)
        largo_px = max(rect[1])
        if largo_px > 0:
            if img_debug is not None:
                box = np.int64(cv2.boxPoints(rect))
                cv2.drawContours(img_debug, [box], 0, (255, 255, 0), 3)
            dibujar_etiqueta_referencia(img_debug, "AMARILLA")
            return largo_px / LARGO_REAL_REFERENCIA_MM, True, "AMARILLA", None

    return 1.0, False, None, None


def procesar_imagen(
    img_original,
    params,
    px_por_mm_qr=None,
    generar_debug=True,
    recortar=True,
    devolver_mascaras=False,
    referencia_fija=None,
):
    """
    recortar=False: no vuelve a recortar bordes blancos. procesar_carpetas ya
    recorta antes de llamar a esta función, así que en el uso normal esto
    recorta DOS veces (nunca rompió nada porque el segundo recorte es casi
    siempre un no-op sobre fondo negro, pero es doble trabajo). Lo dejamos en
    True por compatibilidad y porque cambiarlo ahora correría el origen de
    coordenadas de resultados ya validados. OptimizarParametros.py y
    EtiquetarMascara.py sí necesitan False: tienen que operar sobre EXACTAMENTE
    el mismo recorte que se pintó a mano, o la máscara pintada queda
    desalineada en píxeles respecto de lo que el algoritmo realmente mide.

    devolver_mascaras=True agrega al resultado 'mask_semillas_pred' y
    'mask_radiculas_pred': máscaras binarias rellenas con SOLO los contornos
    que sobrevivieron todos los filtros (no la máscara cruda de color). Son
    las que hay que comparar con una máscara pintada a mano.

    referencia_fija=(px_por_mm, ref_encontrada, ref_origen, rect_regla) salta
    por completo detectar_referencia(). La detección de escala (regla blanca,
    QR, etiqueta, regla amarilla) no depende de 'params' en absoluto, así que
    recalcularla en cada evaluación durante una búsqueda de parámetros es
    trabajo tirado: OptimizarParametros.py la calcula una vez por foto y la
    reusa en todas las iteraciones.
    """
    if recortar:
        img = recortar_bordes_blancos(img_original.copy())
        if img is None or img.size == 0:
            img = img_original.copy()
    else:
        img = img_original
    img_debug = img.copy() if generar_debug else None
    h_img, w_img = img.shape[:2]

    # 1. Suavizado
    k_blur = params["blur_k"] if params["blur_k"] % 2 == 1 else params["blur_k"] + 1
    if k_blur < 1:
        k_blur = 1
    img_blur = cv2.GaussianBlur(img, (k_blur, k_blur), 0)

    # 2. Silueta Total
    b, g, r = cv2.split(img_blur)
    _, mask_total = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k_clean = params["clean_k"]
    if k_clean > 0:
        mask_total = cv2.morphologyEx(
            mask_total, cv2.MORPH_OPEN, np.ones((k_clean, k_clean), np.uint8)
        )

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. Semillas
    mask_semillas = detectar_mascara_semillas(hsv, mask_total, params)

    # 4. Calibración. Va antes de las raíces porque la regla blanca, al ser
    #    brillante y desaturada, cae dentro de la máscara de radículas y hay que
    #    borrarla explícitamente para no medirla como si fuera una raíz.
    if referencia_fija is not None:
        px_por_mm, ref_encontrada, ref_origen, rect_regla = referencia_fija
        if img_debug is not None and ref_encontrada:
            dibujar_etiqueta_referencia(img_debug, f"{ref_origen} (fija)")
            if rect_regla is not None:
                box = np.int64(cv2.boxPoints(rect_regla))
                cv2.drawContours(img_debug, [box], 0, (0, 255, 255), 3)
    else:
        px_por_mm, ref_encontrada, ref_origen, rect_regla = detectar_referencia(
            hsv, img_debug, params, img_original=img, px_por_mm_qr=px_por_mm_qr
        )
    px_por_cm = px_por_mm * 10.0

    # 5. Raíces
    mask_radiculas = cv2.subtract(mask_total, mask_semillas)

    if rect_regla is not None:
        (cx, cy), (rw, rh), angulo = rect_regla
        rect_dilatado = (
            (cx, cy),
            (rw + 2 * REGLA_BLANCA_DILATACION_PX, rh + 2 * REGLA_BLANCA_DILATACION_PX),
            angulo,
        )
        box_regla = np.int64(cv2.boxPoints(rect_dilatado))
        cv2.drawContours(mask_radiculas, [box_regla], 0, 0, -1)

    k_bridge = (
        params["root_bridge"]
        if params["root_bridge"] % 2 == 1
        else params["root_bridge"] + 1
    )
    if k_bridge > 1:
        kernel_connect = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_bridge, k_bridge)
        )
        mask_radiculas = cv2.morphologyEx(
            mask_radiculas, cv2.MORPH_CLOSE, kernel_connect
        )

    if k_clean > 0:
        mask_radiculas = cv2.morphologyEx(
            mask_radiculas, cv2.MORPH_OPEN, np.ones((k_clean, k_clean), np.uint8)
        )

    mask_radiculas, y_limite_qr = excluir_zona_qr_inferior(mask_radiculas)

    # Texto de escala
    if img_debug is not None and ref_encontrada:
        cv2.putText(
            img_debug,
            f"Scale: {px_por_cm:.1f} px/cm ({px_por_mm:.2f} px/mm)",
            (20, h_img - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
    elif img_debug is not None:
        cv2.putText(
            img_debug,
            "NO REF",
            (20, h_img - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    detalles_semillas = []
    detalles_radiculas = []

    # Procesar Semillas
    detalles_semillas = extraer_semillas_desde_mascara(
        mask_semillas, params["min_area_seed"]
    )
    if img_debug is not None:
        for semilla in detalles_semillas:
            cv2.drawContours(img_debug, [semilla["contour"]], -1, (255, 0, 0), -1)

    # Procesar Raíces
    cnts_rad, _ = cv2.findContours(
        mask_radiculas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for i, cnt in enumerate(cnts_rad):
        area = cv2.contourArea(cnt)
        if area > params["min_area_root"]:
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue

            solidez = float(area) / hull_area
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / h if h > 0 else 0

            if contorno_toca_zona_qr_inferior(y, h, y_limite_qr):
                continue
            if solidez > params["root_solidity_max"]:
                continue
            if aspect > params["root_aspect_max"] or aspect < params["root_aspect_min"]:
                continue

            largo_px = obtener_largo_esqueleto_contorno(cnt, (h_img, w_img))
            largo_mm = largo_px / px_por_mm
            if largo_mm > MAX_ROOT_LENGTH_MM:
                continue

            detalles_radiculas.append(
                {
                    "id": i,
                    "length_mm": largo_mm,
                    "area_cm2": area / (px_por_cm**2),
                    "contour": cnt,
                }
            )
            if img_debug is not None:
                cv2.drawContours(img_debug, [cnt], -1, (0, 255, 0), 2)

    # ========================================================
    #  VISUALIZACIÓN DE ESTADÍSTICAS EN TIEMPO REAL (UPDATE)
    # ========================================================
    cantidad_semillas = len(detalles_semillas)
    largos_encontrados = [r["length_mm"] for r in detalles_radiculas]

    if img_debug is not None:
        color_semillas = (255, 255, 0) if cantidad_semillas > 0 else (0, 0, 255)
        cv2.putText(
            img_debug,
            f"Semillas: {cantidad_semillas}",
            (20, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_semillas,
            2,
        )

    if img_debug is not None and largos_encontrados:
        val_min = min(largos_encontrados)
        val_max = max(largos_encontrados)
        cant = len(largos_encontrados)

        # Texto en Cyan (Azul+Verde)
        texto_stats = f"Raices: {cant} | Min: {val_min:.2f} mm | Max: {val_max:.2f} mm"
        cv2.putText(
            img_debug,
            texto_stats,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
    elif img_debug is not None:
        cv2.putText(
            img_debug,
            "Sin Raices",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    resultado = {
        "lista_semillas": detalles_semillas,
        "lista_radiculas": detalles_radiculas,
        "img_debug": img_debug,
        "ref_encontrada": ref_encontrada,
        "ref_origen": ref_origen,
        "px_por_mm": px_por_mm,
        "px_por_cm": px_por_cm,
    }

    if devolver_mascaras:
        # Rellenas con los contornos que pasaron TODOS los filtros, no la
        # máscara de color cruda: esto es lo comparable contra un pintado a
        # mano, que marca "esto es raíz/semilla", no "esto tiene este color".
        mask_semillas_pred = np.zeros((h_img, w_img), dtype=np.uint8)
        for semilla in detalles_semillas:
            cv2.drawContours(mask_semillas_pred, [semilla["contour"]], -1, 255, -1)

        mask_radiculas_pred = np.zeros((h_img, w_img), dtype=np.uint8)
        for radicula in detalles_radiculas:
            cv2.drawContours(mask_radiculas_pred, [radicula["contour"]], -1, 255, -1)

        resultado["mask_semillas_pred"] = mask_semillas_pred
        resultado["mask_radiculas_pred"] = mask_radiculas_pred

    return resultado


# ==========================================
# PANEL DE CONTROL (GUI)
# ==========================================


def nothing(x):
    pass


def abrir_panel_calibracion(ruta_primera_imagen, archivo_config=ARCHIVO_CONFIG):
    print("\n--- MODO CALIBRACIÓN ---")
    print("Ajusta los sliders. Presiona 'Q' para GUARDAR y procesar todo.")

    img_raw = leer_imagen(ruta_primera_imagen)
    if img_raw is None:
        print(
            f"[ALERTA] No se pudo leer la imagen para calibración: {ruta_primera_imagen}"
        )
        return

    img_raw = redimensionar_para_analisis(img_raw)

    img_crop = recortar_bordes_blancos(img_raw)

    h, w = img_crop.shape[:2]
    factor = 1.0
    if h > 900:
        factor = 900 / h
        img_crop = cv2.resize(img_crop, (0, 0), fx=factor, fy=factor)

    px_por_mm_qr, _ = detectar_referencia_qr(img_crop)
    win_name = "Panel de Calibracion"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Trackbars
    cv2.createTrackbar("Suavizado (Blur)", win_name, PARAMS["blur_k"], 15, nothing)
    cv2.createTrackbar("Hue Max Semilla", win_name, PARAMS["seed_h_max"], 60, nothing)
    cv2.createTrackbar("Sat Min Semilla", win_name, PARAMS["seed_s_min"], 255, nothing)
    cv2.createTrackbar("Val Min Semilla", win_name, PARAMS["seed_v_min"], 255, nothing)
    cv2.createTrackbar("Limpieza Ruido", win_name, PARAMS["clean_k"], 10, nothing)
    cv2.createTrackbar("Pegamento Semilla", win_name, PARAMS["seed_glue"], 20, nothing)
    cv2.createTrackbar("Sat Max Raiz", win_name, PARAMS["root_s_max"], 255, nothing)
    cv2.createTrackbar("Val Min Raiz", win_name, PARAMS["root_v_min"], 255, nothing)
    cv2.createTrackbar("Area Min Sem", win_name, PARAMS["min_area_seed"], 1000, nothing)
    cv2.createTrackbar("Puente Raices", win_name, PARAMS["root_bridge"], 10, nothing)
    cv2.createTrackbar("Area Min Raiz", win_name, PARAMS["min_area_root"], 200, nothing)
    cv2.createTrackbar("Ref Amarillo Min", win_name, PARAMS["hsv_h_min"], 179, nothing)

    ultimos_valores = None
    ultima_vista = None

    while True:
        # Leer valores
        valores_actuales = {
            "blur_k": cv2.getTrackbarPos("Suavizado (Blur)", win_name),
            "seed_h_max": cv2.getTrackbarPos("Hue Max Semilla", win_name),
            "seed_s_min": cv2.getTrackbarPos("Sat Min Semilla", win_name),
            "seed_v_min": cv2.getTrackbarPos("Val Min Semilla", win_name),
            "clean_k": cv2.getTrackbarPos("Limpieza Ruido", win_name),
            "seed_glue": cv2.getTrackbarPos("Pegamento Semilla", win_name),
            "root_s_max": cv2.getTrackbarPos("Sat Max Raiz", win_name),
            "root_v_min": cv2.getTrackbarPos("Val Min Raiz", win_name),
            "min_area_seed": cv2.getTrackbarPos("Area Min Sem", win_name),
            "root_bridge": cv2.getTrackbarPos("Puente Raices", win_name),
            "min_area_root": cv2.getTrackbarPos("Area Min Raiz", win_name),
            "hsv_h_min": cv2.getTrackbarPos("Ref Amarillo Min", win_name),
        }

        if valores_actuales != ultimos_valores or ultima_vista is None:
            PARAMS.update(valores_actuales)
            res = procesar_imagen(
                img_crop,
                PARAMS,
                px_por_mm_qr=px_por_mm_qr,
                generar_debug=True,
            )
            ultima_vista = res["img_debug"]
            ultimos_valores = valores_actuales.copy()

        if ultima_vista is not None:
            cv2.imshow(win_name, ultima_vista)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break

        try:
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cv2.destroyAllWindows()
    guardar_configuracion(archivo_config)
    print("\n[OK] Procesando lote completo con los nuevos parámetros...")


# ==========================================
# MAIN
# ==========================================


def redimensionar_para_ver(imagen, alto_maximo):
    h, w = imagen.shape[:2]
    if h > alto_maximo:
        ratio = alto_maximo / float(h)
        return cv2.resize(imagen, (int(w * ratio), int(h * ratio)))
    return imagen


def seleccionar_carpetas(directorio_raiz):
    if not os.path.exists(directorio_raiz):
        return []
    todas = obtener_carpetas_disponibles(directorio_raiz)
    if not todas:
        return []
    print("\nCarpetas encontradas:")
    for i, n in enumerate(todas):
        print(f"[{i+1}] {n}")
    sel = input(
        "\nElige números de la lista o nombres de carpeta (separados por coma), "
        "o ENTER para todas: "
    )
    if not sel.strip():
        return todas

    # Acepta tanto el número de la lista ("16") como el nombre de la carpeta
    # tal cual ("20260911") — escribir la fecha es lo más natural y antes
    # quedaba silenciosamente interpretado como un índice fuera de rango,
    # dando una selección vacía sin ningún aviso.
    elegidas = []
    no_reconocidas = []
    for token in sel.split(","):
        token = token.strip()
        if not token:
            continue
        if token in todas:
            elegidas.append(token)
            continue
        try:
            idx = int(token) - 1
        except ValueError:
            no_reconocidas.append(token)
            continue
        if 0 <= idx < len(todas):
            elegidas.append(todas[idx])
        else:
            no_reconocidas.append(token)

    if no_reconocidas:
        print(
            f"[ALERTA] No reconocí: {', '.join(no_reconocidas)} "
            "(ni como número de la lista ni como nombre de carpeta). Se ignoran."
        )

    # Preservar el orden de 'todas', sin duplicados, por si el usuario repitió
    # una carpeta por número y por nombre a la vez.
    vistas = set()
    resultado = []
    for carpeta in todas:
        if carpeta in elegidas and carpeta not in vistas:
            resultado.append(carpeta)
            vistas.add(carpeta)

    if not resultado:
        print("[ALERTA] Ninguna carpeta válida seleccionada.")
    return resultado


def obtener_carpetas_disponibles(directorio_raiz=DIRECTORIO_RAIZ):
    if not os.path.exists(directorio_raiz):
        return []
    return sorted(
        [
            d
            for d in os.listdir(directorio_raiz)
            if os.path.isdir(os.path.join(directorio_raiz, d))
        ]
    )


def obtener_directorio_fotos(ruta_experimento):
    ruta_experimento = Path(ruta_experimento)
    ruta_fotos = ruta_experimento / "raw" / "photos"
    if ruta_fotos.is_dir():
        return str(ruta_fotos)
    return str(ruta_experimento)


def obtener_archivos_imagen(ruta_carpeta):
    patrones = (
        "*.jpg",
        "*.JPG",
        "*.jpeg",
        "*.JPEG",
        "*.png",
        "*.PNG",
        "*.heic",
        "*.HEIC",
        "*.heif",
        "*.HEIF",
    )
    archivos = []
    for patron in patrones:
        archivos.extend(glob.glob(os.path.join(ruta_carpeta, patron)))
    return sorted(set(archivos))


PISOS_BANDEJA = ["A", "B", "C"]
POSICIONES_BANDEJA = list(range(1, 10))


def clave_orden_natural(nombre):
    """IMG_999 antes que IMG_1000 (el orden alfabético puro los invierte)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", nombre)]


def inferir_tray_ids(carpeta, secuencia):
    """Completa los tray_id de las fotos sin QR legible usando el orden de toma.

    secuencia: lista de (nombre_foto, tray_id_o_None) en orden de toma.
    Las fotos se sacan siempre recorriendo la torre en un orden fijo: pisos
    A->B->C o C->B->A, y dentro de cada piso posiciones 1->9 o 9->1. Con los QR
    que sí se leyeron se detecta cuál de esos 4 órdenes se usó, y cada hueco se
    completa SOLO si la cantidad de fotos sin QR entre dos QR leídos coincide
    exactamente con la cantidad de posiciones que faltan entre ellos. Si algo
    no cierra (foto repetida, salteada, QR de otra carpeta), no se asigna nada
    en ese tramo y se avisa: preferible una foto sin tray que una mal asignada.

    Devuelve (asignaciones {nombre: tray_id}, lista de advertencias).
    """
    advertencias = []
    patron = re.compile(rf"^{re.escape(carpeta)}_([ABC])([1-9])$")

    anclas = []  # (indice_en_secuencia, piso, posicion)
    for i, (nombre, tray_id) in enumerate(secuencia):
        if tray_id is None:
            continue
        m = patron.match(tray_id)
        if m:
            anclas.append((i, m.group(1), int(m.group(2))))

    if len(anclas) < 2:
        if any(t is None for _, t in secuencia):
            advertencias.append(
                "Menos de 2 QR legibles: no se puede deducir el orden de toma."
            )
        return {}, advertencias

    ordenes_validos = []
    for pisos in (PISOS_BANDEJA, PISOS_BANDEJA[::-1]):
        for posiciones in (POSICIONES_BANDEJA, POSICIONES_BANDEJA[::-1]):
            orden = [f"{p}{n}" for p in pisos for n in posiciones]
            indices = [orden.index(f"{piso}{pos}") for _, piso, pos in anclas]
            if all(b > a for a, b in zip(indices, indices[1:])):
                ordenes_validos.append((orden, indices))

    if len(ordenes_validos) != 1:
        advertencias.append(
            "Los QR leídos no siguen un orden de toma único y consistente "
            f"({len(ordenes_validos)} órdenes posibles). No se infiere nada."
        )
        return {}, advertencias

    orden, indices = ordenes_validos[0]
    usados = {f"{piso}{pos}" for _, piso, pos in anclas}
    asignaciones = {}

    # Tramos: (inicio en secuencia, fin, primera posición libre, última posición libre)
    tramos = []
    primera_i, ultima_i = anclas[0][0], anclas[-1][0]
    tramos.append((0, primera_i, 0, indices[0]))
    for (i_a, _, _), (i_b, _, _), k_a, k_b in zip(
        anclas, anclas[1:], indices, indices[1:]
    ):
        tramos.append((i_a + 1, i_b, k_a + 1, k_b))
    tramos.append((ultima_i + 1, len(secuencia), indices[-1] + 1, len(orden)))

    for ini, fin, pos_ini, pos_fin in tramos:
        huecos = list(range(ini, fin))
        if not huecos:
            continue
        faltantes = orden[pos_ini:pos_fin]
        nombres = [secuencia[i][0] for i in huecos]
        if len(faltantes) != len(huecos):
            advertencias.append(
                f"No cierra el tramo {', '.join(nombres)}: {len(huecos)} foto(s) "
                f"sin QR para {len(faltantes)} posición(es) libre(s). Revisar a mano."
            )
            continue
        for i, codigo in zip(huecos, faltantes):
            if secuencia[i][1] is not None:
                # QR leído pero de otra carpeta: nunca lo pisamos.
                continue
            if codigo in usados:
                advertencias.append(
                    f"{secuencia[i][0]}: {codigo} ya está usado por otra foto. Se omite."
                )
                continue
            asignaciones[secuencia[i][0]] = f"{carpeta}_{codigo}"
            usados.add(codigo)

    return asignaciones, advertencias


def resolver_archivo_config_exp(ruta_experimento, archivo_config):
    """<experimento>/processed/parametros_calibracion.json si existe (por
    ejemplo, el que escribe OptimizarParametros.py), si no el config que
    haya pasado el caller (por defecto el global de src/).

    Antes 'main()' siempre usaba el config global sin importar qué carpetas
    elegías, así que un experimento con su propio calibración.json (tunéado
    a mano o por OptimizarParametros.py) quedaba completamente ignorado al
    correr Analizar.py desde la terminal — solo PipelineGUI lo respetaba.
    """
    ruta_exp_config = Path(ruta_experimento) / "processed" / "parametros_calibracion.json"
    if ruta_exp_config.exists():
        return str(ruta_exp_config)
    return archivo_config


def resolver_archivo_salida_exp(ruta_experimento, archivo_salida):
    ruta_experimento = Path(ruta_experimento)
    processed_dir = ruta_experimento / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if archivo_salida:
        ruta = Path(archivo_salida)
        if ruta.is_absolute() or len(ruta.parts) > 1:
            return str(ruta)
        return str(processed_dir / ruta)

    return str(processed_dir / ARCHIVO_SALIDA)


COLUMNAS_CSV = ["tray_id", "seed_id", "length_mm", "area_cm2"]
COLUMNAS_CSV_CRUDO = COLUMNAS_CSV + ["source_photo"]
COLUMNAS_CSV_FOTOS = [
    "source_photo",
    "tray_id",
    "tray_origen",  # "QR" o "INFERIDO" (por orden de toma, ver inferir_tray_ids)
    "n_semillas",
    "n_radiculas",
    "ref_origen",
    "px_por_mm",
]


def resolver_archivo_crudo_exp(archivo_salida_exp):
    """El caché crudo vive siempre al lado de root_lengths.csv."""
    return str(Path(archivo_salida_exp).with_name(ARCHIVO_SALIDA_CRUDO))


def resolver_archivo_fotos_exp(archivo_salida_exp):
    return str(Path(archivo_salida_exp).with_name(ARCHIVO_FOTOS_PROCESADAS))


def _leer_csv_cache(ruta, columnas, columna_clave):
    ruta = Path(ruta)
    if not ruta.exists():
        return pd.DataFrame(columns=columnas)

    try:
        df = pd.read_csv(ruta)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f"[ALERTA] No se pudo leer el caché {ruta} ({exc}). Se reprocesa todo.")
        return pd.DataFrame(columns=columnas)

    if columna_clave not in df.columns:
        print(f"[ALERTA] {ruta} no tiene columna {columna_clave}. Se reprocesa todo.")
        return pd.DataFrame(columns=columnas)

    for columna in columnas:
        if columna not in df.columns:
            df[columna] = None

    return df[columnas]


def leer_cache_crudo(archivo_crudo):
    """Lee las mediciones ya hechas, sin limpiar outliers.

    Es la fuente de verdad del procesamiento incremental: root_lengths.csv es
    siempre un derivado de este archivo. Guardar el crudo evita que la limpieza
    por IQR se aplique dos veces sobre las mismas filas.
    """
    return _leer_csv_cache(archivo_crudo, COLUMNAS_CSV_CRUDO, "source_photo")


def leer_cache_fotos(archivo_fotos):
    """Una fila por foto ya medida con éxito.

    Hace falta además del caché crudo porque una foto sin radículas detectadas no
    aporta ninguna fila a aquel, y sin este registro se reprocesaría para siempre.
    También conserva el conteo de semillas, que el resumen necesita y que no se
    puede reconstruir a partir de las filas de radículas.
    """
    return _leer_csv_cache(archivo_fotos, COLUMNAS_CSV_FOTOS, "source_photo")


def contadores_semillas_desde_cache(df_crudo):
    """Retoma la numeración seed_NNN de cada tray donde quedó.

    Sin esto, una corrida incremental volvería a emitir seed_001 y pisaría las
    semillas ya medidas de ese mismo tray.
    """
    contadores = {}
    for tray_id, seed_id in zip(df_crudo["tray_id"], df_crudo["seed_id"]):
        match = re.search(r"(\d+)$", str(seed_id))
        if not match:
            continue
        numero = int(match.group(1))
        if numero > contadores.get(tray_id, 0):
            contadores[tray_id] = numero
    return contadores


def resolver_carpeta_testigos_exp(ruta_experimento, carpeta_testigos):
    ruta_experimento = Path(ruta_experimento)
    processed_dir = ruta_experimento / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if carpeta_testigos:
        ruta = Path(carpeta_testigos)
        if ruta.is_absolute() or len(ruta.parts) > 1:
            return str(ruta)
        return str(processed_dir / ruta)

    return str(processed_dir / CARPETA_TESTIGOS)


def calcular_resumen_mediciones(filas_csv, semillas_detectadas):
    total_raices = len(filas_csv)
    media_largo_raiz_mm = float("nan")

    if filas_csv:
        largos = [
            float(fila["length_mm"])
            for fila in filas_csv
            if fila.get("length_mm") is not None
        ]
        if largos:
            media_largo_raiz_mm = float(np.mean(largos))

    return {
        "total_semillas": int(semillas_detectadas),
        "total_raices": int(total_raices),
        "media_largo_raiz_mm": media_largo_raiz_mm,
    }


def limpiar_outliers_lejanos(
    df,
    *,
    columna="length_mm",
    factor_iqr=6.0,
):
    if df.empty or columna not in df.columns:
        return df.copy(), pd.DataFrame(columns=df.columns), None

    df_limpio = df.copy()
    df_limpio[columna] = pd.to_numeric(df_limpio[columna], errors="coerce")
    valores = df_limpio[columna].dropna()
    if valores.empty:
        return df_limpio, pd.DataFrame(columns=df.columns), None

    q1 = float(valores.quantile(0.25))
    q3 = float(valores.quantile(0.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return df_limpio, pd.DataFrame(columns=df.columns), None

    umbral_superior = q3 + (factor_iqr * iqr)
    mask_outliers = df_limpio[columna] > umbral_superior
    outliers = df_limpio.loc[mask_outliers].copy()
    filtrado = df_limpio.loc[~mask_outliers].copy()

    return filtrado, outliers, umbral_superior


def imprimir_resumen_outliers(outliers, umbral_superior, *, prefijo=""):
    cantidad = len(outliers)
    if cantidad == 0 or umbral_superior is None:
        print(f"{prefijo}Outliers lejanos removidos: 0")
        return

    print(
        f"{prefijo}Outliers lejanos removidos: {cantidad} "
        f"(umbral superior length_mm > {umbral_superior:.2f})"
    )
    top_outliers = outliers.sort_values("length_mm", ascending=False).head(10)
    for _, fila in top_outliers.iterrows():
        tray_id = fila.get("tray_id", "sin_tray")
        seed_id = fila.get("seed_id", "sin_seed")
        length_mm = fila.get("length_mm", float("nan"))
        print(f"{prefijo}  - {tray_id} / {seed_id}: {length_mm:.2f} mm")


def imprimir_resumen_mediciones(resumen, *, prefijo=""):
    media = resumen["media_largo_raiz_mm"]
    media_texto = f"{media:.2f} mm" if not np.isnan(media) else "sin datos"
    print(f"{prefijo}Semillas detectadas: {resumen['total_semillas']}")
    print(f"{prefijo}Raices detectadas: {resumen['total_raices']}")
    print(f"{prefijo}Media largo de raiz: {media_texto}")


def procesar_carpetas(
    carpetas,
    directorio_raiz=DIRECTORIO_RAIZ,
    archivo_salida=None,
    carpeta_testigos=None,
    abrir_calibracion=False,
    archivo_config=ARCHIVO_CONFIG,
    guardar_imagenes_testigo=GUARDAR_IMAGENES_TESTIGO,
    reprocesar=False,
):
    if not carpetas:
        print("[ALERTA] No hay carpetas seleccionadas para procesar.")
        return None

    if guardar_imagenes_testigo:
        for carpeta in carpetas:
            ruta_exp = os.path.join(directorio_raiz, carpeta)
            os.makedirs(
                resolver_carpeta_testigos_exp(ruta_exp, carpeta_testigos),
                exist_ok=True,
            )

    imagen_ejemplo = None
    carpeta_ejemplo = None
    for carpeta in carpetas:
        ruta_experimento = os.path.join(directorio_raiz, carpeta)
        ruta_fotos = obtener_directorio_fotos(ruta_experimento)
        archivos = obtener_archivos_imagen(ruta_fotos)
        if archivos:
            imagen_ejemplo = archivos[0]
            carpeta_ejemplo = carpeta
            break

    if abrir_calibracion and imagen_ejemplo:
        ruta_exp_ejemplo = os.path.join(directorio_raiz, carpeta_ejemplo)
        archivo_config_ejemplo = resolver_archivo_config_exp(ruta_exp_ejemplo, archivo_config)
        # abrir_panel_calibracion() no carga el config solo: inicializa los
        # sliders con lo que YA esté en el PARAMS global en este momento. Si
        # no lo cargamos acá antes de abrirlo, el panel arranca desde los
        # valores de fábrica (no los tuneados de esta carpeta) y, si se
        # cierra con 'q', los GUARDA pisando el config bueno con esos
        # valores de fábrica.
        PARAMS.clear()
        PARAMS.update(PARAMS_POR_DEFECTO)
        cargar_configuracion(archivo_config_ejemplo)
        abrir_panel_calibracion(imagen_ejemplo, archivo_config=archivo_config_ejemplo)
    elif abrir_calibracion:
        print("[ALERTA] No se encontraron imágenes para calibrar.")

    resultados = []
    columnas_csv = COLUMNAS_CSV
    total_semillas_global = 0
    filas_csv_global = []

    for carpeta in carpetas:
        ruta_experimento = os.path.join(directorio_raiz, carpeta)
        ruta_fotos = obtener_directorio_fotos(ruta_experimento)
        archivos = obtener_archivos_imagen(ruta_fotos)

        # Cada experimento puede tener su propio calibración.json (el que
        # escribe OptimizarParametros.py, o uno tuneado a mano). Se resuelve
        # y se carga acá, DENTRO del loop, para que un batch con varias
        # carpetas no termine usando el config de una sola para todas.
        PARAMS.clear()
        PARAMS.update(PARAMS_POR_DEFECTO)
        cargar_configuracion(resolver_archivo_config_exp(ruta_experimento, archivo_config))

        archivo_salida_exp = resolver_archivo_salida_exp(
            ruta_experimento, archivo_salida
        )
        archivo_crudo_exp = resolver_archivo_crudo_exp(archivo_salida_exp)
        archivo_fotos_exp = resolver_archivo_fotos_exp(archivo_salida_exp)

        if reprocesar:
            df_crudo_previo = pd.DataFrame(columns=COLUMNAS_CSV_CRUDO)
            df_fotos_previo = pd.DataFrame(columns=COLUMNAS_CSV_FOTOS)
        else:
            df_crudo_previo = leer_cache_crudo(archivo_crudo_exp)
            df_fotos_previo = leer_cache_fotos(archivo_fotos_exp)

        fotos_hechas = set(df_fotos_previo["source_photo"].astype(str))
        # Solo se conservan las mediciones de fotos que siguen registradas como
        # hechas; así, si se borra el registro de una foto, sus filas viejas no
        # quedan huérfanas y duplicadas cuando se la vuelva a medir.
        if not df_crudo_previo.empty:
            df_crudo_previo = df_crudo_previo[
                df_crudo_previo["source_photo"].astype(str).isin(fotos_hechas)
            ]

        filas_csv = list(df_crudo_previo.to_dict("records"))
        filas_fotos = list(df_fotos_previo.to_dict("records"))
        fotos_sin_qr = []
        fotos_sin_referencia = []
        fotos_no_legibles = []
        seed_counters = contadores_semillas_desde_cache(df_crudo_previo)
        medidas = []
        semillas_detectadas_exp = int(
            pd.to_numeric(df_fotos_previo["n_semillas"], errors="coerce")
            .fillna(0)
            .sum()
        )

        pendientes = [
            ruta for ruta in archivos if os.path.basename(ruta) not in fotos_hechas
        ]
        salteadas = len(archivos) - len(pendientes)

        print(f"\nProcesando {carpeta} ({len(archivos)} imágenes)...")
        if salteadas:
            print(f"  [INFO] {salteadas} fotos ya medidas, se saltean.")
        if not pendientes:
            print("  [INFO] No hay fotos nuevas para medir.")

        for idx_archivo, ruta_img in enumerate(pendientes, start=1):
            imagen = leer_imagen(ruta_img)
            if imagen is None:
                print(f"[ALERTA] No se pudo leer la imagen: {ruta_img}")
                fotos_no_legibles.append(os.path.basename(ruta_img))
                continue

            imagen = redimensionar_para_analisis(imagen)
            imagen = recortar_bordes_blancos(imagen)
            if imagen is None or imagen.size == 0:
                print(f"[ALERTA] No se pudo recortar el borde blanco en: {ruta_img}")
                fotos_no_legibles.append(os.path.basename(ruta_img))
                continue

            nombre_archivo = os.path.basename(ruta_img)
            print(f"  [{idx_archivo}/{len(pendientes)}] {nombre_archivo}")
            tray_id, px_por_mm_qr = detectar_info_qr(imagen, carpeta)
            res = procesar_imagen(
                imagen,
                PARAMS,
                px_por_mm_qr=px_por_mm_qr,
                generar_debug=guardar_imagenes_testigo,
            )
            if not res:
                continue

            if res["img_debug"] is not None:
                cv2.putText(
                    res["img_debug"],
                    nombre_archivo,
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )
                if tray_id:
                    dibujar_etiqueta_tray(res["img_debug"], tray_id, inferido=False)

            if not res["ref_encontrada"]:
                print(
                    f"[ALERTA] No se encontró la referencia de escala en {nombre_archivo}. "
                    "Se omite."
                )
                fotos_sin_referencia.append(nombre_archivo)

            ruta_testigo = None
            if res["img_debug"] is not None and guardar_imagenes_testigo:
                carpeta_testigos_exp = resolver_carpeta_testigos_exp(
                    ruta_experimento, carpeta_testigos
                )
                ruta_testigo = os.path.join(
                    carpeta_testigos_exp, f"proc_{Path(nombre_archivo).stem}.png"
                )
                cv2.imwrite(ruta_testigo, res["img_debug"])

            # Todavía no se escriben filas: el tray de las fotos sin QR se
            # decide recién al final, cuando se conoce el orden completo.
            # Se guarda solo lo necesario (no la imagen) para no llenar memoria.
            medidas.append(
                {
                    "nombre": nombre_archivo,
                    "tray_qr": tray_id,
                    "ref_encontrada": res["ref_encontrada"],
                    "ref_origen": res["ref_origen"],
                    "px_por_mm": res["px_por_mm"],
                    "n_semillas": len(res["lista_semillas"]),
                    "radiculas": [
                        (r["length_mm"], r["area_cm2"]) for r in res["lista_radiculas"]
                    ],
                    "ruta_testigo": ruta_testigo,
                }
            )

        # Inferencia de tray para las fotos sin QR legible, a partir del orden
        # de toma. Las fotos ya medidas en corridas anteriores también cuentan
        # como anclas (su tray está en el caché).
        tray_cache = dict(
            zip(
                df_fotos_previo["source_photo"].astype(str),
                df_fotos_previo["tray_id"],
            )
        )
        tray_qr_nuevo = {m["nombre"]: m["tray_qr"] for m in medidas}
        secuencia = []
        for ruta in sorted(
            archivos, key=lambda r: clave_orden_natural(os.path.basename(r))
        ):
            nombre = os.path.basename(ruta)
            secuencia.append(
                (nombre, tray_cache.get(nombre) or tray_qr_nuevo.get(nombre))
            )

        inferidos, advertencias = inferir_tray_ids(carpeta, secuencia)
        for adv in advertencias:
            print(f"[ALERTA] Inferencia de tray: {adv}")
        if inferidos:
            print(f"  [INFO] {len(inferidos)} tray(s) inferido(s) por orden de toma:")
            for nombre, tray in sorted(
                inferidos.items(), key=lambda kv: clave_orden_natural(kv[0])
            ):
                print(f"    {nombre} -> {tray}")

        for m in medidas:
            nombre_archivo = m["nombre"]
            tray_id = m["tray_qr"]
            tray_origen = "QR"
            if tray_id is None and nombre_archivo in inferidos:
                tray_id = inferidos[nombre_archivo]
                tray_origen = "INFERIDO"

            if m["ruta_testigo"] and m["tray_qr"] is None:
                testigo = cv2.imread(m["ruta_testigo"])
                if testigo is not None:
                    if tray_id:
                        dibujar_etiqueta_tray(testigo, tray_id, inferido=True)
                    else:
                        cv2.putText(
                            testigo, "NO QR", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                        )
                    cv2.imwrite(m["ruta_testigo"], testigo)

            if tray_id is None:
                fotos_sin_qr.append(nombre_archivo)
                continue
            if not m["ref_encontrada"]:
                continue

            if not tray_id.startswith(f"{carpeta}_"):
                print(
                    f"[ALERTA] El QR de {nombre_archivo} apunta a {tray_id}, "
                    f"pero la foto está en {carpeta}."
                )

            semillas_detectadas_exp += m["n_semillas"]
            contador_actual = seed_counters.get(tray_id, 0)
            for length_mm, area_cm2 in m["radiculas"]:
                contador_actual += 1
                filas_csv.append(
                    {
                        "tray_id": tray_id,
                        "seed_id": f"seed_{contador_actual:03d}",
                        "length_mm": length_mm,
                        "area_cm2": area_cm2,
                        "source_photo": nombre_archivo,
                    }
                )
            seed_counters[tray_id] = contador_actual

            # Solo las fotos con tray (leído o inferido) y referencia entran al
            # registro; las demás se reintentan en la próxima corrida.
            filas_fotos.append(
                {
                    "source_photo": nombre_archivo,
                    "tray_id": tray_id,
                    "tray_origen": tray_origen,
                    "n_semillas": m["n_semillas"],
                    "n_radiculas": len(m["radiculas"]),
                    "ref_origen": m["ref_origen"],
                    "px_por_mm": m["px_por_mm"],
                }
            )

        carpeta_salida = os.path.dirname(archivo_salida_exp)
        if carpeta_salida:
            os.makedirs(carpeta_salida, exist_ok=True)

        # El crudo es la fuente de verdad del incremental; root_lengths.csv es
        # siempre un derivado, recalculado sobre el conjunto completo para que la
        # limpieza de outliers no se aplique dos veces sobre las mismas filas.
        # Se ordena para que el resultado no dependa de en cuántas tandas se
        # midieron las fotos: una corrida completa y dos incrementales dan el
        # mismo archivo.
        df_crudo = pd.DataFrame(filas_csv, columns=COLUMNAS_CSV_CRUDO).sort_values(
            ["tray_id", "seed_id"], kind="stable", ignore_index=True
        )
        df_crudo.to_csv(archivo_crudo_exp, index=False, float_format=FLOAT_FORMAT_CSV)
        pd.DataFrame(filas_fotos, columns=COLUMNAS_CSV_FOTOS).sort_values(
            "source_photo", kind="stable", ignore_index=True
        ).to_csv(archivo_fotos_exp, index=False, float_format=FLOAT_FORMAT_CSV)

        df = df_crudo[columnas_csv]
        df_limpio, df_outliers, umbral_outliers = limpiar_outliers_lejanos(df)
        df_limpio.to_csv(archivo_salida_exp, index=False, float_format=FLOAT_FORMAT_CSV)
        print(f"\nListo. Datos guardados en {archivo_salida_exp}")
        imprimir_resumen_outliers(df_outliers, umbral_outliers, prefijo="  ")

        filas_csv_limpias = df_limpio.to_dict("records")
        resumen_exp = calcular_resumen_mediciones(
            filas_csv_limpias, semillas_detectadas_exp
        )
        print("Resumen del experimento:")
        imprimir_resumen_mediciones(resumen_exp, prefijo="  ")

        if filas_fotos:
            conteo_ref = {}
            for fila in filas_fotos:
                origen = fila.get("ref_origen") or "?"
                conteo_ref[origen] = conteo_ref.get(origen, 0) + 1
            detalle_ref = ", ".join(
                f"{origen}: {cantidad}"
                for origen, cantidad in sorted(conteo_ref.items())
            )
            print(f"  Referencia de escala usada -> {detalle_ref}")

        if fotos_sin_qr:
            print(f"[INFO] Fotos omitidas por QR no legible: {', '.join(fotos_sin_qr)}")
        if fotos_sin_referencia:
            print(
                "[INFO] Fotos omitidas por referencia no detectada: "
                + ", ".join(fotos_sin_referencia)
            )
        if fotos_no_legibles:
            print(f"[INFO] Fotos no legibles: {', '.join(fotos_no_legibles)}")

        total_semillas_global += resumen_exp["total_semillas"]
        filas_csv_global.extend(filas_csv_limpias)
        resultados.append(
            {
                "exp_id": carpeta,
                "archivo_salida": archivo_salida_exp,
                "cantidad_filas": len(df_limpio),
                "cantidad_outliers_removidos": int(len(df_outliers)),
                "resumen_mediciones": resumen_exp,
                "carpeta_testigos": (
                    resolver_carpeta_testigos_exp(ruta_experimento, carpeta_testigos)
                    if guardar_imagenes_testigo
                    else None
                ),
            }
        )

    if not resultados:
        print("[ALERTA] No se generaron mediciones.")
        return None

    filas_totales = sum(r["cantidad_filas"] for r in resultados)
    resumen_global = calcular_resumen_mediciones(
        filas_csv_global, total_semillas_global
    )
    print("\nResumen global:")
    imprimir_resumen_mediciones(resumen_global, prefijo="  ")

    return {
        "carpetas_procesadas": list(carpetas),
        "experimentos": resultados,
        "cantidad_filas_total": filas_totales,
        "resumen_mediciones": resumen_global,
    }


def main():
    # Por defecto solo se miden las fotos nuevas. Con --reprocesar se vuelve a
    # medir todo desde cero, descartando el caché.
    reprocesar = "--reprocesar" in sys.argv
    abrir_calibracion = "--sin-calibracion" not in sys.argv

    carpetas = seleccionar_carpetas(DIRECTORIO_RAIZ)
    if not carpetas:
        return

    procesar_carpetas(
        carpetas,
        directorio_raiz=DIRECTORIO_RAIZ,
        archivo_salida=ARCHIVO_SALIDA,
        carpeta_testigos=CARPETA_TESTIGOS,
        abrir_calibracion=abrir_calibracion,
        archivo_config=ARCHIVO_CONFIG,
        guardar_imagenes_testigo=GUARDAR_IMAGENES_TESTIGO,
        reprocesar=reprocesar,
    )
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
