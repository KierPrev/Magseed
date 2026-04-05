import cv2
import numpy as np
import pandas as pd
import os
import glob
import json
from skimage.morphology import skeletonize

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================
DIRECTORIO_RAIZ = "./Experimentos"
ARCHIVO_SALIDA = "resultados_detallados_calibrados.csv"
ARCHIVO_CONFIG = "parametros_calibracion.json"
GUARDAR_IMAGENES_TESTIGO = True
CARPETA_TESTIGOS = "./procesadas_debug"

# CORRECCIÓN IMPORTANTE: 11.09 mm = 1.109 cm
LARGO_REAL_REFERENCIA_CM = 11.09

# Valores por defecto
PARAMS = {
    "blur_k": 5,
    "seed_thresh": 5,
    "clean_k": 1,  # Default bajo
    "seed_glue": 7,
    "root_bridge": 3,
    "min_area_root": 30,  # Default bajo para raíces pequeñas
    "hsv_h_min": 15,
    "hsv_h_max": 40,
}

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
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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


def obtener_largo_esqueleto(img_binaria_radicula):
    bool_image = img_binaria_radicula > 0
    skeleton = skeletonize(bool_image)
    return np.count_nonzero(skeleton)


def detectar_referencia(hsv, img_debug, params):
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
        box = np.int64(cv2.boxPoints(rect))
        cv2.drawContours(img_debug, [box], 0, (255, 255, 0), 3)
        largo_px = max(rect[1])
        if largo_px > 0:
            return largo_px / LARGO_REAL_REFERENCIA_CM, True

    return 1.0, False


def procesar_imagen(img_original, params):
    img = img_original.copy()
    img_debug = img.copy()
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

    # 3. Semillas
    diff_r_g = cv2.subtract(r, g)
    _, mask_semillas_raw = cv2.threshold(
        diff_r_g, params["seed_thresh"], 255, cv2.THRESH_BINARY
    )

    k_glue = (
        params["seed_glue"] if params["seed_glue"] % 2 == 1 else params["seed_glue"] + 1
    )
    if k_glue < 1:
        k_glue = 1
    kernel_strong = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_glue, k_glue))
    mask_semillas_closed = cv2.morphologyEx(
        mask_semillas_raw, cv2.MORPH_CLOSE, kernel_strong, iterations=2
    )

    cnts_fill, _ = cv2.findContours(
        mask_semillas_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mask_semillas = np.zeros_like(mask_semillas_raw)
    for c in cnts_fill:
        if cv2.contourArea(c) > 50:
            cv2.drawContours(mask_semillas, [c], -1, 255, -1)

    mask_semillas = cv2.morphologyEx(
        mask_semillas, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    mask_semillas = cv2.bitwise_and(mask_total, mask_semillas)

    # 4. Raíces
    mask_radiculas = cv2.subtract(mask_total, mask_semillas)

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

    # 5. Calibración
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    px_por_cm, ref_encontrada = detectar_referencia(hsv, img_debug, params)

    # Texto de escala
    if ref_encontrada:
        cv2.putText(
            img_debug,
            f"Scale: {px_por_cm:.1f} px/cm",
            (20, h_img - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
    else:
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
    cnts_sem, _ = cv2.findContours(
        mask_semillas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for i, cnt in enumerate(cnts_sem):
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / h if h > 0 else 0
            if 0.3 < aspect < 4.0:
                detalles_semillas.append({"id": i, "area_px": area})
                cv2.drawContours(img_debug, [cnt], -1, (255, 0, 0), -1)

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

            if solidez > 0.6:
                continue
            if aspect > 8.0 or aspect < 0.12:
                continue

            mask_single = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.drawContours(mask_single, [cnt], -1, 255, -1)
            largo_px = obtener_largo_esqueleto(mask_single)

            detalles_radiculas.append(
                {"id": i, "largo_cm": largo_px / px_por_cm, "area_px": area}
            )
            cv2.drawContours(img_debug, [cnt], -1, (0, 255, 0), 2)

    # ========================================================
    #  VISUALIZACIÓN DE ESTADÍSTICAS EN TIEMPO REAL (UPDATE)
    # ========================================================
    largos_encontrados = [r["largo_cm"] for r in detalles_radiculas]

    if largos_encontrados:
        val_min = min(largos_encontrados)
        val_max = max(largos_encontrados)
        cant = len(largos_encontrados)

        # Texto en Cyan (Azul+Verde)
        texto_stats = f"Raices: {cant} | Min: {val_min:.2f} cm | Max: {val_max:.2f} cm"
        cv2.putText(
            img_debug,
            texto_stats,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
    else:
        cv2.putText(
            img_debug,
            "Sin Raices",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return {
        "lista_semillas": detalles_semillas,
        "lista_radiculas": detalles_radiculas,
        "img_debug": img_debug,
        "px_por_cm": px_por_cm,
    }


# ==========================================
# PANEL DE CONTROL (GUI)
# ==========================================


def nothing(x):
    pass


def abrir_panel_calibracion(ruta_primera_imagen, archivo_config=ARCHIVO_CONFIG):
    print("\n--- MODO CALIBRACIÓN ---")
    print("Ajusta los sliders. Presiona 'Q' para GUARDAR y procesar todo.")

    img_raw = cv2.imread(ruta_primera_imagen)
    if img_raw is None:
        return

    img_crop = recortar_bordes_blancos(img_raw)

    h, w = img_crop.shape[:2]
    factor = 1.0
    if h > 900:
        factor = 900 / h
        img_crop = cv2.resize(img_crop, (0, 0), fx=factor, fy=factor)

    win_name = "Panel de Calibracion"
    cv2.namedWindow(win_name)

    # Trackbars
    cv2.createTrackbar("Suavizado (Blur)", win_name, PARAMS["blur_k"], 15, nothing)
    cv2.createTrackbar("Umbral Semilla", win_name, PARAMS["seed_thresh"], 50, nothing)
    cv2.createTrackbar("Limpieza Ruido", win_name, PARAMS["clean_k"], 10, nothing)
    cv2.createTrackbar("Pegamento Semilla", win_name, PARAMS["seed_glue"], 20, nothing)
    cv2.createTrackbar("Puente Raices", win_name, PARAMS["root_bridge"], 10, nothing)
    cv2.createTrackbar("Area Min Raiz", win_name, PARAMS["min_area_root"], 200, nothing)
    cv2.createTrackbar("Ref Amarillo Min", win_name, PARAMS["hsv_h_min"], 179, nothing)

    while True:
        # Leer valores
        PARAMS["blur_k"] = cv2.getTrackbarPos("Suavizado (Blur)", win_name)
        PARAMS["seed_thresh"] = cv2.getTrackbarPos("Umbral Semilla", win_name)
        PARAMS["clean_k"] = cv2.getTrackbarPos("Limpieza Ruido", win_name)
        PARAMS["seed_glue"] = cv2.getTrackbarPos("Pegamento Semilla", win_name)
        PARAMS["root_bridge"] = cv2.getTrackbarPos("Puente Raices", win_name)
        PARAMS["min_area_root"] = cv2.getTrackbarPos("Area Min Raiz", win_name)
        PARAMS["hsv_h_min"] = cv2.getTrackbarPos("Ref Amarillo Min", win_name)

        res = procesar_imagen(img_crop, PARAMS)
        cv2.imshow(win_name, res["img_debug"])

        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
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
    sel = input("\nElige números (separados por coma) o ENTER para todas: ")
    if not sel.strip():
        return todas
    try:
        idxs = [int(x) - 1 for x in sel.split(",")]
        return [todas[i] for i in idxs if 0 <= i < len(todas)]
    except:
        return []


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


def obtener_archivos_imagen(ruta_carpeta):
    patrones = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
    archivos = []
    for patron in patrones:
        archivos.extend(glob.glob(os.path.join(ruta_carpeta, patron)))
    return sorted(set(archivos))


def procesar_carpetas(
    carpetas,
    directorio_raiz=DIRECTORIO_RAIZ,
    archivo_salida=ARCHIVO_SALIDA,
    carpeta_testigos=CARPETA_TESTIGOS,
    abrir_calibracion=False,
    archivo_config=ARCHIVO_CONFIG,
    guardar_imagenes_testigo=GUARDAR_IMAGENES_TESTIGO,
):
    cargar_configuracion(archivo_config)

    if not carpetas:
        print("[ALERTA] No hay carpetas seleccionadas para procesar.")
        return None

    if guardar_imagenes_testigo:
        os.makedirs(carpeta_testigos, exist_ok=True)

    imagen_ejemplo = None
    for carpeta in carpetas:
        ruta_carpeta = os.path.join(directorio_raiz, carpeta)
        archivos = obtener_archivos_imagen(ruta_carpeta)
        if archivos:
            imagen_ejemplo = archivos[0]
            break

    if abrir_calibracion and imagen_ejemplo:
        abrir_panel_calibracion(imagen_ejemplo, archivo_config=archivo_config)
    elif abrir_calibracion:
        print("[ALERTA] No se encontraron imágenes para calibrar.")

    filas_csv = []

    for carpeta in carpetas:
        ruta_carpeta = os.path.join(directorio_raiz, carpeta)
        archivos = obtener_archivos_imagen(ruta_carpeta)

        print(f"\nProcesando {carpeta} ({len(archivos)} imágenes)...")

        for ruta_img in archivos:
            imagen = cv2.imread(ruta_img)
            if imagen is None:
                print(f"[ALERTA] No se pudo leer la imagen: {ruta_img}")
                continue

            nombre_archivo = os.path.basename(ruta_img)
            nombre_clean = os.path.splitext(nombre_archivo)[0].lower()
            tipo_grupo = (
                "Control"
                if "control" in nombre_clean
                else "Magnetico" if "cm" in nombre_clean else "Desc"
            )
            replica = "".join(filter(str.isdigit, nombre_clean))
            if not replica:
                replica = "1"

            res = procesar_imagen(imagen, PARAMS)
            if not res:
                continue

            factor = res["px_por_cm"]
            base_info = {
                "Fecha": carpeta,
                "Grupo": tipo_grupo,
                "Replica": replica,
                "Archivo": nombre_archivo,
                "Escala_Usada": factor,
            }

            for s in res["lista_semillas"]:
                row = base_info.copy()
                row.update(
                    {
                        "Tipo_Objeto": "Semilla",
                        "ID_Objeto": s["id"],
                        "Medida_Tipo": "Area_cm2",
                        "Medida_Valor": s["area_px"] / (factor**2),
                    }
                )
                filas_csv.append(row)

            for r in res["lista_radiculas"]:
                row_l = base_info.copy()
                row_l.update(
                    {
                        "Tipo_Objeto": "Radicula",
                        "ID_Objeto": r["id"],
                        "Medida_Tipo": "Largo_cm",
                        "Medida_Valor": r["largo_cm"],
                    }
                )
                filas_csv.append(row_l)

                row_a = base_info.copy()
                row_a.update(
                    {
                        "Tipo_Objeto": "Radicula",
                        "ID_Objeto": r["id"],
                        "Medida_Tipo": "Area_cm2",
                        "Medida_Valor": r["area_px"] / (factor**2),
                    }
                )
                filas_csv.append(row_a)

            if res["img_debug"] is not None and guardar_imagenes_testigo:
                nombre_unico = f"proc_{carpeta}_{nombre_archivo}"
                cv2.imwrite(
                    os.path.join(carpeta_testigos, nombre_unico), res["img_debug"]
                )

    if not filas_csv:
        print("[ALERTA] No se generaron mediciones.")
        return None

    carpeta_salida = os.path.dirname(archivo_salida)
    if carpeta_salida:
        os.makedirs(carpeta_salida, exist_ok=True)

    df = pd.DataFrame(filas_csv)
    cols = [
        "Fecha",
        "Grupo",
        "Replica",
        "Archivo",
        "Tipo_Objeto",
        "ID_Objeto",
        "Medida_Tipo",
        "Medida_Valor",
        "Escala_Usada",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(archivo_salida, index=False)
    print(f"\nListo. Datos guardados en {archivo_salida}")

    return {
        "archivo_salida": archivo_salida,
        "carpetas_procesadas": list(carpetas),
        "cantidad_filas": len(df),
        "carpeta_testigos": carpeta_testigos if guardar_imagenes_testigo else None,
    }


def main():
    carpetas = seleccionar_carpetas(DIRECTORIO_RAIZ)
    if not carpetas:
        return

    procesar_carpetas(
        carpetas,
        directorio_raiz=DIRECTORIO_RAIZ,
        archivo_salida=ARCHIVO_SALIDA,
        carpeta_testigos=CARPETA_TESTIGOS,
        abrir_calibracion=True,
        archivo_config=ARCHIVO_CONFIG,
        guardar_imagenes_testigo=GUARDAR_IMAGENES_TESTIGO,
    )
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
