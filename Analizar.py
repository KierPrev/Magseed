import cv2
import numpy as np
import pandas as pd
import os
import glob

# ==========================================
# PANEL DE CONTROL
# ==========================================

# 1. Rutas
DIRECTORIO_RAIZ = './Experimentos' 
ARCHIVO_SALIDA = 'resultados_detallados_raw.csv' 

# 2. Calibración
PIXELES_POR_CM = 1.0  

# 3. Umbrales de Color (HSV)
RADICULA_MIN_HSV = np.array([30, 0, 0])    
RADICULA_MAX_HSV = np.array([60, 255, 225]) 

SEMILLA_MIN_HSV = np.array([0, 0, 53])
SEMILLA_MAX_HSV = np.array([15, 255, 255])

# 4. Filtros de Tamaño (ÁREA)
AREA_MIN_RADICULA = 200     
AREA_MAX_RADICULA = 50000   

AREA_MIN_SEMILLA = 800
AREA_MAX_SEMILLA = 50000

# 5. Filtro de Adyacencia
DISTANCIA_MAX_UNION = 20 

# 6. Visualización y Recorte
MOSTRAR_EN_PANTALLA = True 
ALTO_MAXIMO_PANTALLA = 800 
GUARDAR_IMAGENES_TESTIGO = True
CARPETA_TESTIGOS = './procesadas_debug'

# CONFIGURACIÓN DE RECORTE AUTOMÁTICO
# Brillo mínimo para considerar algo como "borde blanco" (0-255).
# 200 suele funcionar bien para cajas de luz blancas.
UMBRAL_BLANCO = 190 
MARGEN_SEGURIDAD = 30 # Píxeles extra para no cortar muy pegado al papel

# ==========================================
# FIN DEL PANEL DE CONTROL
# ==========================================

def recortar_bordes_blancos(img):
    """
    Detecta la zona oscura (papel madera) y elimina los bordes blancos sobrantes.
    """
    if img is None:
        return None

    h_img, w_img = img.shape[:2]

    # 1. Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Aplicar umbral: Todo lo muy claro se vuelve negro, lo oscuro se vuelve blanco
    # (THRESH_BINARY_INV invierte: claro->0, oscuro->255)
    _, thresh = cv2.threshold(gray, UMBRAL_BLANCO, 255, cv2.THRESH_BINARY_INV)

    # 3. Encontrar todos los puntos que NO son el fondo blanco
    puntos = cv2.findNonZero(thresh)

    if puntos is not None:
        # 4. Calcular el rectángulo que encierra esos puntos
        x, y, w, h = cv2.boundingRect(puntos)

        # 5. Aplicar un margen de seguridad (para no cortar justo en el borde del papel)
        x = max(0, x - MARGEN_SEGURIDAD)
        y = max(0, y - MARGEN_SEGURIDAD)
        w = min(w_img - x, w + (MARGEN_SEGURIDAD * 2))
        h = min(h_img - y, h + (MARGEN_SEGURIDAD * 2))

        # 6. Recortar
        img_recortada = img[y:y+h, x:x+w]
        return img_recortada
    
    # Si no encontró nada oscuro, devuelve la imagen original
    return img

def obtener_largo_esqueleto(img_binaria_radicula):
    size = np.size(img_binaria_radicula)
    skel = np.zeros(img_binaria_radicula.shape, np.uint8)
    
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    img_copy = img_binaria_radicula.copy()
    
    done = False
    while not done:
        eroded = cv2.erode(img_copy, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img_copy, temp)
        skel = cv2.bitwise_or(skel, temp)
        img_copy = eroded.copy()
        
        zeros = size - cv2.countNonZero(img_copy)
        if zeros == size:
            done = True
            
    largo_pixeles = cv2.countNonZero(skel)
    return largo_pixeles

def procesar_imagen(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    if img is None:
        return None

    # --- NUEVO: RECORTE AUTOMÁTICO ---
    # Esto elimina los bordes blancos antes de analizar nada
    img = recortar_bordes_blancos(img)
    # ---------------------------------

    h, w = img.shape[:2]
    
    # Reducir ruido
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # ============================================================
    # 1. DETECCIÓN (ESTRATEGIA LAB + CLAHE MEJORADA)
    # ============================================================
    lab = cv2.cvtColor(img_blur, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # --- Semillas (Canal A) ---
    a_clahe = clahe.apply(a_channel)
    _, mask_sem = cv2.threshold(a_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel_sem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_sem = cv2.morphologyEx(mask_sem, cv2.MORPH_CLOSE, kernel_sem, iterations=3)
    mask_sem = cv2.morphologyEx(mask_sem, cv2.MORPH_OPEN, kernel_sem, iterations=1)

    # --- Radículas (Canal L) ---
    l_clahe = clahe.apply(l_channel)
    mask_rad = cv2.adaptiveThreshold(l_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 21, -5)
    
    kernel_rad = np.ones((3,3), np.uint8)
    mask_rad = cv2.morphologyEx(mask_rad, cv2.MORPH_OPEN, kernel_rad, iterations=1)
    
    # Restar semillas de las raíces
    mask_rad = cv2.subtract(mask_rad, mask_sem)

    # Contornos
    cnt_rad, _ = cv2.findContours(mask_rad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_sem, _ = cv2.findContours(mask_sem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_debug = img.copy()

    # ============================================================
    # 2. PREPARACIÓN DE OBJETOS
    # ============================================================
    
    detalles_semillas = []
    detalles_radiculas = []
    
    mask_acumulada_validos = np.zeros((h, w), dtype=np.uint8)
    
    # Procesar SEMILLAS primero
    for i, cnt in enumerate(cnt_sem):
        area = cv2.contourArea(cnt)
        if AREA_MIN_SEMILLA < area < AREA_MAX_SEMILLA:
            detalles_semillas.append({
                'id': i,
                'area_px': area
            })
            cv2.drawContours(mask_acumulada_validos, [cnt], -1, 255, -1)
            cv2.drawContours(img_debug, [cnt], -1, (255, 0, 0), 2) # Azul

    # B. Lista de candidatas (Raíces filtradas por área)
    candidatos_radiculas = []
    kernel_dilatacion = np.ones((DISTANCIA_MAX_UNION, DISTANCIA_MAX_UNION), np.uint8)

    for i, cnt in enumerate(cnt_rad):
        area = cv2.contourArea(cnt)
        if AREA_MIN_RADICULA < area < AREA_MAX_RADICULA:
            mask_single = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask_single, [cnt], -1, 255, -1)
            mask_dilated = cv2.dilate(mask_single, kernel_dilatacion, iterations=1)
            
            candidatos_radiculas.append({
                'id': i,
                'cnt': cnt,
                'mask': mask_single,
                'mask_dilated': mask_dilated,
                'valid': False 
            })

    # ============================================================
    # 3. LÓGICA DE PROPAGACIÓN (Conectividad)
    # ============================================================
    hubo_cambios = True
    while hubo_cambios:
        hubo_cambios = False
        for item in candidatos_radiculas:
            if not item['valid']:
                interseccion = cv2.bitwise_and(item['mask_dilated'], mask_acumulada_validos)
                if cv2.countNonZero(interseccion) > 0:
                    item['valid'] = True
                    hubo_cambios = True
                    cv2.bitwise_or(mask_acumulada_validos, item['mask'], mask_acumulada_validos)

    # ============================================================
    # 4. EXTRACCIÓN DE DATOS FINALES
    # ============================================================
    
    for item in candidatos_radiculas:
        if item['valid']:
            largo_px = obtener_largo_esqueleto(item['mask'])
            detalles_radiculas.append({
                'id': item['id'],
                'largo_cm': largo_px / PIXELES_POR_CM
            })
            cv2.drawContours(img_debug, [item['cnt']], -1, (0, 255, 0), 2)
        else:
            cv2.drawContours(img_debug, [item['cnt']], -1, (0, 0, 255), 1)

    return {
        'lista_semillas': detalles_semillas,
        'lista_radiculas': detalles_radiculas,
        'img_debug': img_debug
    }

def redimensionar_para_ver(imagen, alto_maximo):
    alto_original, ancho_original = imagen.shape[:2]
    if alto_original > alto_maximo:
        ratio = alto_maximo / float(alto_original)
        nuevo_ancho = int(ancho_original * ratio)
        nuevo_alto = int(alto_original * ratio)
        return cv2.resize(imagen, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)
    return imagen

def main():
    global MOSTRAR_EN_PANTALLA
    
    print(f"Iniciando análisis CON RECORTE AUTOMÁTICO en: {DIRECTORIO_RAIZ}")
    print("Controles: [ESPACIO] Siguiente | [ESC] Salir")
    
    if GUARDAR_IMAGENES_TESTIGO and not os.path.exists(CARPETA_TESTIGOS):
        os.makedirs(CARPETA_TESTIGOS)

    filas_csv = []

    for root, dirs, files in os.walk(DIRECTORIO_RAIZ):
        nombre_carpeta = os.path.basename(root)
        archivos_en_carpeta = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for archivo in archivos_en_carpeta:
            ruta_completa = os.path.join(root, archivo)
            
            nombre_clean = os.path.splitext(archivo)[0].lower()
            tipo = "Desconocido"
            replica = "NA"

            if "control" in nombre_clean:
                tipo = "Control"
                replica = ''.join(filter(str.isdigit, nombre_clean))
            elif "cm" in nombre_clean:
                tipo = "Magnetico"
                replica = ''.join(filter(str.isdigit, nombre_clean))
            else:
                continue 

            print(f"  ... Procesando: {archivo}")

            res = procesar_imagen(ruta_completa)
            
            if res:
                if MOSTRAR_EN_PANTALLA and res['img_debug'] is not None:
                    img_ver = redimensionar_para_ver(res['img_debug'], ALTO_MAXIMO_PANTALLA)
                    cv2.imshow("Verde=Conectada | Azul=Semilla", img_ver)
                    key = cv2.waitKey(0) & 0xFF
                    if key == 27: 
                        MOSTRAR_EN_PANTALLA = False
                        cv2.destroyAllWindows()

                if GUARDAR_IMAGENES_TESTIGO and res['img_debug'] is not None:
                    nombre_testigo = f"{nombre_carpeta}_{archivo}"
                    cv2.imwrite(os.path.join(CARPETA_TESTIGOS, nombre_testigo), res['img_debug'])
                
                for rad in res['lista_radiculas']:
                    filas_csv.append({
                        'Fecha': nombre_carpeta,
                        'Grupo': tipo,
                        'Replica': replica,
                        'Archivo': archivo,
                        'Tipo_Objeto': 'Radicula',
                        'ID_Objeto': rad['id'],
                        'Medida_Valor': round(rad['largo_cm'], 4),
                        'Medida_Tipo': 'Largo_cm'
                    })
                
                for sem in res['lista_semillas']:
                    filas_csv.append({
                        'Fecha': nombre_carpeta,
                        'Grupo': tipo,
                        'Replica': replica,
                        'Archivo': archivo,
                        'Tipo_Objeto': 'Semilla',
                        'ID_Objeto': sem['id'],
                        'Medida_Valor': round(sem['area_px'], 2),
                        'Medida_Tipo': 'Area_px'
                    })

    if MOSTRAR_EN_PANTALLA:
        cv2.destroyAllWindows()

    if filas_csv:
        df = pd.DataFrame(filas_csv)
        try:
            df = df.sort_values(by=['Fecha', 'Grupo', 'Replica', 'Tipo_Objeto'])
        except: pass
        
        print(f"\nGuardando CSV detallado en: {ARCHIVO_SALIDA}...")
        df.to_csv(ARCHIVO_SALIDA, index=False, sep=',', encoding='utf-8')
        print(df.head(10)) 
    else:
        print("No se encontraron datos.")

if __name__ == "__main__":
    main()