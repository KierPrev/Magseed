import cv2
import numpy as np
import pandas as pd
import os
import glob
import re

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
UMBRAL_BLANCO = 190 
MARGEN_SEGURIDAD = 30 

# ==========================================
# FIN DEL PANEL DE CONTROL
# ==========================================

def recortar_bordes_blancos(img):
    if img is None:
        return None

    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, UMBRAL_BLANCO, 255, cv2.THRESH_BINARY_INV)
    puntos = cv2.findNonZero(thresh)

    if puntos is not None:
        x, y, w, h = cv2.boundingRect(puntos)
        x = max(0, x - MARGEN_SEGURIDAD)
        y = max(0, y - MARGEN_SEGURIDAD)
        w = min(w_img - x, w + (MARGEN_SEGURIDAD * 2))
        h = min(h_img - y, h + (MARGEN_SEGURIDAD * 2))
        return img[y:y+h, x:x+w]
    
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

    img = recortar_bordes_blancos(img)

    h, w = img.shape[:2]
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # 1. DETECCIÓN
    lab = cv2.cvtColor(img_blur, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # Semillas
    a_clahe = clahe.apply(a_channel)
    _, mask_sem = cv2.threshold(a_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_sem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_sem = cv2.morphologyEx(mask_sem, cv2.MORPH_CLOSE, kernel_sem, iterations=3)
    mask_sem = cv2.morphologyEx(mask_sem, cv2.MORPH_OPEN, kernel_sem, iterations=1)

    # Radículas
    l_clahe = clahe.apply(l_channel)
    mask_rad = cv2.adaptiveThreshold(l_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 21, -5)
    kernel_rad = np.ones((3,3), np.uint8)
    mask_rad = cv2.morphologyEx(mask_rad, cv2.MORPH_OPEN, kernel_rad, iterations=1)
    mask_rad = cv2.subtract(mask_rad, mask_sem)

    cnt_rad, _ = cv2.findContours(mask_rad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_sem, _ = cv2.findContours(mask_sem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_debug = img.copy()

    # 2. PREPARACIÓN DE OBJETOS
    detalles_semillas = []
    detalles_radiculas = []
    mask_acumulada_validos = np.zeros((h, w), dtype=np.uint8)
    
    for i, cnt in enumerate(cnt_sem):
        area = cv2.contourArea(cnt)
        if AREA_MIN_SEMILLA < area < AREA_MAX_SEMILLA:
            detalles_semillas.append({'id': i, 'area_px': area})
            cv2.drawContours(mask_acumulada_validos, [cnt], -1, 255, -1)
            cv2.drawContours(img_debug, [cnt], -1, (255, 0, 0), 2)

    candidatos_radiculas = []
    kernel_dilatacion = np.ones((DISTANCIA_MAX_UNION, DISTANCIA_MAX_UNION), np.uint8)

    for i, cnt in enumerate(cnt_rad):
        area = cv2.contourArea(cnt)
        if AREA_MIN_RADICULA < area < AREA_MAX_RADICULA:
            mask_single = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask_single, [cnt], -1, 255, -1)
            mask_dilated = cv2.dilate(mask_single, kernel_dilatacion, iterations=1)
            
            candidatos_radiculas.append({
                'id': i, 'cnt': cnt, 'mask': mask_single,
                'mask_dilated': mask_dilated, 'valid': False 
            })

    # 3. LÓGICA DE PROPAGACIÓN
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

    # 4. EXTRACCIÓN
    for item in candidatos_radiculas:
        if item['valid']:
            largo_px = obtener_largo_esqueleto(item['mask'])
            detalles_radiculas.append({'id': item['id'], 'largo_cm': largo_px / PIXELES_POR_CM})
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

def dibujar_info_overlay(img, cant_semillas, cant_radiculas, nombre_archivo):
    """
    Agrega texto GRANDE y adaptable al tamaño de la imagen.
    """
    img_out = img.copy()
    h, w = img.shape[:2]
    
    # --- ESCALADO DINÁMICO ---
    # Si la imagen tiene 4000px de alto, el scale será aprox 4.0 (muy grande)
    # Si la imagen tiene 800px de alto, el scale será aprox 1.5 (normal)
    scale = max(1.5, h / 800.0) 
    
    # Grosor proporcional al tamaño
    thickness = int(scale * 2.5)
    
    lineas = [
        f"Archivo: {nombre_archivo}",
        f"Semillas: {cant_semillas}",
        f"Radiculas: {cant_radiculas}"
    ]
    
    # Posición inicial y espaciado
    x = int(30 * scale)
    y = int(50 * scale)
    line_height = int(50 * scale) # Espacio entre líneas
    
    for linea in lineas:
        # 1. Dibujar borde negro (Sombra para contraste)
        cv2.putText(img_out, linea, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 4)
        # 2. Dibujar texto amarillo brillante
        cv2.putText(img_out, linea, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), thickness)
        
        y += line_height

    return img_out

def seleccionar_carpetas(directorio_raiz):
    if not os.path.exists(directorio_raiz):
        print(f"Error: No existe el directorio {directorio_raiz}")
        return []

    todas_las_carpetas = [d for d in os.listdir(directorio_raiz) 
                          if os.path.isdir(os.path.join(directorio_raiz, d))]
    
    patron_fecha = re.compile(r'^\s*\d{2}-\d{2}-\d{2}\s*$')

    carpetas_fechas = [d for d in todas_las_carpetas if patron_fecha.match(d)]
    carpetas_fechas.sort()

    if not carpetas_fechas:
        print("No se encontraron carpetas con formato YY-MM-DD (ej: 25-11-12).")
        print(f"Carpetas encontradas: {todas_las_carpetas}")
        return []

    print("\n=== FECHAS DISPONIBLES ===")
    for idx, nombre in enumerate(carpetas_fechas):
        print(f"[{idx+1}] {nombre}")
    print("==========================")
    print("Escribe los números separados por coma (ej: 1,3) o 'todas'.")
    
    seleccion = input("\nTu elección: ").strip().lower()

    carpetas_elegidas = []

    if seleccion == 'todas' or seleccion == '':
        carpetas_elegidas = carpetas_fechas
        print(">> Seleccionado: TODAS.")
    else:
        try:
            indices = [int(x.strip()) - 1 for x in seleccion.split(',')]
            for i in indices:
                if 0 <= i < len(carpetas_fechas):
                    carpetas_elegidas.append(carpetas_fechas[i])
            print(f">> Seleccionado: {len(carpetas_elegidas)} carpeta(s).")
        except ValueError:
            print("Entrada no válida.")
            return []
            
    return carpetas_elegidas

def main():
    global MOSTRAR_EN_PANTALLA
    
    carpetas_a_procesar = seleccionar_carpetas(DIRECTORIO_RAIZ)
    
    if not carpetas_a_procesar:
        return

    print(f"\nIniciando análisis en {len(carpetas_a_procesar)} carpeta(s)...")
    print("Controles: [ESPACIO] Siguiente | [ESC] Salir")
    
    if GUARDAR_IMAGENES_TESTIGO and not os.path.exists(CARPETA_TESTIGOS):
        os.makedirs(CARPETA_TESTIGOS)

    filas_csv = []

    for nombre_carpeta in carpetas_a_procesar:
        ruta_carpeta = os.path.join(DIRECTORIO_RAIZ, nombre_carpeta)
        
        patrones = ['*.jpg', '*.jpeg', '*.png']
        archivos_en_carpeta = []
        for patron in patrones:
            archivos_en_carpeta.extend(glob.glob(os.path.join(ruta_carpeta, patron)))
            archivos_en_carpeta.extend(glob.glob(os.path.join(ruta_carpeta, patron.upper())))
        
        archivos_en_carpeta = sorted(list(set(archivos_en_carpeta)))

        print(f"\n---> Entrando en: {nombre_carpeta} ({len(archivos_en_carpeta)} imágenes)")

        for ruta_completa in archivos_en_carpeta:
            archivo = os.path.basename(ruta_completa)
            
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
                # --- NUEVO: Agregar Overlay ---
                cant_semillas = len(res['lista_semillas'])
                cant_radiculas = len(res['lista_radiculas'])
                
                # Dibujamos sobre la imagen de debug
                if res['img_debug'] is not None:
                    res['img_debug'] = dibujar_info_overlay(
                        res['img_debug'], 
                        cant_semillas, 
                        cant_radiculas, 
                        archivo
                    )

                # --- Visualización ---
                if MOSTRAR_EN_PANTALLA and res['img_debug'] is not None:
                    img_ver = redimensionar_para_ver(res['img_debug'], ALTO_MAXIMO_PANTALLA)
                    cv2.imshow(f"Analizando: {archivo}", img_ver)
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