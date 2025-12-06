import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import math

# =============================================================================
# 🎛️ TABLERO DE CONTROL (V15 - CONTROL MANUAL DE TONOS)
# =============================================================================

PARAMETROS = {
    # --- 0. AJUSTE DE TONOS (Lightroom Style) ---
    # Sombras (Gamma): > 1.0 aclara sombras (ej: 1.5). < 1.0 oscurece sombras.
    # Recomendado: 1.3 a 1.8 para ver raíces en la oscuridad.
    "AJUSTE_SOMBRAS": 1.7, 

    # Highlights (Ganancia lineal en zonas altas): 
    # < 1.0 oscurece los blancos (recupera detalle). > 1.0 quema los blancos.
    # Recomendado: 0.8 a 0.9 para que el cartón brillante no moleste.
    "AJUSTE_HIGHLIGHTS": 0.80,

    # Contraste Global: 1.0 = normal. 1.2 = +20% contraste.
    "AJUSTE_CONTRASTE": 1.5,


    # --- 1. Detección de SEMILLAS ---
    "SEMILLA_MIN_AREA": 500,
    "SEMILLA_MAX_AREA": 60000, 
    "HSV_SAT_MIN": 95, 
    "SEMILLA_MORPH_KERNEL": (7, 7), 

    # --- 2. Detección de RAÍCES ---
    "RAIZ_DIFF_UMBRAL": 15, 
    "RAIZ_MIN_AREA": 80,

    # --- 3. Contexto ---
    "ZONA_SEGURA_DILATACION": 15,
    "ZONA_SEGURA_ITERACIONES": 6,

    # --- 4. General ---
    "MARGEN_ROI": 0.10
}

# =============================================================================
# 📸 FUNCIÓN DE AJUSTE DE TONOS (MANUAL)
# =============================================================================
def ajustar_tonos(img, gamma_sombras, gain_highlights, contraste):
    """
    Simula ajustes de Shadows/Highlights usando corrección Gamma y Ganancia.
    """
    # 1. Convertir a LAB para tocar solo Luminosidad
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Normalizar L a 0-1.0 para operar matemáticamente
    l_norm = l.astype(np.float32) / 255.0

    # 2. Ajuste de SOMBRAS (Corrección Gamma)
    # Gamma < 1 aclara sombras. Gamma > 1 oscurece.
    # (Invertimos la lógica del input para que sea más intuitivo: >1 aclara)
    if gamma_sombras != 1.0:
        inv_gamma = 1.0 / gamma_sombras
        l_norm = np.power(l_norm, inv_gamma)

    # 3. Ajuste de HIGHLIGHTS (Ganancia en zona alta)
    # Aplicamos una curva suave que comprime los blancos
    if gain_highlights != 1.0:
        l_norm = l_norm * gain_highlights
    
    # 4. Contraste Global
    if contraste != 1.0:
        l_norm = (l_norm - 0.5) * contraste + 0.5

    # Clip para asegurar rango válido y volver a 8-bit
    l_norm = np.clip(l_norm, 0, 1)
    l_processed = (l_norm * 255).astype(np.uint8)

    # Reconstruir imagen
    lab_merged = cv2.merge((l_processed, a, b))
    img_final = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
    
    return img_final


# =============================================================================
# ⚙️ MOTOR DE VISIÓN V15
# =============================================================================

def analizar_parametrico(ruta, params):
    
    img_full = cv2.imread(ruta)
    if img_full is None: return [], None, 0
    
    # --- APLICAR AJUSTE DE TONOS ---
    img = ajustar_tonos(img_full, 
                        params["AJUSTE_SOMBRAS"], 
                        params["AJUSTE_HIGHLIGHTS"],
                        params["AJUSTE_CONTRASTE"])
    
    alto, ancho = img.shape[:2]

    # --- PASO A: SEMILLAS (HSV) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_1 = np.array([0, params["HSV_SAT_MIN"], 50])
    upper_1 = np.array([25, 255, 255])
    mask_1 = cv2.inRange(hsv, lower_1, upper_1)
    
    lower_2 = np.array([125, params["HSV_SAT_MIN"], 50])
    upper_2 = np.array([180, 255, 255])
    mask_2 = cv2.inRange(hsv, lower_2, upper_2)
    
    mascara_semillas_raw = cv2.bitwise_or(mask_1, mask_2)
    
    kernel_semilla = np.ones(params["SEMILLA_MORPH_KERNEL"], np.uint8)
    mascara_semillas_raw = cv2.morphologyEx(mascara_semillas_raw, cv2.MORPH_CLOSE, kernel_semilla)
    mascara_semillas_raw = cv2.morphologyEx(mascara_semillas_raw, cv2.MORPH_OPEN, kernel_semilla)

    mascara_semillas_filtrada = np.zeros_like(mascara_semillas_raw)
    contornos_semillas, _ = cv2.findContours(mascara_semillas_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # VISUALIZACIÓN: Usamos la imagen 'img' (la ajustada manualmente)
    img_visual = img.copy() 
    
    semillas_validas = 0
    for c in contornos_semillas:
        area = cv2.contourArea(c)
        if params["SEMILLA_MIN_AREA"] < area < params["SEMILLA_MAX_AREA"]:
            cv2.drawContours(mascara_semillas_filtrada, [c], -1, 255, -1)
            # Dibujamos MUY GRUESO (10px)
            cv2.drawContours(img_visual, [c], -1, (255, 0, 255), 10) 
            semillas_validas += 1

    # --- PASO B: RAÍCES ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    
    fondo_local = cv2.medianBlur(l_channel, 65)
    diff = cv2.subtract(l_channel, fondo_local)
    
    _, mascara_raices_raw = cv2.threshold(diff, params["RAIZ_DIFF_UMBRAL"], 255, cv2.THRESH_BINARY)
    mascara_raices = cv2.morphologyEx(mascara_raices_raw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    semillas_dilatadas = cv2.dilate(mascara_semillas_filtrada, np.ones((11,11), np.uint8))
    mascara_solo_raices = cv2.subtract(mascara_raices, semillas_dilatadas)

    # --- PASO C: CONTEXTO ---
    zona_segura_kernel = np.ones((params["ZONA_SEGURA_DILATACION"], params["ZONA_SEGURA_DILATACION"]), np.uint8)
    zona_segura = cv2.dilate(mascara_semillas_filtrada, zona_segura_kernel, iterations=params["ZONA_SEGURA_ITERACIONES"])
    
    margen_x = int(ancho * params["MARGEN_ROI"])
    margen_y = int(alto * params["MARGEN_ROI"])
    mascara_roi = np.zeros_like(mascara_solo_raices)
    mascara_roi[margen_y:alto-margen_y, margen_x:ancho-margen_x] = 255
    mascara_solo_raices = cv2.bitwise_and(mascara_solo_raices, mascara_roi)

    contornos_raices, _ = cv2.findContours(mascara_solo_raices, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largos = []
    
    for c in contornos_raices:
        if cv2.contourArea(c) < params["RAIZ_MIN_AREA"]: continue
        
        mask_obj = np.zeros_like(mascara_solo_raices)
        cv2.drawContours(mask_obj, [c], -1, 255, -1)
        
        if cv2.countNonZero(cv2.bitwise_and(mask_obj, zona_segura)) == 0:
            continue 
            
        largo = cv2.arcLength(c, True) / 2
        largos.append(largo)
        
        # Raíz Verde GRUESA
        cv2.drawContours(img_visual, [c], -1, (0, 255, 0), 8)
        
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            # Texto Gigante
            cv2.putText(img_visual, f"{int(largo)}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 0), 5)

    img_thumb = cv2.resize(img_visual, (0,0), fx=0.2, fy=0.2)
    img_thumb = cv2.cvtColor(img_thumb, cv2.COLOR_BGR2RGB)
    
    return largos, img_thumb, semillas_validas


# --- CLASE SLIDESHOW ---
class ImageSlideshow:
    def __init__(self, imagenes, titulos):
        self.imagenes = imagenes
        self.titulos = titulos
        self.ind = 0
        if not imagenes: return
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('Visor V15 (Ajuste Manual Tonos)')
        self.im_obj = self.ax.imshow(self.imagenes[self.ind])
        self.title_obj = self.ax.set_title(self.get_title_str(), fontsize=14, color='blue')
        self.ax.axis('off')
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        print("\n--- CONTROLES --- [Espacio/Flecha Der] -> Siguiente | [Backspace/Flecha Izq] -> Anterior")
        plt.tight_layout(); plt.show()
    def get_title_str(self): return f"Imagen {self.ind + 1}/{len(self.imagenes)}\n{self.titulos[self.ind]}"
    def update_view(self):
        self.im_obj.set_data(self.imagenes[self.ind])
        self.title_obj.set_text(self.get_title_str())
        self.fig.canvas.draw_idle()
    def next_image(self): self.ind = (self.ind + 1) % len(self.imagenes); self.update_view()
    def prev_image(self): self.ind = (self.ind - 1) % len(self.imagenes); self.update_view()
    def on_click(self, event):
        if event.button == 1: self.next_image()
        elif event.button == 3: self.prev_image()
    def on_key(self, event):
        if event.key in ['right', ' ', 'enter', 'n']: self.next_image()
        elif event.key in ['left', 'backspace', 'p']: self.prev_image()

# --- CONTROLADOR ---
def procesar_v15(carpeta_raiz):
    if not os.path.exists(carpeta_raiz): print(f"❌ ERROR: carpeta '{carpeta_raiz}' no existe."); return
    base_de_datos = []; galeria_imagenes = []; titulos_imagenes = []
    print(f"Iniciando V15 (Control Manual de Tonos) en: {carpeta_raiz}...\n")
    archivos_encontrados = 0
    for ruta_actual, carpetas, archivos in os.walk(carpeta_raiz):
        nombre_carpeta = os.path.basename(ruta_actual)
        for archivo in archivos:
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                archivos_encontrados += 1
                ruta_completa = os.path.join(ruta_actual, archivo).replace("\\", "/") 
                tipo = "Control" if "Control" in archivo else "Magnetico" if "CM" in archivo else "Otro"
                print(f"--> Analizando: {archivo} ({tipo})...", end=" ")
                mediciones, foto_mini, n_semillas = analizar_parametrico(ruta_completa, PARAMETROS)
                print(f"| Semillas: {n_semillas} | Raíces: {len(mediciones)}")
                if foto_mini is not None:
                    galeria_imagenes.append(foto_mini)
                    titulos_imagenes.append(f"Exp: {nombre_carpeta} | {archivo}\nSemillas: {n_semillas} | Raíces: {len(mediciones)}")
                for i, largo in enumerate(mediciones):
                    base_de_datos.append({"Fecha": nombre_carpeta, "Archivo": archivo, "Tratamiento": tipo, "Semilla_ID": i + 1, "Largo_px": largo})
    if archivos_encontrados == 0: print("\n⚠️ No se encontraron imágenes."); return
    if len(base_de_datos) > 0:
        df = pd.DataFrame(base_de_datos)
        print("\n--- ¡PROCESO TERMINADO! ---")
        df.to_csv("resultados_v15_tonos.csv", index=False)
        print(df.groupby(["Fecha", "Tratamiento"])["Largo_px"].agg(['mean', 'std', 'count']))
        if len(galeria_imagenes) > 0: ImageSlideshow(galeria_imagenes, titulos_imagenes)
    else:
        print("\n❌ NO SE DETECTARON DATOS.")

# EJECUTAR
procesar_v15('./Experimentos')