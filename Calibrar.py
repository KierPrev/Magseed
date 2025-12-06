import cv2
import numpy as np

def nada(x):
    pass

def calibrar_colores(ruta_imagen):
    # Cargar imagen
    img = cv2.imread(ruta_imagen)
    if img is None:
        print("Error: No se encuentra la imagen")
        return

    # Redimensionar para que quepa en pantalla (igual que el script anterior)
    alto_maximo = 600
    alto, ancho = img.shape[:2]
    if alto > alto_maximo:
        ratio = alto_maximo / float(alto)
        img = cv2.resize(img, (int(ancho*ratio), int(alto*ratio)))

    # Crear ventana de controles
    cv2.namedWindow('Calibrador')
    
    # Crear Trackbars (Barras deslizantes)
    # H: Matiz (Color), S: Saturación (Pureza), V: Valor (Brillo)
    cv2.createTrackbar('H Min', 'Calibrador', 0, 179, nada)
    cv2.createTrackbar('S Min', 'Calibrador', 0, 255, nada)
    cv2.createTrackbar('V Min', 'Calibrador', 0, 255, nada)
    
    cv2.createTrackbar('H Max', 'Calibrador', 179, 179, nada)
    cv2.createTrackbar('S Max', 'Calibrador', 255, 255, nada)
    cv2.createTrackbar('V Max', 'Calibrador', 255, 255, nada)

    print("Mueve las barras hasta que SOLO veas lo que quieres detectar en BLANCO.")
    print("Presiona 'q' para salir.")

    while True:
        # Leer valores actuales de las barras
        h_min = cv2.getTrackbarPos('H Min', 'Calibrador')
        s_min = cv2.getTrackbarPos('S Min', 'Calibrador')
        v_min = cv2.getTrackbarPos('V Min', 'Calibrador')
        
        h_max = cv2.getTrackbarPos('H Max', 'Calibrador')
        s_max = cv2.getTrackbarPos('S Max', 'Calibrador')
        v_max = cv2.getTrackbarPos('V Max', 'Calibrador')

        # Definir rangos
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # Convertir a HSV y crear máscara
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        # Mostrar resultado (Blanco = Detectado, Negro = Ignorado)
        cv2.imshow('Calibrador', mask)
        
        # Mostrar imagen original al lado para comparar (opcional)
        # cv2.imshow('Original', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\n--- VALORES FINALES ---")
            print(f"MIN_HSV = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"MAX_HSV = np.array([{h_max}, {s_max}, {v_max}])")
            break

    cv2.destroyAllWindows()

# --- EJECUCIÓN ---
# Pon aquí la ruta de UNA de tus fotos típicas
calibrar_colores('./Experimentos/25-12-5/CM1.jpg')