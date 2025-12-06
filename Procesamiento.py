import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def graficar_comparacion(ruta_csv):
    # 1. Cargar la tabla de datos
    try:
        df = pd.read_csv(ruta_csv)
    except FileNotFoundError:
        print("No encuentro el archivo 'resultados_experimentos.csv'.")
        print("Asegurate de haber corrido el script anterior primero.")
        return

    print("--- DATOS CARGADOS ---")
    print(df.groupby('Tratamiento')['Largo_px'].describe())

    # Separamos los datos en dos grupos para graficar
    # Asumimos que en la columna 'Tratamiento' dice 'Control' y 'Magnetico'
    datos_control = df[df['Tratamiento'] == 'Control']['Largo_px']
    datos_mag = df[df['Tratamiento'] == 'Magnetico']['Largo_px']

    # --- CONFIGURACIÓN DE GRÁFICOS ---
    plt.style.use('bmh') # Un estilo visual limpio y académico
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Análisis Comparativo: Maíz Tratado vs Control', fontsize=16)

    # GRÁFICO 1: Boxplot (Cajas y Bigotes)
    # Ideal para ver medianas y dispersión
    bp = ax1.boxplot([datos_control, datos_mag], 
                     labels=['Control', 'Magnético'], 
                     patch_artist=True,
                     notch=True) # notch=True hace una muesca en la mediana (intervalo de confianza visual)

    # Colorear las cajas
    colores = ['lightgreen', 'salmon']
    for patch, color in zip(bp['boxes'], colores):
        patch.set_facecolor(color)

    # Agregar los puntos reales (jitter) para ver la "nube" de semillas
    # Esto le da mucha credibilidad física a tu gráfico
    x_control = np.random.normal(1, 0.04, size=len(datos_control))
    x_mag = np.random.normal(2, 0.04, size=len(datos_mag))
    
    ax1.plot(x_control, datos_control, 'r.', alpha=0.5)
    ax1.plot(x_mag, datos_mag, 'r.', alpha=0.5)

    ax1.set_title('Comparación de Longitudes (Boxplot)')
    ax1.set_ylabel('Largo de Radícula (píxeles)')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # GRÁFICO 2: Histogramas Superpuestos
    # Ideal para ver si la población se "desplazó"
    ax2.hist(datos_control, bins=15, alpha=0.6, color='green', label='Control', density=True)
    ax2.hist(datos_mag, bins=15, alpha=0.6, color='red', label='Magnético', density=True)
    
    # Líneas de promedio
    ax2.axvline(datos_control.mean(), color='green', linestyle='dashed', linewidth=2)
    ax2.axvline(datos_mag.mean(), color='red', linestyle='dashed', linewidth=2)

    ax2.set_title('Distribución de Poblaciones (Histograma)')
    ax2.set_xlabel('Largo (píxeles)')
    ax2.set_ylabel('Densidad de Probabilidad')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# EJECUTAR
graficar_comparacion('resultados_v5_visual.csv')