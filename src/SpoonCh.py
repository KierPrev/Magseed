import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# CONFIGURACIÓN: Modifica aquí la ruta del CSV
# ==========================================
RUTA_ARCHIVO = "./data/extra/CaracterizacionCuchara.csv"


def generar_graficos(ruta_csv):
    if not os.path.exists(ruta_csv):
        print(f"Error: No se encontró el archivo en la ruta: {ruta_csv}")
        return

    # Se lee el archivo sin encabezado y especificando la coma como separador decimal
    df = pd.read_csv(ruta_csv, header=None, decimal=",")
    df.columns = ["medicion"]

    valores = df["medicion"]
    mean = valores.mean()
    std_dev = valores.std()
    median = valores.median()
    n = len(valores)

    # Estructura de la figura: 1 fila, 2 columnas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Definición de los bins centrados en los valores enteros para datos discretos
    val_min = int(valores.min())
    val_max = int(valores.max())
    bins = np.arange(val_min - 0.5, val_max + 1.5, 1)

    # 1. Histograma con estimación de densidad (KDE)
    sns.histplot(
        valores, bins=bins, kde=True, ax=ax1, color="skyblue", edgecolor="black"
    )

    # Líneas para media y mediana
    ax1.axvline(
        mean,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"Media: {mean:.2f}",
    )
    ax1.axvline(
        median,
        color="darkorange",
        linestyle="-",
        linewidth=2,
        label=f"Mediana: {median:.2f}",
    )

    # Cuadro de texto con estadísticas adicionales (SD y N)
    texto_stats = f"SD: {std_dev:.2f}\nN: {n}"
    ax1.text(
        0.05,
        0.95,
        texto_stats,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax1.set_title("Histograma de Mediciones")
    ax1.set_xlabel("Valor medido")
    ax1.set_ylabel("Frecuencia")
    ax1.legend(loc="upper right")

    # 2. Diagrama de caja (Boxplot)
    sns.boxplot(x=valores, ax=ax2, color="lightgreen")
    ax2.set_title("Diagrama de Caja (Boxplot)")
    ax2.set_xlabel("Valor medido")

    plt.tight_layout()

    # Opcional: plt.savefig('analisis_mediciones.png', dpi=300)
    plt.show()

    # Imprimir los estadísticos en la consola
    print("Estadísticos descriptivos:")
    print(f"Media: {mean:.2f}")
    print(f"Desviación Estándar (SD): {std_dev:.2f}")
    print(f"Mediana: {median:.2f}")
    print(f"Tamaño de la muestra (N): {n}")


if __name__ == "__main__":
    generar_graficos(RUTA_ARCHIVO)
