import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from matplotlib.widgets import Button # <--- Importamos el botón

# ==========================================
# 📊 PANEL DE CONTROL
# ==========================================
ARCHIVO_DATOS = 'resultados_detallados_raw.csv' 
COLOR_CONTROL = '#3498db' 
COLOR_TRATADO = '#e74c3c' 

OBJETO = 'Radicula' 
NIVEL_CONFIANZA = 0.05 

# ==========================================
# 1. CARGA DE DATOS
# ==========================================
try:
    df = pd.read_csv(ARCHIVO_DATOS)
except FileNotFoundError:
    print("⚠️ No se encontró el archivo. Usando datos de prueba.")
    np.random.seed(42)
    data = {
        'Grupo': ['Control']*50 + ['Magnetico']*50,
        'Tipo_Objeto': ['Radicula']*100,
        'Medida_Valor': np.concatenate([np.random.normal(100, 15, 50), np.random.normal(115, 18, 50)]),
        'Medida_Tipo': ['Largo_cm']*100
    }
    df = pd.DataFrame(data)

df_subset = df[df['Tipo_Objeto'] == OBJETO].copy()
datos_control = df_subset[df_subset['Grupo'] == 'Control']['Medida_Valor']
datos_mag = df_subset[df_subset['Grupo'] == 'Magnetico']['Medida_Valor']

# ==========================================
# 2. CÁLCULOS ESTADÍSTICOS (Backend)
# ==========================================
# Normalidad
_, p_norm_c = stats.shapiro(datos_control)
_, p_norm_m = stats.shapiro(datos_mag)
es_normal = (p_norm_c > 0.05) and (p_norm_m > 0.05)

# Test de Hipótesis
if es_normal:
    test_nombre = "T-Test (Student)"
    stat, p_value = stats.ttest_ind(datos_control, datos_mag, equal_var=False) 
else:
    test_nombre = "Mann-Whitney U"
    stat, p_value = stats.mannwhitneyu(datos_control, datos_mag)

# Tamaño del Efecto
mean_c, mean_m = datos_control.mean(), datos_mag.mean()
std_c, std_m = datos_control.std(), datos_mag.std()
n_c, n_m = len(datos_control), len(datos_mag)

if (n_c + n_m - 2) > 0:
    pooled_std = np.sqrt(((n_c - 1) * std_c**2 + (n_m - 1) * std_m**2) / (n_c + n_m - 2))
    cohen_d = (mean_m - mean_c) / pooled_std
else:
    cohen_d = 0

# Interpretación
significativo = p_value < NIVEL_CONFIANZA
resultado_texto = "SIGNIFICATIVA" if significativo else "NO SIGNIFICATIVA"
estrellas = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

# ==========================================
# 3. VISUALIZACIÓN GRÁFICA
# ==========================================
sns.set_theme(style="whitegrid", context="notebook")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# --- BOXPLOT ---
sns.boxplot(data=df_subset, x='Grupo', y='Medida_Valor', 
            palette={'Control': COLOR_CONTROL, 'Magnetico': COLOR_TRATADO},
            showfliers=False, ax=axes[0], width=0.5)
sns.stripplot(data=df_subset, x='Grupo', y='Medida_Valor', 
              color='black', alpha=0.3, size=3, ax=axes[0])

# Línea de significancia visual
y_max = df_subset['Medida_Valor'].max()
h = y_max * 0.05
axes[0].plot([0, 0, 1, 1], [y_max+h, y_max+2*h, y_max+2*h, y_max+h], lw=1.5, c='k')
axes[0].text(0.5, y_max + 2.5*h, f"{estrellas}\n(p={p_value:.3f})", 
             ha='center', va='bottom', color='k', fontsize=12, fontweight='bold')

axes[0].set_title(f'Comparación: {OBJETO}', fontsize=14)
axes[0].set_ylabel(f'Medida ({df_subset["Medida_Tipo"].iloc[0]})')
axes[0].set_ylim(top=y_max + 4*h)

# --- HISTOGRAMA ---
sns.histplot(data=df_subset, x='Medida_Valor', hue='Grupo', 
             palette={'Control': COLOR_CONTROL, 'Magnetico': COLOR_TRATADO},
             kde=True, element="step", stat="density", common_norm=False, ax=axes[1])

axes[1].axvline(mean_c, color=COLOR_CONTROL, linestyle='--', label=f'Media C: {mean_c:.1f}')
axes[1].axvline(mean_m, color=COLOR_TRATADO, linestyle='--', label=f'Media M: {mean_m:.1f}')
axes[1].legend()
axes[1].set_title(f'Distribución de {OBJETO}s', fontsize=14)

# Ajustamos espacio abajo para poner el botón
plt.subplots_adjust(bottom=0.2) 

# ==========================================
# 4. BOTÓN Y POPUP (NUEVO CÓDIGO)
# ==========================================

# Función que se ejecuta al hacer clic
def mostrar_popup(event):
    # Crear una nueva figura pequeña (simulando un popup)
    fig_pop = plt.figure(figsize=(5, 4), facecolor='#f0f0f0')
    fig_pop.canvas.manager.set_window_title('Reporte Estadístico Detallado')
    
    # Texto del reporte
    stats_text = (
        f"📋 REPORTE ESTADÍSTICO\n"
        f"========================\n\n"
        f"• Objeto: {OBJETO}\n"
        f"• Test: {test_nombre}\n"
        f"• Resultado: {resultado_texto}\n"
        f"  (p-value = {p_value:.5f})\n\n"
        f"• Tamaño del Efecto:\n"
        f"  Cohen's d = {cohen_d:.3f}\n\n"
        f"------------------------\n"
        f"🔹 CONTROL:\n"
        f"   N = {n_c}\n"
        f"   Media = {mean_c:.2f} ± {std_c:.2f}\n\n"
        f"🔸 TRATAMIENTO:\n"
        f"   N = {n_m}\n"
        f"   Media = {mean_m:.2f} ± {std_m:.2f}"
    )
    
    # Escribir el texto en la nueva ventana
    plt.text(0.05, 0.95, stats_text, fontsize=11, family='monospace',
             transform=fig_pop.transFigure, va='top')
    
    # Quitar ejes para que parezca un cuadro de texto limpio
    plt.axis('off')
    plt.show()

# Crear el área del botón en la figura principal [x, y, ancho, alto]
ax_boton = plt.axes([0.4, 0.05, 0.2, 0.075]) 

# Crear el botón
boton_stats = Button(ax_boton, 'Ver Estadística', color='white', hovercolor='#e0e0e0')
boton_stats.on_clicked(mostrar_popup)

plt.show()