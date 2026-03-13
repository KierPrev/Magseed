import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.widgets import Slider

# ==========================================
# ⚙️ CONFIGURACIÓN
# ==========================================
ARCHIVO_DATOS = 'resultados_limpios.csv' 
COLOR_CONTROL = '#3498db' 
COLOR_TRATADO = '#e74c3c' 

# --- AJUSTA ESTOS DOS VALORES ---
OBJETO_A_GRAFICAR = 'Radicula'   
TIPO_MEDIDA = 'Largo_cm'         # 'Area_cm2' o 'Largo_cm'

# ==========================================
# 1. CARGA Y LIMPIEZA
# ==========================================
try:
    df = pd.read_csv(ARCHIVO_DATOS)
except FileNotFoundError:
    print(f"[ERROR] No encuentro el archivo: {ARCHIVO_DATOS}")
    df = pd.DataFrame()

# ==========================================
# 2. COMPROBACIÓN Y PREPARACIÓN
# ==========================================

columnas_necesarias = ['Grupo', 'Tipo_Objeto', 'Medida_Tipo', 'Medida_Valor', 'Replica']
if not df.empty and all(col in df.columns for col in columnas_necesarias):
    
    # 2.1. Limpieza de texto
    cols_texto = ['Grupo', 'Tipo_Objeto', 'Medida_Tipo']
    for col in cols_texto:
        df[col] = df[col].astype(str).str.strip()
    
    # 2.2. Conversión de Medida_Valor
    df['Medida_Valor'] = pd.to_numeric(
        df['Medida_Valor'].astype(str).str.replace(',', '.'), 
        errors='coerce'
    )

    df['ID_Muestra'] = df['Grupo'].astype(str).str.strip() + " " + df['Replica'].astype(str)

    # 2.3. Filtrado
    df_subset = df[
        (df['Tipo_Objeto'] == OBJETO_A_GRAFICAR) & 
        (df['Medida_Tipo'] == TIPO_MEDIDA)
    ].copy()

    df_subset = df_subset.dropna(subset=['Medida_Valor'])

    if df_subset.empty:
        print(f"[ALERTA] No hay datos para graficar '{OBJETO_A_GRAFICAR}' con medida '{TIPO_MEDIDA}'.")
    else:
        
        # 2.4. Configuración estética general
        df_subset.sort_values(by=['Grupo', 'Replica'], inplace=True)
        
        sns.set_theme(style="whitegrid", context="notebook")
        max_valor_datos = df_subset['Medida_Valor'].max()
        limite_inicial = max_valor_datos * 1.30
        offset_text = max_valor_datos * 0.05 
        fmt = "{:.5f}" if "Area" in TIPO_MEDIDA else "{:.2f}"
        unidad = "(cm2)" if "Area" in TIPO_MEDIDA else "(cm)"
        paleta_colores = {'Control': COLOR_CONTROL, 'Magnetico': COLOR_TRATADO}


        # ==========================================
        # 🖥️ VENTANA 1: DETALLADO (POR RÉPLICA)
        # ==========================================
        fig1, ax1 = plt.subplots(figsize=(10, 7))
        fig1.canvas.manager.set_window_title('Ventana 1: Detalle por Réplica')
        plt.subplots_adjust(bottom=0.25)

        sns.boxplot(data=df_subset, x='ID_Muestra', y='Medida_Valor', hue='Grupo',             
                    palette=paleta_colores, showfliers=False, dodge=False, width=0.6, ax=ax1)

        sns.stripplot(data=df_subset, x='ID_Muestra', y='Medida_Valor', 
                      color='black', alpha=0.3, size=4, ax=ax1)

        ax1.set_title(f'Detalle por Réplica: {OBJETO_A_GRAFICAR}', fontsize=14)
        ax1.set_ylabel(f'{TIPO_MEDIDA} {unidad}')
        ax1.set_xlabel('Réplica Individual')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0, limite_inicial)

        # Etiquetas Ventana 1
        stats1 = df_subset.groupby('ID_Muestra')['Medida_Valor'].agg(['mean', 'max']).reset_index()
        x_positions = {label.get_text(): i for i, label in enumerate(ax1.get_xticklabels())}
        
        for i, row in stats1.iterrows():
            id_muestra = row['ID_Muestra']
            if id_muestra in x_positions:
                y_pos_text = row['max'] * 1.05 + offset_text 
                
                ax1.text(x_positions[id_muestra], y_pos_text, 
                        f"Media:\n{fmt.format(row['mean'])}", ha='center', va='bottom', 
                        fontsize=8, fontweight='bold', color='#333333')
        
        # --- SLIDER VENTANA 1 ---
        ax_slider1 = fig1.add_axes([0.15, 0.1, 0.65, 0.03], facecolor='lightgray')
        slider1 = Slider(ax_slider1, 'Zoom Y ', max_valor_datos * 0.1, max_valor_datos * 3.0, valinit=limite_inicial)
        
        def update1_simple(val):
             ax1.set_ylim(0, val)
             fig1.canvas.draw_idle()
             
        slider1.on_changed(update1_simple)


        # ==========================================
        # 🖥️ VENTANA 2: COMPARACIÓN GLOBAL
        # ==========================================
        fig2, ax2 = plt.subplots(figsize=(6, 7))
        fig2.canvas.manager.set_window_title('Ventana 2: Comparación Global')
        plt.subplots_adjust(bottom=0.25)

        # Boxplot Global
        sns.boxplot(data=df_subset, x='Grupo', y='Medida_Valor', 
                    palette=paleta_colores, showfliers=False, width=0.5, ax=ax2)

        sns.stripplot(data=df_subset, x='Grupo', y='Medida_Valor', 
                      color='black', alpha=0.3, size=4, ax=ax2)

        ax2.set_title(f'COMPARACIÓN FINAL\n{OBJETO_A_GRAFICAR} (Pool de datos)', fontsize=14, fontweight='bold')
        ax2.set_ylabel(f'{TIPO_MEDIDA} {unidad}')
        ax2.set_xlabel('Condición Experimental')
        ax2.set_ylim(0, limite_inicial)

        # Etiquetas Ventana 2 (Media)
        stats2 = df_subset.groupby('Grupo')['Medida_Valor'].agg(mean='mean', ymax='max').reset_index()
        grupos_ordenados = [label.get_text() for label in ax2.get_xticklabels()]
        
        for i, grupo in enumerate(grupos_ordenados):
            row = stats2[stats2['Grupo'] == grupo]
            if not row.empty:
                val_mean = row.iloc[0]['mean']
                val_max = row.iloc[0]['ymax']
                y_pos_text = val_max * 1.05 + offset_text 
                
                ax2.text(i, y_pos_text, f"Media: {fmt.format(val_mean)}", 
                         ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

        # --- SLIDER VENTANA 2 ---
        ax_slider2 = fig2.add_axes([0.15, 0.1, 0.65, 0.03], facecolor='lightgray')
        slider2 = Slider(ax_slider2, 'Zoom Y ', max_valor_datos * 0.1, max_valor_datos * 3.0, valinit=limite_inicial)
        
        def update2_simple(val):
            ax2.set_ylim(0, val)
            fig2.canvas.draw_idle()
            
        slider2.on_changed(update2_simple)

        print("\n[OK] Generadas 2 ventanas interactivas.")
        plt.show()

else:
    if not df.empty:
        print("[ERROR] El CSV no contiene todas las columnas necesarias (Grupo, Tipo_Objeto, Medida_Tipo, Medida_Valor, Replica).")
    else:
        print("No se cargó el archivo o el DataFrame está vacío.")