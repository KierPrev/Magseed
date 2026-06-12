import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


def graficar_datos(
    archivo_datos=ARCHIVO_DATOS,
    objeto_a_graficar=OBJETO_A_GRAFICAR,
    tipo_medida=TIPO_MEDIDA,
    archivos_multiples=None,
):
    if archivos_multiples is not None:
        dfs = []
        for fecha, ruta in archivos_multiples:
            try:
                df_temp = pd.read_csv(ruta)
                df_temp['Fecha'] = str(fecha)
                dfs.append(df_temp)
            except FileNotFoundError:
                print(f"[ERROR] No encuentro el archivo: {ruta}")
        if not dfs:
            return None
        df = pd.concat(dfs, ignore_index=True)
    else:
        try:
            df = pd.read_csv(archivo_datos)
            df['Fecha'] = 'Única'
        except FileNotFoundError:
            print(f"[ERROR] No encuentro el archivo: {archivo_datos}")
            return None

    columnas_necesarias = ['Grupo', 'Tipo_Objeto', 'Medida_Tipo', 'Medida_Valor', 'Replica']
    if not all(col in df.columns for col in columnas_necesarias):
        print("[ERROR] El CSV no contiene todas las columnas necesarias (Grupo, Tipo_Objeto, Medida_Tipo, Medida_Valor, Replica).")
        return None

    cols_texto = ['Grupo', 'Tipo_Objeto', 'Medida_Tipo', 'Fecha']
    for col in cols_texto:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df['Medida_Valor'] = pd.to_numeric(
        df['Medida_Valor'].astype(str).str.replace(',', '.'),
        errors='coerce'
    )

    if len(df['Fecha'].unique()) > 1:
        df['ID_Muestra'] = df['Fecha'] + " - " + df['Grupo'] + " " + df['Replica']
    else:
        df['ID_Muestra'] = df['Grupo'].astype(str).str.strip() + " " + df['Replica'].astype(str)

    df_subset = df[
        (df['Tipo_Objeto'] == objeto_a_graficar) &
        (df['Medida_Tipo'] == tipo_medida)
    ].copy()

    df_subset = df_subset.dropna(subset=['Medida_Valor'])

    if df_subset.empty:
        print(f"[ALERTA] No hay datos para graficar '{objeto_a_graficar}' con medida '{tipo_medida}'.")
        return None

    df_subset.sort_values(by=['Grupo', 'Replica'], inplace=True)

    sns.set_theme(style="whitegrid", context="notebook")
    max_valor_datos = df_subset['Medida_Valor'].max()
    limite_inicial = max_valor_datos * 1.30 if max_valor_datos > 0 else 1.0
    offset_text = max_valor_datos * 0.05 if max_valor_datos > 0 else 0.05
    fmt = "{:.5f}" if "Area" in tipo_medida else "{:.2f}"
    unidad = "(cm2)" if "Area" in tipo_medida else "(cm)"
    paleta_colores = {'Control': COLOR_CONTROL, 'Magnetico': COLOR_TRATADO}
    grupos_presentes = set(df_subset['Grupo'].dropna().astype(str).unique())
    paleta = paleta_colores if grupos_presentes.issubset(paleta_colores) else None

    fig1, ax1 = plt.subplots(figsize=(10, 7))
    fig1.canvas.manager.set_window_title('Ventana 1: Detalle por Réplica')
    plt.subplots_adjust(bottom=0.25)

    sns.boxplot(
        data=df_subset,
        x='ID_Muestra',
        y='Medida_Valor',
        hue='Grupo',
        palette=paleta,
        showfliers=False,
        dodge=False,
        width=0.6,
        ax=ax1,
    )

    sns.stripplot(
        data=df_subset,
        x='ID_Muestra',
        y='Medida_Valor',
        color='black',
        alpha=0.3,
        size=4,
        ax=ax1,
    )

    ax1.set_title(f'Detalle por Réplica: {objeto_a_graficar}', fontsize=14)
    ax1.set_ylabel(f'{tipo_medida} {unidad}')
    ax1.set_xlabel('Réplica Individual')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, limite_inicial)

    stats1 = df_subset.groupby('ID_Muestra')['Medida_Valor'].agg(['mean', 'max']).reset_index()
    x_positions = {label.get_text(): i for i, label in enumerate(ax1.get_xticklabels())}

    for _, row in stats1.iterrows():
        id_muestra = row['ID_Muestra']
        if id_muestra in x_positions:
            y_pos_text = row['max'] * 1.05 + offset_text
            ax1.text(
                x_positions[id_muestra],
                y_pos_text,
                f"Media:\n{fmt.format(row['mean'])}",
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold',
                color='#333333',
            )

    ax_slider1 = fig1.add_axes([0.15, 0.1, 0.65, 0.03], facecolor='lightgray')
    slider1 = Slider(
        ax_slider1,
        'Zoom Y ',
        max(limite_inicial * 0.1, 0.1),
        max(limite_inicial * 3.0, 1.0),
        valinit=limite_inicial,
    )

    def update1_simple(val):
        ax1.set_ylim(0, val)
        fig1.canvas.draw_idle()

    slider1.on_changed(update1_simple)

    fig2, ax2 = plt.subplots(figsize=(6, 7))
    fig2.canvas.manager.set_window_title('Ventana 2: Comparación Global')
    plt.subplots_adjust(bottom=0.25)

    sns.boxplot(
        data=df_subset,
        x='Grupo',
        y='Medida_Valor',
        palette=paleta,
        showfliers=False,
        width=0.5,
        ax=ax2,
    )

    sns.stripplot(
        data=df_subset,
        x='Grupo',
        y='Medida_Valor',
        color='black',
        alpha=0.3,
        size=4,
        ax=ax2,
    )

    ax2.set_title(f'COMPARACIÓN FINAL\n{objeto_a_graficar} (Pool de datos)', fontsize=14, fontweight='bold')
    ax2.set_ylabel(f'{tipo_medida} {unidad}')
    ax2.set_xlabel('Condición Experimental')
    ax2.set_ylim(0, limite_inicial)

    stats2 = df_subset.groupby('Grupo')['Medida_Valor'].agg(mean='mean', ymax='max').reset_index()
    grupos_ordenados = [label.get_text() for label in ax2.get_xticklabels()]

    for i, grupo in enumerate(grupos_ordenados):
        row = stats2[stats2['Grupo'] == grupo]
        if not row.empty:
            val_mean = row.iloc[0]['mean']
            val_max = row.iloc[0]['ymax']
            y_pos_text = val_max * 1.05 + offset_text

            ax2.text(
                i,
                y_pos_text,
                f"Media: {fmt.format(val_mean)}",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color='black',
            )

    ax_slider2 = fig2.add_axes([0.15, 0.1, 0.65, 0.03], facecolor='lightgray')
    slider2 = Slider(
        ax_slider2,
        'Zoom Y ',
        max(limite_inicial * 0.1, 0.1),
        max(limite_inicial * 3.0, 1.0),
        valinit=limite_inicial,
    )

    def update2_simple(val):
        ax2.set_ylim(0, val)
        fig2.canvas.draw_idle()

    slider2.on_changed(update2_simple)

    print("\n[OK] Generadas 2 ventanas interactivas.")
    plt.show()
    return {
        "archivos_datos": [a[1] for a in archivos_multiples] if archivos_multiples else [archivo_datos],
        "objeto": objeto_a_graficar,
        "tipo_medida": tipo_medida,
        "cantidad_datos": len(df_subset),
    }


def main():
    graficar_datos()


if __name__ == "__main__":
    main()
