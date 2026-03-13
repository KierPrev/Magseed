import pandas as pd
import numpy as np

# ==========================================
# CONFIGURACIÓN
# ==========================================
ARCHIVO_ENTRADA = 'resultados_detallados_calibrados.csv'
ARCHIVO_SALIDA = 'resultados_limpios.csv'

# Factor de agresividad para borrar outliers.
# 1.5 es el estándar estadístico (borra lo "raro").
# 3.0 es más conservador (borra solo lo "extremadamente raro").
FACTOR_IQR = 1.5 

def main():
    print(f"--- INICIANDO LIMPIEZA DE DATOS ({ARCHIVO_ENTRADA}) ---")
    
    try:
        df = pd.read_csv(ARCHIVO_ENTRADA)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ARCHIVO_ENTRADA}")
        return

    # Limpieza inicial de columnas de texto
    cols_texto = ['Grupo', 'Tipo_Objeto', 'Medida_Tipo']
    for col in cols_texto:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Asegurar que los valores sean numéricos
    df['Medida_Valor'] = pd.to_numeric(df['Medida_Valor'], errors='coerce')
    
    # Eliminamos filas que no tengan valor numérico (errores de lectura)
    df_inicial = df.dropna(subset=['Medida_Valor'])
    total_inicial = len(df_inicial)
    
    print(f"Datos totales leídos: {total_inicial}")

    # ==========================================
    # LÓGICA DE DETECCIÓN DE OUTLIERS
    # ==========================================
    # Agrupamos los datos para no mezclar peras con manzanas.
    # Calculamos outliers independientemente para cada combinación de:
    # Grupo (Control/Mag) + Tipo (Semilla/Raiz) + Medida (Area/Largo)
    
    grupos = df_inicial.groupby(['Grupo', 'Tipo_Objeto', 'Medida_Tipo'])
    
    indices_validos = []
    reporte_eliminados = []

    for nombre_grupo, datos_grupo in grupos:
        grupo_str = f"{nombre_grupo[0]} - {nombre_grupo[1]} ({nombre_grupo[2]})"
        
        # Si hay muy pocos datos, no podemos calcular estadísticas fiables
        if len(datos_grupo) < 5:
            indices_validos.extend(datos_grupo.index)
            continue

        # 1. Calcular Cuartiles
        Q1 = datos_grupo['Medida_Valor'].quantile(0.25)
        Q3 = datos_grupo['Medida_Valor'].quantile(0.75)
        IQR = Q3 - Q1
        
        # 2. Definir Límites (Bigotes)
        limite_inferior = Q1 - (FACTOR_IQR * IQR)
        limite_superior = Q3 + (FACTOR_IQR * IQR)
        
        # 3. Filtrar
        mascara_validos = (datos_grupo['Medida_Valor'] >= limite_inferior) & \
                          (datos_grupo['Medida_Valor'] <= limite_superior)
        
        # Guardamos los índices de las filas que SE QUEDAN
        indices_grupo_validos = datos_grupo[mascara_validos].index
        indices_validos.extend(indices_grupo_validos)
        
        # Reporte
        eliminados = len(datos_grupo) - len(indices_grupo_validos)
        if eliminados > 0:
            reporte_eliminados.append((grupo_str, eliminados, len(datos_grupo)))

    # ==========================================
    # GUARDAR RESULTADOS
    # ==========================================
    df_final = df_inicial.loc[indices_validos].copy()
    
    print("\n--- REPORTE DE LIMPIEZA ---")
    if reporte_eliminados:
        print(f"{'GRUPO':<40} | {'ELIMINADOS':<10} | {'TOTAL PREVIO':<10}")
        print("-" * 70)
        for nombre, elim, total in reporte_eliminados:
            print(f"{nombre:<40} | {elim:<10} | {total:<10}")
    else:
        print("No se encontraron outliers estadísticos.")

    total_final = len(df_final)
    borrados_totales = total_inicial - total_final
    porcentaje = (borrados_totales / total_inicial) * 100
    
    print("-" * 70)
    print(f"Total inicial: {total_inicial}")
    print(f"Total final:   {total_final}")
    print(f"Se eliminaron {borrados_totales} datos ruidosos ({porcentaje:.2f}%)")
    
    df_final.to_csv(ARCHIVO_SALIDA, index=False)
    print(f"\n[OK] Archivo limpio guardado como: {ARCHIVO_SALIDA}")
    print("Ahora puedes usar este archivo nuevo en tus gráficas.")

if __name__ == "__main__":
    main()