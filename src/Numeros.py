import pandas as pd
import numpy as np
from scipy import stats

# ==========================================
# ⚙️ CONFIGURACIÓN
# ==========================================
ARCHIVO = 'resultados_limpios.csv'

# ¿Qué queremos medir? 
# Cambiá a 'Area_cm2' si querés analizar área en vez de largo.
TIPO_MEDIDA = 'Largo_cm'  
TIPO_OBJETO = 'Radicula'

def analizar_estadisticas(
    archivo=ARCHIVO,
    tipo_medida=TIPO_MEDIDA,
    tipo_objeto=TIPO_OBJETO,
):
    print(f"\n[ANALISIS] INICIANDO ANÁLISIS ESTADÍSTICO ({tipo_medida})")
    print("=" * 60)

    try:
        # 1. Cargar datos
        df = pd.read_csv(archivo)
        
        # Limpieza de strings (sacar espacios en blanco)
        df['Grupo'] = df['Grupo'].astype(str).str.strip()
        df['Tipo_Objeto'] = df['Tipo_Objeto'].astype(str).str.strip()
        df['Medida_Tipo'] = df['Medida_Tipo'].astype(str).str.strip()
        
        # Asegurar numéricos
        df['Medida_Valor'] = pd.to_numeric(df['Medida_Valor'], errors='coerce')

        # 2. Filtrar lo que nos interesa
        df_subset = df[
            (df['Tipo_Objeto'] == tipo_objeto) & 
            (df['Medida_Tipo'] == tipo_medida)
        ].copy()

        if df_subset.empty:
            print(f"[ERROR] No encontré datos para '{tipo_objeto}' con medida '{tipo_medida}'.")
            print("Revisá que el CSV tenga datos de 'Largo_cm' o cambiá la configuración a 'Area_cm2'.")
            return None

        # ==========================================
        # ANÁLISIS 1: INDIVIDUAL (Promedio por Raíz)
        # ==========================================
        print("\n[PASO 1] ESTADÍSTICAS INDIVIDUALES (Promedio por Raíz)")
        print("-" * 60)
        
        stats_ind = df_subset.groupby('Grupo')['Medida_Valor'].agg(['count', 'mean', 'std'])
        stats_ind.columns = ['N (Raíces)', 'Media (cm)', 'Desv. Std (cm)']
        print(stats_ind.round(4))

        # ==========================================
        # ANÁLISIS 2: ACUMULADO (Biomasa por Réplica)
        # ==========================================
        print("\n[PASO 2] LONGITUD TOTAL ACUMULADA (Biomasa por Réplica)")
        print("-" * 60)
        
        df_replicas = df_subset.groupby(['Grupo', 'Replica'])['Medida_Valor'].sum().reset_index()
        
        stats_rep = df_replicas.groupby('Grupo')['Medida_Valor'].agg(['count', 'mean', 'std'])
        stats_rep.columns = ['N (Réplicas)', 'Total Promedio (cm)', 'Desv. Std Totales']
        print(stats_rep.round(4))
        
        print("\n    > Valores Totales por Réplica (Crudos):")
        print(df_replicas)

        # ==========================================
        # ANÁLISIS 3: TEST ANOVA (Significancia)
        # ==========================================
        print("\n[PASO 3] TEST DE HIPÓTESIS (ANOVA de un factor)")
        print("-" * 60)
        
        grupo_control = df_subset[df_subset['Grupo'] == 'Control']['Medida_Valor']
        grupos_tratamiento = df_subset[df_subset['Grupo'] != 'Control']['Grupo'].unique()
        
        for trat in grupos_tratamiento:
            grupo_mag = df_subset[df_subset['Grupo'] == trat]['Medida_Valor']
            
            # Ejecutar ANOVA
            f_val, p_val = stats.f_oneway(grupo_control, grupo_mag)
            
            print(f"COMPARACIÓN: Control vs {trat}")
            print(f"    > Valor F: {f_val:.4f}")
            print(f"    > P-Value: {p_val:.5f}")
            
            alpha = 0.05
            if p_val < alpha:
                print("    > VERDICTO: SIGNIFICATIVO (Hay diferencia real)")
                print(f"      Con 95% de confianza, el tratamiento {trat} afectó el crecimiento.")
            else:
                print("    > VERDICTO: NO SIGNIFICATIVO (Pudo ser azar)")
                print("      No hay evidencia suficiente para decir que son distintos.")
            print("-" * 30)

        return {
            "archivo": archivo,
            "tipo_objeto": tipo_objeto,
            "tipo_medida": tipo_medida,
            "cantidad_registros": len(df_subset),
            "tratamientos": list(grupos_tratamiento),
        }

    except FileNotFoundError:
        print(f"No encontré el archivo: {archivo}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
    return None

def main():
    analizar_estadisticas()

if __name__ == "__main__":
    main()
