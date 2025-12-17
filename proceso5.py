import pandas as pd
import numpy as np
import os
import sys
import csv

# --- CONFIGURACIÓN DE ARCHIVOS ---
# 🛑 CORREGIDO: Usar el archivo de resultados consolidado
ARCHIVO_ORIGEN = 'resultados_coordenadas.csv' 
ARCHIVO_ELEMENTOS = 'elementos.csv'
ARCHIVO_SALIDA = 'resultados_distancia_final_completo.csv'

# --- FÓRMULA DE HAVERSINE ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000 # Radio de la Tierra en metros
    lat1_rad = np.radians(lat1); lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2); lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad; dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# --- FUNCIÓN PRINCIPAL DE PROCESAMIENTO ---

def procesar_archivos():
    print(f"Iniciando procesamiento de {ARCHIVO_ORIGEN} y {ARCHIVO_ELEMENTOS}...")
    
    # 1. Cargar datos y verificar existencia de archivos
    archivos_requeridos = [ARCHIVO_ORIGEN, ARCHIVO_ELEMENTOS]
    for archivo in archivos_requeridos:
        if not os.path.exists(archivo):
            print(f"❌ ERROR: El archivo '{archivo}' no existe. Por favor, asegúrate de que esté en la carpeta y que los procesos anteriores lo hayan generado.")
            
            # Si el error es elementos.csv, simular su creación para que el proceso no falle inmediatamente.
            if archivo == ARCHIVO_ELEMENTOS:
                try:
                    with open(ARCHIVO_ELEMENTOS, 'w', encoding='utf-8', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['elemento', 'Latitud', 'Longitud', 'segmento'])
                        writer.writerow(['EJEMPLO', '20.0', '-103.0', 'TEST_SEG'])
                    print(f"⚠️ ADVERTENCIA: '{ARCHIVO_ELEMENTOS}' no fue encontrado. Se creó un archivo simulado para evitar un fallo total.")
                except Exception as e:
                    print(f"❌ ERROR: No se pudo crear el archivo simulado '{ARCHIVO_ELEMENTOS}'. {e}")
                    return

    try:
        # Intentar cargar archivos
        df_origen = pd.read_csv(ARCHIVO_ORIGEN)
        # Usamos 'utf-8' como principal, luego 'latin-1' como fallback para 'elementos.csv'
        try:
            df_elementos = pd.read_csv(ARCHIVO_ELEMENTOS, encoding='utf-8')
        except UnicodeDecodeError:
            df_elementos = pd.read_csv(ARCHIVO_ELEMENTOS, encoding='latin-1')
            
    except FileNotFoundError:
        # Este error solo debería ocurrir si la verificación anterior falló
        print(f"❌ ERROR: No se pudieron cargar los archivos.")
        return
    except pd.errors.EmptyDataError:
        print("❌ ERROR: Uno de los archivos está vacío. Verifica que contengan datos.")
        return
    except UnicodeDecodeError:
        print(f"❌ ERROR de Decodificación: El archivo '{ARCHIVO_ELEMENTOS}' no pudo leerse. Verifica su codificación.")
        return

    # 2. Preparar el DataFrame de Origen (df_origen)
    print("Preparando datos de origen...")
    
    filas_originales = len(df_origen) 

    df_origen['ot'] = df_origen['OT'].astype(str).str.strip()
    df_origen['coordenada'] = df_origen['Latitud_Decimal'].astype(str) + ',' + df_origen['Longitud_Decimal'].astype(str)
    
    # Asegurar que la columna 'elemento' para el merge tenga el mismo nombre en ambos DataFrames
    # Usaremos una columna intermedia para mantener el nombre original de la imagen
    df_origen['elemento_merge'] = df_origen['Resto_Nombre'].astype(str).str.replace('_', '', regex=False).str.strip()
    
    # Definición de estatus para registros filtrados (NO ENCONTRADO en procesos OCR)
    df_origen['estatus_match'] = np.where(
        df_origen['Estatus'] == 'NO ENCONTRADO', 
        'FILTRADO POR ESTATUS', 
        ''
    )
    
    df_calc = df_origen[df_origen['estatus_match'] != 'FILTRADO POR ESTATUS'].copy()

    # 2.1 Limpieza y preparación de df_calc
    df_calc = df_calc.rename(columns={'Latitud_Decimal': 'latitud_origen', 'Longitud_Decimal': 'longitud_origen'})
    
    df_calc['latitud_origen'] = pd.to_numeric(df_calc['latitud_origen'], errors='coerce')
    df_calc['longitud_origen'] = pd.to_numeric(df_calc['longitud_origen'], errors='coerce')
    
    df_calc_valid = df_calc.dropna(subset=['latitud_origen', 'longitud_origen', 'elemento_merge', 'ot'])
    
    print(f" 	Registros originales: {filas_originales}. Registros a procesar: {len(df_calc_valid)}.")


    # 3. Preparar el DataFrame de Elementos
    print("Preparando datos de elementos...")
    
    # 🛑 Asegurar que la columna de merge sea 'elemento_merge' para la consistencia
    df_elementos = df_elementos.rename(columns={
        'Latitud': 'latitud_elemento', 
        'Longitud': 'longitud_elemento', 
        'segmento': 'segmento_elemento',
        'elemento': 'elemento_merge' # Asegurar que la columna de merge sea la misma
    })
    
    df_elementos['coordenada_elemento'] = df_elementos['latitud_elemento'].astype(str) + ',' + df_elementos['longitud_elemento'].astype(str)
    df_elementos['latitud_elemento'] = pd.to_numeric(df_elementos['latitud_elemento'], errors='coerce')
    df_elementos['longitud_elemento'] = pd.to_numeric(df_elementos['longitud_elemento'], errors='coerce')

    # 4. Realizar la Unión (Merge)
    print("Realizando la comparación (se permite duplicación temporal)...")
    
    calc_cols = ['ot', 'elemento_merge', 'coordenada', 'latitud_origen', 'longitud_origen', 'estatus_match']
    elementos_cols = ['elemento_merge', 'coordenada_elemento', 'segmento_elemento', 'latitud_elemento', 'longitud_elemento']
    
    # Merge usando la columna consistente 'elemento_merge'
    df_merged = pd.merge(df_calc_valid[calc_cols], df_elementos[elementos_cols], on='elemento_merge', how='left')

    # 5. Cálculo de Distancia
    print("Calculando distancias (Haversine)...")
    
    valid_coords_mask = df_merged['latitud_origen'].notna() & df_merged['longitud_origen'].notna() & \
                        df_merged['latitud_elemento'].notna() & df_merged['longitud_elemento'].notna()
    
    df_merged['distancia_float'] = np.nan
    
    df_merged.loc[valid_coords_mask, 'distancia_float'] = haversine(
        df_merged.loc[valid_coords_mask, 'latitud_origen'], df_merged.loc[valid_coords_mask, 'longitud_origen'],
        df_merged.loc[valid_coords_mask, 'latitud_elemento'], df_merged.loc[valid_coords_mask, 'longitud_elemento']
    )
    
    # 6. Desduplicación y Selección del Mejor Match
    print("Seleccionando el match más cercano (distancia mínima) para cada registro...")
    
    df_with_match = df_merged.dropna(subset=['distancia_float']).copy()
    
    match_output_cols_base = ['ot', 'coordenada', 'coordenada_elemento', 'distancia_float', 'segmento_elemento']
    
    if df_with_match.empty:
        df_best_match = pd.DataFrame(columns=match_output_cols_base + ['estatus_match'])
    else:
        # Agrupamos por ot y coordenada (origen) para encontrar la distancia mínima
        idx = df_with_match.groupby(['ot', 'coordenada'])['distancia_float'].idxmin()
        df_best_match = df_with_match.loc[idx].copy()
        
        df_best_match['estatus_match_y'] = 'MATCH ENCONTRADO' # Usaremos '_y' para el merge
        df_best_match = df_best_match[match_output_cols_base + ['estatus_match_y']]

    # 7. Reincorporar TODOS los registros y definir el Estatus Final
    
    # Unir el df_origen completo (estatus_match_x) con los resultados del mejor match (estatus_match_y)
    df_final = pd.merge(
        df_origen,
        df_best_match.rename(columns={'distancia_float': 'distancia_float_y', 'coordenada_elemento': 'coordenada_elemento_y', 'segmento_elemento': 'segmento_elemento_y'}),
        on=['ot', 'coordenada'],
        how='left'
    )
    
    # Consolidar estatus
    df_final['estatus_match_final'] = np.where(
        df_final['estatus_match'] == 'FILTRADO POR ESTATUS', # si fue filtrado por fallo OCR
        'FILTRADO POR ESTATUS',
        np.where(
            df_final['estatus_match_y'] == 'MATCH ENCONTRADO',
            'MATCH ENCONTRADO',
            'NO MATCH (NO HAY COINCIDENCIA DE ELEMENTO)'
        )
    )
    
    # 🌟 CORRECCIÓN CLAVE: Convertir la columna a float antes de redondear
    df_final['distancia'] = pd.to_numeric(df_final['distancia_float_y'], errors='coerce').round(2)
    
    # 🌟 CREACIÓN DE LA COLUMNA DE VALIDACIÓN DE DISTANCIA
    df_final['validacion_distancia'] = np.where(
        (df_final['estatus_match_final'] == 'MATCH ENCONTRADO') & (df_final['distancia'] > 50),
        'excede distancia',
        np.where(
             df_final['estatus_match_final'] == 'MATCH ENCONTRADO',
            'ok',
            'N/A'
        )
    )
    
    # 8. Generar el Resultado Final y Exportar
    
    df_final['elemento'] = df_final['Resto_Nombre'].astype(str).str.replace('_', '', regex=False).str.strip()
    
    # Mapeo de columnas para la salida
    df_salida = pd.DataFrame()
    df_salida['OT'] = df_final['OT']
    df_salida['Resto_Nombre'] = df_final['Resto_Nombre']
    df_salida['Latitud_Extraida'] = df_final['Latitud_Extraida']
    df_salida['Longitud_Extraida'] = df_final['Longitud_Extraida']
    df_salida['Latitud_Decimal'] = df_final['Latitud_Decimal']
    df_salida['Longitud_Decimal'] = df_final['Longitud_Decimal']
    df_salida['Estatus_OCR'] = df_final['Estatus'] # Cambiar a Estatus_OCR
    df_salida['Metodo_Extraccion'] = df_final['Metodo_Extraccion']
    
    # Columnas de Match
    df_salida['coordenada_match'] = df_final['coordenada_elemento_y'].fillna('N/A')
    df_salida['distancia_metros'] = df_final['distancia'].fillna('N/A')
    df_salida['segmento_match'] = df_final['segmento_elemento_y'].fillna('N/A')
    df_salida['Estatus_Match'] = df_final['estatus_match_final']
    df_salida['validacion_distancia'] = df_final['validacion_distancia']
    
    df_salida.to_csv(ARCHIVO_SALIDA, index=False)

    print("-" * 50)
    print(f"✅ ¡Procesamiento completado con éxito!")
    print(f"El archivo de salida '{ARCHIVO_SALIDA}' ha sido generado.")
    print(f"Registros totales del Origen: {filas_originales}")
    print(f"Registros de Salida generados: {len(df_salida)}")
    print("-" * 50)

if __name__ == "__main__":
    procesar_archivos()