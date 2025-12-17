import os
import re
import csv
import cv2 
import numpy as np 
from PIL import Image
import pytesseract
from collections import Counter 
import sys # Importado para leer argumentos de línea de comandos

# =========================================================================
# 🛑 CONFIGURACIÓN DE TESSERACT OCR & RUTAS
# =========================================================================
# 🛑 RUTA HARDCODEADA ELIMINADA: La ruta se configura dinámicamente al inicio
# pytesseract.pytesseract.tesseract_cmd se establecerá con sys.argv[2]

CARPETA_IMAGENES = 'fotos'
CSV_ENTRADA = 'resultados_coordenadas.csv' # Archivo de entrada (generado por proceso1)
CSV_SALIDA = 'resultados_coordenadas.csv'  # Archivo a sobrescribir
ESTATUS_FALLO = 'NO ENCONTRADO' 

# =========================================================================
# 📌 CONFIGURACIÓN Y PATRONES GLOBALES
# =========================================================================
RECORTE_PORCENTAJE = 0.40 # 40% inferior

LAT_MIN_ESPERADA = 14.0
LAT_MAX_ESPERADA = 33.0
LON_MIN_ESPERADA = -119.0
LON_MAX_ESPERADA = -86.0

RANGOS_CIUDADES = {
    'ENSENADA': (31.0, 33.0, -117.5, -115.5), 'TIJUANA': (32.0, 34.0, -117.5, -115.5), 
    'CHIHUAHUA': (27.5, 29.5, -107.0, -105.0), 'SALTILLO': (24.5, 26.5, -102.0, -100.0), 
    'CIUDAD VICTORIA': (23.5, 25.5, -99.5, -97.5), 'MONTERREY': (25.0, 27.0, -101.0, -99.0), 
    'NUEVO LAREDO': (27.0, 29.0, -100.5, -98.5), 'TAMPICO': (21.0, 23.0, -99.0, -97.0), 
    'GUADALAJARA': (20.0, 22.0, -104.5, -102.5), 'QUERETARO': (20.0, 22.0, -101.5, -99.5), 
    'SAN LUIS POTOSI': (21.0, 23.0, -102.0, -100.0), 'TOLUCA': (18.5, 20.5, -100.5, -98.5) 
}

# --- PATRONES ROBUSTOS ---
PATRON_DMS_ROBUSTO = r'(\d{1,2}°\d{1,2}\'\d{1,2}\")[\s,\-\/]*(-?\d{1,3}°\d{1,2}\'\d{1,2}\")'
PATRON_DECIMAL_ROBUSTO = r'(-?\d{1,3}\.\d+[NS])[\s,\-\/]*(-?\d{1,3}\.\d+[WE])' 

def obtener_rangos_por_ciudad(ciudad):
    """Devuelve los rangos específicos para la ciudad."""
    return RANGOS_CIUDADES.get(ciudad.upper())

# --------------------------------------------------------------------------
# I. FUNCIONES AUXILIARES 
# --------------------------------------------------------------------------

def separar_por_posicion(nombre_archivo_base):
    """Separa el nombre del archivo en OT (primeros 8 caracteres) y el resto."""
    ot = nombre_archivo_base[:8]
    resto_nombre = nombre_archivo_base[8:]
    return ot, resto_nombre

def convertir_a_decimal(coord_con_cardinal):
    """Convierte una coordenada con Cardinal/Signo (ej. 25.82588N o -100.555) a Decimal puro."""
    if not coord_con_cardinal or coord_con_cardinal == 'FALLO': return None
    coord_con_cardinal = str(coord_con_cardinal).strip()
    if '°' in coord_con_cardinal: return None 
    signo = -1 if coord_con_cardinal.startswith('-') else 1
    cardinal = coord_con_cardinal[-1].upper() if coord_con_cardinal and coord_con_cardinal[-1].isalpha() else None
    valor_str = coord_con_cardinal[:-1] if cardinal else coord_con_cardinal
    valor_str = valor_str.replace(',', '.').replace('-', '').strip() 
    try: valor = float(valor_str)
    except ValueError: return None
    if cardinal in ('S', 'W'): return -valor
    return valor * signo

def corregir_latitud_ocr(lat_extraida, lon_extraida, ciudad_rangos):
    """Aplica la corrección heurística."""
    if not lat_extraida or not ciudad_rangos: return lat_extraida, lon_extraida
    match_lat = re.match(r'(\d)[.,]', lat_extraida)
    if not match_lat: return lat_extraida, lon_extraida
    primer_digito_encontrado = match_lat.group(1)
    parte_entera_esperada = str(int(ciudad_rangos[0])) 
    digito_esperado = parte_entera_esperada[0] 
    digitos_erroneos_de_2 = ['9', '7', '5', '0']
    
    if primer_digito_encontrado in digitos_erroneos_de_2 and digito_esperado in ['1', '2', '3']: 
        lat_corregida = lat_extraida
        if primer_digito_encontrado == '0' and parte_entera_esperada.startswith('20'):
            lat_corregida = '20' + lat_extraida[1:]
        elif primer_digito_encontrado == '0':
            lat_corregida = digito_esperado + lat_extraida[1:]
        else:
            lat_corregida = digito_esperado + lat_extraida[1:]
        
        print(f" 	⚙️ CORRECCIÓN HEURÍSTICA: '{lat_extraida}' ajustado a '{lat_corregida}'.")
        return lat_corregida, lon_extraida
        
    return lat_extraida, lon_extraida

def validar_rango_geografico(latitud_dec, longitud_dec):
    if latitud_dec is None or longitud_dec is None: return False
    lat_valida = -90.0 <= latitud_dec <= 90.0
    lon_valida = -180.0 <= longitud_dec <= 180.0
    return lat_valida and lon_valida

def validar_rango_proximidad(lat_dec, lon_dec, lat_min, lat_max, lon_min, lon_max):
    if lat_dec is None or lon_dec is None: return False
    lat_en_rango = lat_min <= lat_dec <= lat_max
    lon_en_rango = lon_min <= lon_dec <= lon_max
    return lat_en_rango and lon_en_rango

def exportar_a_csv(datos, nombre_archivo):
    """Exporta los resultados a un archivo CSV."""
    campos = ['OT', 'Resto_Nombre', 'Latitud_Extraida', 'Longitud_Extraida', 'Latitud_Decimal', 'Longitud_Decimal', 'Estatus', 'Metodo_Extraccion']
    try:
        with open(nombre_archivo, 'w', newline='', encoding='utf-8') as archivo_csv:
            # Añadimos 'Metodo_Extraccion' al escritor de columnas
            escritor = csv.DictWriter(archivo_csv, fieldnames=campos, restval='') 
            escritor.writeheader()
            escritor.writerows(datos)
        print(f"\n✅ EXPORTACIÓN EXITOSA: Los resultados se guardaron en {nombre_archivo}")
    except Exception as e:
        print(f"\n❌ ERROR DE EXPORTACIÓN: No se pudo escribir el archivo CSV. {e}")

# --------------------------------------------------------------------------
# II. MOTOR DE DETECCIÓN MULTINIVEL (ROBUSTO)
# --------------------------------------------------------------------------

def reconocer_y_extraer_robusto(img_pil, config_ocr='--psm 6'):
    """Aplica OCR y usa patrones robustos de DMS y Decimal."""
    try:
        texto_extraido = pytesseract.image_to_string(img_pil, lang='eng', config=config_ocr)
        
        # 1. Intento Decimal
        match_dec = re.search(PATRON_DECIMAL_ROBUSTO, texto_extraido, re.IGNORECASE)
        if match_dec:
              lat, lon = match_dec.groups()
              return lat.strip().upper(), lon.strip().upper(), "Patron_Decimal_Robusto"

        # 2. Intento DMS
        match_dms = re.search(PATRON_DMS_ROBUSTO, texto_extraido)
        if match_dms:
              lat, lon = match_dms.groups()
              return lat.strip(), lon.strip(), "Patron_DMS_Robusto"
        
    except Exception:
        pass
    return None, None, "Fallo_Patron"

def preprocesar_otsu(img_cv):
    """Aplica Binarización de Otsu para alto contraste."""
    if img_cv is None or img_cv.size == 0: return None
    gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binaria

def intento_multinivel_robusto(img_cv_recortada, nombre_debug=""):
    """
    Estrategia de dos niveles: RAW (crudo) y PROCESADO (Otsu).
    Guarda imágenes de debug si falla.
    """

    # --- INTENTO 1: RAW OCR (Sin preprocesamiento) con PSM 6 ---
    img_pil_raw = Image.fromarray(cv2.cvtColor(img_cv_recortada, cv2.COLOR_BGR2RGB))
    lat, lon, metodo = reconocer_y_extraer_robusto(img_pil_raw, config_ocr='--psm 6')
    if lat and lon: return lat, lon, f"{metodo}_RAW_PSM6"

    # --- INTENTO 2: RAW OCR (Sin preprocesamiento) con PSM 11 (Sparse Text) ---
    lat, lon, metodo = reconocer_y_extraer_robusto(img_pil_raw, config_ocr='--psm 11')
    if lat and lon: return lat, lon, f"{metodo}_RAW_PSM11"

    # --- INTENTO 3: PROCESADO OCR (Binarización Otsu) ---
    img_procesada = preprocesar_otsu(img_cv_recortada)
    if img_procesada is None:
        if nombre_debug:
            Image.fromarray(cv2.cvtColor(img_cv_recortada, cv2.COLOR_BGR2RGB)).save(os.path.join('diagnostico_ocr', f"DEBUG_{nombre_debug}_Recorte.png"))
        return None, None, "Error_Preprocesamiento"
        
    img_pil_procesada = Image.fromarray(img_procesada)
    lat, lon, metodo = reconocer_y_extraer_robusto(img_pil_procesada, config_ocr='--psm 6')
    if lat and lon: return lat, lon, f"{metodo}_PROCESADO_OTSU_PSM6"

    # Si falla, guardar imágenes de diagnóstico
    if nombre_debug:
        ruta_debug = os.path.join('diagnostico_ocr', f"DEBUG_{nombre_debug}_RAW_Fallido.png")
        img_pil_raw.save(ruta_debug)
        ruta_debug_proc = os.path.join('diagnostico_ocr', f"DEBUG_{nombre_debug}_Otsu_Fallido.png")
        img_pil_procesada.save(ruta_debug_proc)
        print(f" 	(DEBUG: Imágenes de diagnóstico guardadas en '{os.path.basename(ruta_debug)}' y '{os.path.basename(ruta_debug_proc)}')")

    return None, None, "Fallo_OCR_Final"


# --------------------------------------------------------------------------
# III. FUNCIÓN PRINCIPAL DE PROCESAMIENTO (FILTRADO POR CSV)
# --------------------------------------------------------------------------

def procesar_fallas_csv(ciudad_seleccionada):
    """Lee el CSV, identifica fallas, procesa SOLO esas imágenes y actualiza el CSV."""
    
    # 0. CONFIGURACIÓN DE CONTEXTO DE CIUDAD (Usando el valor recibido por sys.argv)
    print("\n\n--- CONTEXTO DE CIUDAD ---")
    print(f"[proceso3.py] Procesando con el contexto de CIUDAD: {ciudad_seleccionada}")
    
    rangos_especificos = obtener_rangos_por_ciudad(ciudad_seleccionada)
    
    if rangos_especificos:
        lat_min, lat_max, lon_min, lon_max = rangos_especificos
        print(f"✅ Contexto cargado: {ciudad_seleccionada}. Rango: Lat {lat_min}° a {lat_max}°")
    else:
        print(f"⚠️ Ciudad '{ciudad_seleccionada}' no reconocida. Usando rango general de México.")
        lat_min, lat_max, lon_min, lon_max = LAT_MIN_ESPERADA, LAT_MAX_ESPERADA, LON_MIN_ESPERADA, LON_MAX_ESPERADA

    print("--- 🚀 Iniciando Reprocesamiento de Fallas (Multinivel Robusto) ---")
    
    filas_originales = []
    archivos_a_reprocesar_indices = []
    
    # 1. LEER CSV y FILTRAR por ESTATUS_FALLO
    try:
        with open(CSV_ENTRADA, 'r', newline='', encoding='utf-8') as archivo_csv:
            lector = csv.DictReader(archivo_csv)
            # Asegúrate de que todos los campos esperados existan
            # Añadido Metodo_Extraccion para consistencia si no existe
            campos_esperados = ['OT', 'Resto_Nombre', 'Latitud_Extraida', 'Longitud_Extraida', 'Latitud_Decimal', 'Longitud_Decimal', 'Estatus', 'Metodo_Extraccion']
            
            for i, fila in enumerate(lector):
                # Rellenar campos faltantes con valores por defecto 
                fila_completa = {campo: fila.get(campo, '') for campo in campos_esperados}
                filas_originales.append(fila_completa)
                
                if fila_completa.get('Estatus') == ESTATUS_FALLO:
                    archivos_a_reprocesar_indices.append(i)
    except FileNotFoundError:
        print(f"❌ Error: El archivo de entrada '{CSV_ENTRADA}' no se encontró. Asegúrate de que exista y que proceso1.py lo haya generado.")
        return
    except Exception as e:
        print(f"❌ Error crítico leyendo CSV: {e}")
        return

    print(f"📊 Total registros: {len(filas_originales)} | Fallas a revisar: {len(archivos_a_reprocesar_indices)}")
    
    exitos = 0
    total_procesado = 0
    
    # 2. PROCESAR SOLO LAS FILAS FILTRADAS
    for index in archivos_a_reprocesar_indices:
        fila = filas_originales[index]
        total_procesado += 1
        
        ot_raw = fila.get('OT', '').strip()
        resto = fila.get('Resto_Nombre', '').strip()
        
        ot_pad = ot_raw.zfill(8) 
        nombre_base = ot_pad + resto 

        ruta_imagen_encontrada = None
        
        print(f"\n[{total_procesado}/{len(archivos_a_reprocesar_indices)}] ⏳ Analizando Archivo: {nombre_base}")

        # Búsqueda de la imagen
        for ext in ('.jpg', '.jpeg', '.png', '.tiff'):
            ruta_candidata = os.path.join(CARPETA_IMAGENES, nombre_base + ext)
            if os.path.exists(ruta_candidata):
                ruta_imagen_encontrada = ruta_candidata
                break
        
        if not ruta_imagen_encontrada:
            fila['Metodo_Extraccion'] = 'Error_Ruta_Final'
            print(f" 	❌ Resultado: Imagen no encontrada en carpeta 'fotos'.")
            continue 
            
        try:
            img_cv_original = cv2.imread(ruta_imagen_encontrada)
            if img_cv_original is None: 
                print(f" 	❌ Error: No se pudo cargar la imagen OpenCV desde '{ruta_imagen_encontrada}'.")
                continue

            # Recorte al 40% inferior
            height, width = img_cv_original.shape[:2]
            y_start = int(height * (1 - RECORTE_PORCENTAJE))
            img_cv_recortada = img_cv_original[y_start:height, 0:width]
            
            # EJECUTAR ANÁLISIS MULTINIVEL
            lat_ext, lon_ext, metodo = intento_multinivel_robusto(img_cv_recortada, nombre_debug=nombre_base)

            if lat_ext and lon_ext and "Fallo_OCR_Final" not in metodo:
                
                # APLICAR CORRECCIÓN HEURÍSTICA Y CONVERSIÓN
                lat_ext, lon_ext = corregir_latitud_ocr(lat_ext, lon_ext, rangos_especificos)
                lat_dec = convertir_a_decimal(lat_ext)
                lon_dec = convertir_a_decimal(lon_ext)
                
                # Inicializar variables finales con el resultado extraído
                lat_ext_final, lon_ext_final = lat_ext, lon_ext
                lat_dec_final, lon_dec_final = lat_dec, lon_dec
                estado = "NO ENCONTRADO" # Por defecto
                
                # VALIDACIÓN
                if 'Patron_DMS' in metodo:
                    # Si es DMS, se acepta, pero sin valores decimales completos
                    estado = "CORRECTO" 
                    lat_dec_final = '' 
                    lon_dec_final = '' 
                    print(f" 	✔️ ÉXITO: {metodo} | Coordenadas: {lat_ext}, {lon_ext} | Estatus: CORRECTO (DMS)")
                
                elif validar_rango_geografico(lat_dec, lon_dec) and validar_rango_proximidad(lat_dec, lon_dec, lat_min, lat_max, lon_min, lon_max):
                    
                    # --- ÉXITO (Decimal validado) ---
                    estado = "CORRECTO"
                    # lat_dec_final y lon_dec_final ya tienen los valores correctos
                    print(f" 	✔️ ÉXITO: {metodo} | Coordenadas: {lat_ext}, {lon_ext} | Estatus: CORRECTO")
                else:
                    # --- FALLO DE VALIDACIÓN: COORDENADAS FUERA DE RANGO ---
                    
                    # 1. Documentar el método de extracción añadiendo el sufijo _FUERA_RANGO
                    metodo = f"{metodo}_FUERA_RANGO"
                    
                    # 2. Mantener el Estatus como NO ENCONTRADO (dato no válido)
                    estado = "NO ENCONTRADO"
                    
                    # 3. CONSERVAR las coordenadas extraídas para revisión manual. 
                    print(f" 	❌ FALLA: {metodo} | Coordenadas extraídas: {lat_ext_final}, {lon_ext_final} | Estatus: NO ENCONTRADO (Fuera de Rango)")
                    
            else:
                # --- FALLO DE OCR ---
                print(f" 	❌ FALLA: {metodo} | Estatus: NO ENCONTRADO")
                # Si falló el OCR, sí se sobrescriben con FALLO y vacío.
                lat_ext_final, lon_ext_final = 'FALLO', 'FALLO'
                lat_dec_final, lon_dec_final = '', ''

            # Actualizar la fila en la lista principal
            filas_originales[index]['Latitud_Extraida'] = lat_ext_final
            filas_originales[index]['Longitud_Extraida'] = lon_ext_final
            filas_originales[index]['Latitud_Decimal'] = lat_dec_final
            filas_originales[index]['Longitud_Decimal'] = lon_dec_final
            filas_originales[index]['Estatus'] = estado
            filas_originales[index]['Metodo_Extraccion'] = metodo
            
            if estado == "CORRECTO":
                exitos += 1

        except Exception as e:
            print(f" 	❌ Error en el procesamiento de '{nombre_base}': {e}")
            filas_originales[index]['Metodo_Extraccion'] = f'ERROR_CRITICO: {e}'

    # 3. GUARDAR TODAS LAS FILAS ACTUALIZADAS EN EL CSV DE SALIDA
    exportar_a_csv(filas_originales, CSV_SALIDA)
    print(f"\n✨ REPORTE FINAL: Se recuperaron {exitos} coordenadas adicionales.")

# --------------------------------------------------------------------------
# IV. CONFIGURACIÓN DE EJECUCIÓN (MODIFICADA PARA RECIBIR ARGUMENTOS)
# --------------------------------------------------------------------------

if __name__ == '__main__':
    # Creación de la carpeta de diagnóstico si no existe
    if not os.path.exists('diagnostico_ocr'):
        os.makedirs('diagnostico_ocr')
    
    try:
        # sys.argv[1] es la Ciudad (Argumento 1)
        # sys.argv[2] es la Ruta de Tesseract (Argumento 2)
        if len(sys.argv) < 3:
            raise IndexError("Se esperaban los argumentos de CIUDAD y RUTA TESSERACT.")
            
        ciudad_input = sys.argv[1].upper() 
        tesseract_path = sys.argv[2] # Nuevo argumento
        
        # Configurar Tesseract con la ruta dinámica
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"[proceso3.py] Configurando Tesseract con la ruta dinámica: '{tesseract_path}'")

        print(f"[proceso3.py] Recibida CIUDAD desde el orquestador: '{ciudad_input}'")
        procesar_fallas_csv(ciudad_input)
    except IndexError as e:
        print(f"[proceso3.py] ERROR: {e} Se esperaba el argumento de la CIUDAD y la RUTA TESSERACT desde 'app.py'.")
        sys.exit(1)
    except Exception as e:
        print(f"[proceso3.py] ERROR: Fallo al ejecutar el proceso 3. {e}")
        sys.exit(1)