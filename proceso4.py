import os
import re
import csv
import cv2 
import numpy as np 
from PIL import Image, ImageDraw, ImageFont # Módulos para dibujar en imágenes
import pytesseract
from collections import Counter 
import sys # 🛑 Importado para leer argumentos de línea de comandos

# =========================================================================
# 🛑 CONFIGURACIÓN DE TESSERACT OCR & RUTAS
# =========================================================================
# 🛑 RUTA REMOVIDA: La ruta se recibe como argumento de línea de comandos.

CARPETA_IMAGENES = 'fotos'
# 🛑 Ajustado para usar el archivo secuencial de resultados
CSV_ENTRADA = 'resultados_coordenadas.csv' 
CSV_SALIDA = 'resultados_coordenadas.csv' 
ESTATUS_FALLO = 'NO ENCONTRADO' 
ESTATUS_EXITO = 'CORRECTO' # Definir el estado de éxito

# =========================================================================
# 📌 CONFIGURACIÓN Y PATRONES GLOBALES
# =========================================================================
RECORTE_PORCENTAJE = 0.40 # 40% inferior
FACTOR_ESCALA_OCR = 0.50 # Factor de reducción de escala para el preprocesamiento OCR (50%)

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
    """Convierte una coordenada en formato Decimal con Cardinal/Signo (ej. 25.82588N o -100.555) a Decimal puro."""
    if not coord_con_cardinal or coord_con_cardinal == 'FALLO': return None
    coord_con_cardinal = str(coord_con_cardinal).strip()
    if '°' in coord_con_cardinal: return None 
    signo = -1 if coord_con_cardinal.startswith('-') else 1
    cardinal = coord_con_cardinal[-1].upper() if coord_con_cardinal and coord_con_cardinal[-1].isalpha() else None
    valor_str = coord_con_cardinal[:-1] if cardinal else coord_con_cardinal
    valor_str = valor_str.replace(',', '.').replace('-', '').strip() 
    try: 
        valor = float(valor_str)
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
            escritor = csv.DictWriter(archivo_csv, fieldnames=campos, restval='') 
            escritor.writeheader()
            escritor.writerows(datos)
        print(f"\n✅ EXPORTACIÓN EXITOSA: Los resultados se guardaron en {nombre_archivo}")
    except Exception as e:
        print(f"\n❌ ERROR DE EXPORTACIÓN: No se pudo escribir el archivo CSV. {e}")

def reducir_escala(img_cv, factor_escala):
    """Reduce la escala de la imagen de OpenCV para mejorar el ruido y la calidad del OCR."""
    if img_cv is None or img_cv.size == 0:
        return None
    
    nuevo_ancho = int(img_cv.shape[1] * factor_escala)
    nuevo_alto = int(img_cv.shape[0] * factor_escala)
    
    img_redimensionada = cv2.resize(img_cv, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_CUBIC)
    
    return img_redimensionada

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

def dibujar_coordenadas_en_imagen(img_pil, lat_str, lon_str):
    """Superpone las coordenadas extraídas en una imagen PIL para diagnóstico."""
    draw = ImageDraw.Draw(img_pil)
    try:
        # Intenta usar una fuente de sistema
        font = ImageFont.truetype("arial.ttf", 20) 
    except IOError:
        # Usa la fuente por defecto si no encuentra arial
        font = ImageFont.load_default()
        
    text = f"DETECTADO (FUERA DE RANGO):\nLat: {lat_str}\nLon: {lon_str}"
    
    # Coordenadas de posición (esquina inferior izquierda)
    x_pos = 10
    y_pos = img_pil.height - 70 
    
    # Dibuja la caja de fondo (negro semitransparente)
    draw.rectangle([(x_pos - 5, y_pos - 5), (x_pos + 300, y_pos + 60)], fill=(0, 0, 0, 150))
    # Dibuja el texto (Rojo para destacar el error)
    draw.text((x_pos, y_pos), text, font=font, fill=(255, 0, 0)) 
    return img_pil

def intento_multinivel_robusto(img_cv_recortada, nombre_debug=""):
    """
    Estrategia de tres niveles: ESCALADO, RAW (crudo) y PROCESADO (Otsu).
    Guarda imágenes de debug si falla.
    """
    
    # ----------------------------------------------------------------------
    # --- INTENTO 0: ESCALADO 50% y RAW OCR (Optimización de ruido) ---
    # ----------------------------------------------------------------------
    img_cv_escalada = reducir_escala(img_cv_recortada, factor_escala=FACTOR_ESCALA_OCR)
    if img_cv_escalada is not None:
        img_pil_escalada = Image.fromarray(cv2.cvtColor(img_cv_escalada, cv2.COLOR_BGR2RGB))
        lat, lon, metodo = reconocer_y_extraer_robusto(img_pil_escalada, config_ocr='--psm 6')
        if lat and lon: return lat, lon, f"{metodo}_ESCALADO_{int(FACTOR_ESCALA_OCR*100)}_PSM6"


    # --- INTENTO 1: RAW OCR (Sin preprocesamiento) con PSM 6 ---
    img_pil_raw = Image.fromarray(cv2.cvtColor(img_cv_recortada, cv2.COLOR_BGR2RGB))
    lat, lon, metodo = reconocer_y_extraer_robusto(img_pil_raw, config_ocr='--psm 6')
    if lat and lon: return lat, lon, f"{metodo}_RAW_PSM6"

    # --- INTENTO 2: RAW OCR (Sin preprocesamiento) con PSM 11 (Sparse Text) ---
    lat, lon, metodo = reconocer_y_extraer_robusto(img_pil_raw, config_ocr='--psm 11')
    if lat and lon: return lat, lon, f"{metodo}_RAW_PSM11"

    # --- INTENTO 3: PROCESADO OCR (Binarización Otsu) con PSM 6 ---
    img_procesada = preprocesar_otsu(img_cv_recortada)
    if img_procesada is None:
        if nombre_debug:
            Image.fromarray(cv2.cvtColor(img_cv_recortada, cv2.COLOR_BGR2RGB)).save(os.path.join('diagnostico_ocr', f"DEBUG_{nombre_debug}_Recorte_NoProc.png"))
        return None, None, "Error_Preprocesamiento"
        
    img_pil_procesada = Image.fromarray(img_procesada)
    lat, lon, metodo = reconocer_y_extraer_robusto(img_pil_procesada, config_ocr='--psm 6')
    if lat and lon: return lat, lon, f"{metodo}_PROCESADO_OTSU_PSM6"

    # Si falla completamente, guardar imágenes de diagnóstico
    if nombre_debug:
        img_pil_raw.save(os.path.join('diagnostico_ocr', f"DEBUG_{nombre_debug}_RAW_Fallido.png"))
        if img_procesada is not None:
            img_pil_procesada.save(os.path.join('diagnostico_ocr', f"DEBUG_{nombre_debug}_Otsu_Fallido.png"))
        
    return None, None, "Fallo_OCR_Final"


# --------------------------------------------------------------------------
# III. FUNCIÓN PRINCIPAL DE PROCESAMIENTO (FILTRADO POR CSV)
# --------------------------------------------------------------------------

# 🛑 Acepta 'ciudad_seleccionada' como argumento
def procesar_fallas_csv(ciudad_seleccionada):
    """Lee el CSV, identifica fallas, procesa SOLO esas imágenes y actualiza el CSV."""
    
    # 0. CONFIGURACIÓN DE CONTEXTO DE CIUDAD (Usando el valor recibido por sys.argv)
    print("\n\n--- CONTEXTO DE CIUDAD ---")
    print(f"[proceso4.py] Procesando con el contexto de CIUDAD: {ciudad_seleccionada}")
    
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
            campos_esperados = ['OT', 'Resto_Nombre', 'Latitud_Extraida', 'Longitud_Extraida', 'Latitud_Decimal', 'Longitud_Decimal', 'Estatus', 'Metodo_Extraccion']
            
            for i, fila in enumerate(lector):
                fila_completa = {campo: fila.get(campo, '') for campo in campos_esperados}
                filas_originales.append(fila_completa)
                
                # Filtrar SOLO las filas que tienen el estado de FALLO
                if fila_completa.get('Estatus', '').strip().upper() == ESTATUS_FALLO:
                    archivos_a_reprocesar_indices.append(i)
    except FileNotFoundError:
        print(f"❌ Error: El archivo de entrada '{CSV_ENTRADA}' no se encontró. Asegúrate de que exista.")
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
        
        ot_base = ot_raw.zfill(8)
        nombre_base = ot_base + resto # Nombre original para debug

        ruta_imagen_encontrada = None
        
        print(f"\n[{total_procesado}/{len(archivos_a_reprocesar_indices)}] ⏳ Analizando Archivo: {nombre_base}")
        
        # Lógica de búsqueda robusta
        EXTENSIONES_IMAGEN = ('.jpg', '.jpeg', '.png', '.tiff', '.webp') 
        nombre_busqueda_limpio = (ot_base + resto).upper().replace(' ', '').replace('_', '')
        
        for archivo_en_carpeta in os.listdir(CARPETA_IMAGENES):
            if not archivo_en_carpeta.lower().endswith(EXTENSIONES_IMAGEN):
                continue
            nombre_sin_extension = os.path.splitext(archivo_en_carpeta)[0]
            nombre_archivo_limpio = nombre_sin_extension.upper().replace(' ', '').replace('_', '')

            if nombre_archivo_limpio.startswith(nombre_busqueda_limpio):
                ruta_imagen_encontrada = os.path.join(CARPETA_IMAGENES, archivo_en_carpeta)
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

            # Recortar la zona inferior (40%)
            height, width = img_cv_original.shape[:2]
            y_start = int(height * (1 - RECORTE_PORCENTAJE))
            img_cv_recortada = img_cv_original[y_start:height, 0:width]
            
            # EJECUTAR ANÁLISIS MULTINIVEL
            lat_ext, lon_ext, metodo = intento_multinivel_robusto(img_cv_recortada, nombre_debug=nombre_base)

            # Inicializar estado y valores por defecto (se mantienen si hay fallo total)
            estado = ESTATUS_FALLO
            lat_ext_final, lon_ext_final = fila.get('Latitud_Extraida', 'FALLO'), fila.get('Longitud_Extraida', 'FALLO')
            lat_dec_final, lon_dec_final = fila.get('Latitud_Decimal', ''), fila.get('Longitud_Decimal', '')
            
            if lat_ext and lon_ext and "Fallo_OCR_Final" not in metodo:
                
                # APLICAR CORRECCIÓN HEURÍSTICA Y CONVERSIÓN
                lat_ext_corregida, lon_ext_corregida = corregir_latitud_ocr(lat_ext, lon_ext, rangos_especificos)
                lat_dec = convertir_a_decimal(lat_ext_corregida)
                lon_dec = convertir_a_decimal(lon_ext_corregida)
                
                # --- ACTUALIZACIÓN CLAVE: Al detectar algo, actualizamos los valores finales ---
                lat_ext_final = lat_ext_corregida
                lon_ext_final = lon_ext_corregida
                lat_dec_final = lat_dec
                lon_dec_final = lon_dec
                
                # VALIDACIÓN GEOGRÁFICA
                es_valido = False
                # Si es DMS, se acepta si es extraído, ya que la conversión decimal es compleja/ausente
                if 'Patron_DMS' in metodo:
                    es_valido = True
                    # No guardar decimales para DMS
                    lat_dec_final = ''
                    lon_dec_final = ''
                # Si es Decimal, debe pasar ambos filtros: Rango Mundial y Rango de Proximidad
                elif validar_rango_geografico(lat_dec, lon_dec) and validar_rango_proximidad(lat_dec, lon_dec, lat_min, lat_max, lon_min, lon_max):
                    es_valido = True

                # --- RESULTADO DE LA VALIDACIÓN ---
                if es_valido:
                    # --- ÉXITO ---
                    estado = ESTATUS_EXITO
                    print(f" 	✔️ ÉXITO: {metodo} | Coordenadas: {lat_ext_final}, {lon_ext_final} | Estatus: {ESTATUS_EXITO}")
                
                else:
                    # --- FALLO DE VALIDACIÓN (FUERA DE RANGO) ---
                    metodo = f"{metodo}_FUERA_RANGO"
                    estado = ESTATUS_FALLO # Mantiene el estado de fallo
                    
                    # Las coordenadas detectadas (aunque fuera de rango) ya están en las variables _final
                    print(f" 	❌ FALLA: {metodo} | Estatus: {ESTATUS_FALLO} (Fuera de Rango)")
                    print(f" 	🚨 COORDENADAS DETECTADAS (GUARDADAS): Ext: {lat_ext_final}, {lon_ext_final} | Dec: {lat_dec_final}, {lon_dec_final}")
                    
                    # Generar imagen de debug con coordenadas superpuestas
                    img_pil_recortada = Image.fromarray(cv2.cvtColor(img_cv_recortada, cv2.COLOR_BGR2RGB))
                    img_con_coords = dibujar_coordenadas_en_imagen(img_pil_recortada, lat_ext_final, lon_ext_final)
                    ruta_debug_fuera_rango = os.path.join('diagnostico_ocr', f"DEBUG_{nombre_base}_FUERA_RANGO_{os.path.basename(ruta_imagen_encontrada)}")
                    img_con_coords.save(ruta_debug_fuera_rango)
            
            # Actualizar la fila en la lista principal
            fila['Latitud_Extraida'] = lat_ext_final
            fila['Longitud_Extraida'] = lon_ext_final
            # Asegurar que si es None (por DMS), se guarde vacío en el CSV
            fila['Latitud_Decimal'] = lat_dec_final if lat_dec_final is not None else '' 
            fila['Longitud_Decimal'] = lon_dec_final if lon_dec_final is not None else ''
            fila['Estatus'] = estado
            fila['Metodo_Extraccion'] = metodo
            
            if estado == ESTATUS_EXITO:
                exitos += 1

        except Exception as e:
            print(f" 	❌ Error en el procesamiento de '{nombre_base}': {e}")
            fila['Metodo_Extraccion'] = f'ERROR_CRITICO: {e}'
            # Si hay un error crítico, el estado se mantiene en ESTATUS_FALLO

    # 3. GUARDAR TODAS LAS FILAS ACTUALIZADAS EN EL CSV DE SALIDA
    exportar_a_csv(filas_originales, CSV_SALIDA)
    print(f"\n✨ REPORTE FINAL: Se recuperaron {exitos} coordenadas adicionales.")

# --------------------------------------------------------------------------
# IV. CONFIGURACIÓN DE EJECUCIÓN (MODIFICADA PARA RECIBIR ARGUMENTOS)
# --------------------------------------------------------------------------

if __name__ == '__main__':
    # Crear carpeta de diagnóstico si no existe
    if not os.path.exists('diagnostico_ocr'):
        os.makedirs('diagnostico_ocr')
    
    # Verificar librerías (opcional pero bueno para el usuario)
    print("\nVerificando librerías...")
    try:
        # Esto simplemente verifica si se pueden usar sin fallar al inicio
        cv2.useOptimized() 
        pytesseract.get_tesseract_version()
        print("Librerías (cv2, pytesseract, PIL) verificadas. Listo.")
    except Exception:
        print("🚨 ADVERTENCIA: Las librerías no funcionan correctamente.")

    try:
        # sys.argv[1] es la Ciudad (Argumento 1)
        # sys.argv[2] es la Ruta de Tesseract (Argumento 2)
        if len(sys.argv) < 3:
            raise IndexError("Se esperaban los argumentos de CIUDAD y RUTA TESSERACT.")
            
        ciudad_input = sys.argv[1].upper() 
        tesseract_path = sys.argv[2] # Nuevo argumento
        
        # Configurar Tesseract con la ruta dinámica
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"[proceso4.py] Configurando Tesseract con la ruta dinámica: '{tesseract_path}'")
        
        print(f"[proceso4.py] Recibida CIUDAD desde el orquestador: '{ciudad_input}'")
        procesar_fallas_csv(ciudad_input)
        
    except IndexError as e:
        print(f"[proceso4.py] ERROR: {e} Se esperaba el argumento de la CIUDAD y la RUTA TESSERACT desde 'app.py'.")
        sys.exit(1)
    except Exception as e:
        print(f"[proceso4.py] ERROR: Fallo al ejecutar el proceso 4. {e}")
        sys.exit(1)