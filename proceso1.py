import os
import re
import csv
import cv2 
import numpy as np 
from PIL import Image
import pytesseract
from tqdm import tqdm # Importamos la librería tqdm para la barra de progreso
import sys # 🛑 Necesario para leer argumentos de línea de comandos

# =========================================================================
# 🛑 CONFIGURACIÓN DE TESSERACT OCR
# =========================================================================
# 🛑 RUTA HARDCODEADA ELIMINADA: La ruta se recibe dinámicamente de app.py.
# =========================================================================

# =========================================================================
# 📌 CONFIGURACIÓN DE RECORTE INICIAL Y ROIS
# =========================================================================
PORCENTAJE_RECORTE = 0.30 # Ajuste: 30% inferior
FACTOR_ESCALA_OCR = 0.50 # Factor de reducción de escala (50%)

ROI_LISTA = [
    (450, 750, 1024, 1024), 
    (0, 750, 550, 1024), 
    (0, 800, 1024, 1024) 
]

# =========================================================================
# 📌 RANGOS GLOBALES DE MÉXICO (Fallback)
# =========================================================================
LAT_MIN_ESPERADA = 14.0 	
LAT_MAX_ESPERADA = 33.0 	
LON_MIN_ESPERADA = -119.0 	
LON_MAX_ESPERADA = -86.0 	

# =========================================================================
# 📌 RANGOS DE PROXIMIDAD POR CIUDAD (Lat_Min, Lat_Max, Lon_Min, Lon_Max)
# =========================================================================
RANGOS_CIUDADES = {
    'ENSENADA': (31.0, 33.0, -117.5, -115.5), 	 	
    'TIJUANA': (32.0, 34.0, -117.5, -115.5), 	 	 	
    'CHIHUAHUA': (27.5, 29.5, -107.0, -105.0), 	 	 	
    'SALTILLO': (24.5, 26.5, -102.0, -100.0), 	 	 	
    'CIUDAD VICTORIA': (23.5, 25.5, -99.5, -97.5), 
    'MONTERREY': (25.0, 27.0, -101.0, -99.0), 	 	 	
    'NUEVO LAREDO': (27.0, 29.0, -100.5, -98.5), 	
    'TAMPICO': (21.0, 23.0, -99.0, -97.0), 	 	 	
    'GUADALAJARA': (20.0, 22.0, -104.5, -102.5), 	
    'QUERETARO': (20.0, 22.0, -101.5, -99.5), 	 	 	
    'SAN LUIS POTOSI': (21.0, 23.0, -102.0, -100.0), 
    'TOLUCA': (18.5, 20.5, -100.5, -98.5) 	 	 	 	
}

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
    """Convierte una coordenada con Cardinal (ej. 25.82588N) a Decimal puro."""
    if not coord_con_cardinal: return None
    coord_con_cardinal = str(coord_con_cardinal).strip() 
    
    if coord_con_cardinal.startswith('-'):
        try: return float(coord_con_cardinal)
        except ValueError: return None

    cardinal = coord_con_cardinal[-1].upper()
    if cardinal in ('N', 'S', 'E', 'W'):
        valor_str = coord_con_cardinal[:-1]
    else:
        cardinal = None
        valor_str = coord_con_cardinal
    
    try:
        valor = float(valor_str)
    except ValueError:
        return None
        
    if cardinal in ('S', 'W'):
        return -valor
    else:
        return valor

def corregir_latitud_ocr(lat_extraida, lon_extraida, ciudad_rangos):
    """Aplica una corrección heurística para errores comunes de OCR en el primer dígito."""
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
        
        tqdm.write(f" 	⚙️ CORRECCIÓN HEURÍSTICA: '{lat_extraida}' ajustado a '{lat_corregida}' (Error de dígito inicial: {primer_digito_encontrado} -> {digito_esperado}).")
        return lat_corregida, lon_extraida
        
    return lat_extraida, lon_extraida

def validar_rango_geografico(latitud_dec, longitud_dec):
    """Verifica si las coordenadas decimales están dentro de los rangos válidos mundiales."""
    if latitud_dec is None or longitud_dec is None: return False
    lat_valida = -90.0 <= latitud_dec <= 90.0
    lon_valida = -180.0 <= longitud_dec <= 180.0
    return lat_valida and lon_valida

def validar_rango_proximidad(lat_dec, lon_dec, lat_min, lat_max, lon_min, lon_max):
    """Verifica si las coordenadas decimales están dentro del rango de proximidad definido."""
    if lat_dec is None or lon_dec is None: return False
    lat_en_rango = lat_min <= lat_dec <= lat_max
    lon_en_rango = lon_min <= lon_dec <= lon_max
    return lat_en_rango and lon_en_rango

def exportar_a_csv(datos, nombre_archivo='resultados_coordenadas.csv'):
    """Exporta los resultados a un archivo CSV."""
    campos = ['OT', 'Resto_Nombre', 'Latitud_Extraida', 'Longitud_Extraida', 'Latitud_Decimal', 'Longitud_Decimal', 'Estatus']
    try:
        with open(nombre_archivo, 'w', newline='', encoding='utf-8') as archivo_csv:
            escritor = csv.DictWriter(archivo_csv, fieldnames=campos)
            escritor.writeheader()
            escritor.writerows(datos)
        print(f"\n✅ EXPORTACIÓN EXITOSA: Los resultados se guardaron en {nombre_archivo}")
    except Exception as e:
        print(f"\n❌ ERROR DE EXPORTACIÓN: No se pudo escribir el archivo CSV. {e}")

def verificar_e_instalar_librerias():
    """Verifica si las librerías principales de Python están instaladas."""
    # Nota: Esta función es solo una estructura. La verificación real la hace app.py.
    print("✨ Iniciando verificación de librerías...")
    try:
        if not os.path.exists('diagnostico_ocr'):
            os.makedirs('diagnostico_ocr')
            print("Carpeta 'diagnostico_ocr' creada para guardar las imágenes de prueba.")
    except Exception as e:
        print(f"Ocurrió un error durante la verificación: {e}")
    print("--- Fin de la verificación ---\n")

def reducir_escala_cv(img_cv, factor_escala):
    """
    Realiza una reducción de escala (downsampling) en la imagen. 
    Esto mejora la velocidad y puede funcionar como filtro de ruido para el OCR.
    """
    if img_cv is None or img_cv.size == 0:
        return None
    
    nuevo_ancho = int(img_cv.shape[1] * factor_escala)
    nuevo_alto = int(img_cv.shape[0] * factor_escala)
    
    img_redimensionada = cv2.resize(img_cv, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LINEAR)
    
    return img_redimensionada

# --------------------------------------------------------------------------
# II. LÓGICA DE EXTRACCIÓN Y PREPROCESAMIENTO
# --------------------------------------------------------------------------

def reconocer_y_extraer(img_pil, config_ocr):
    """Aplica OCR y la Regex flexible al objeto de imagen PIL."""
    texto_extraido = pytesseract.image_to_string(img_pil, lang='eng', config=config_ocr)
    
    patron_coordenadas = re.compile(
        r'(-?\d{1,2}[.,]\d{1,}[NWSE]?)[\s,-]*(-?\d{1,3}[.,]\d{1,}[NWSE]?)', 
        re.IGNORECASE 
    )
    
    texto_limpio = texto_extraido.replace('\n', ' ').strip()
    match = patron_coordenadas.search(texto_limpio)
    
    if match:
        lat = match.group(1).replace(',', '.')
        lon = match.group(2).replace(',', '.')
        return lat.upper(), lon.upper()
    else:
        return None, None

def intento1_multiple_passes(img_cv_original):
    """
    Intenta la extracción con múltiples técnicas de preprocesamiento,
    aplicando primero la reducción de escala.
    """
    
    # 1. APLICAR REDUCCIÓN DE ESCALA
    img_cv = reducir_escala_cv(img_cv_original, FACTOR_ESCALA_OCR)
    if img_cv is None: return None, None
    
    gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # PASO 1: Procesamiento Estándar (Otsu)
    desenfoque = cv2.GaussianBlur(gris, (5, 5), 0)
    _, img_binaria_otsu = cv2.threshold(desenfoque, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    img_pil_otsu = Image.fromarray(img_binaria_otsu)
    lat, lon = reconocer_y_extraer(img_pil_otsu, config_ocr='--psm 3')
    if lat and lon: return lat, lon

    # PASO 2: Procesamiento de Alto Contraste (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contraste_mejorado = clahe.apply(gris) 
    desenfoque_clahe = cv2.GaussianBlur(contraste_mejorado, (5, 5), 0)
    _, img_binaria_clahe = cv2.threshold(desenfoque_clahe, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    img_pil_clahe = Image.fromarray(img_binaria_clahe)
    lat, lon = reconocer_y_extraer(img_pil_clahe, config_ocr='--psm 3')
    if lat and lon: return lat, lon
    
    # PASO 3: Umbralización Simple
    _, img_binaria_simple = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY_INV)
    img_pil_simple = Image.fromarray(img_binaria_simple)
    lat, lon = reconocer_y_extraer(img_pil_simple, config_ocr='--psm 3')
    if lat and lon: return lat, lon
        
    return None, None


def intento2_fallback_detallado(ruta_imagen, img_cv):
    """
    Prueba múltiples ROIs *sin* reducción de escala, en la imagen recortada.
    """
    
    for i, roi_coords in enumerate(ROI_LISTA):
        x_start, y_start, x_end, y_end = roi_coords
        
        try:
            # Asegura que las coordenadas del ROI sean válidas para la imagen recortada
            height, width = img_cv.shape[:2]
            x_start = min(x_start, width)
            x_end = min(x_end, width)
            y_start = min(y_start, height)
            y_end = min(y_end, height)
            
            # Recorte
            img_recortada = img_cv[y_start:y_end, x_start:x_end] 

            # Preprocesamiento AVANZADO
            gris = cv2.cvtColor(img_recortada, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)) 
            contraste_mejorado = clahe.apply(gris)
            desenfoque = cv2.GaussianBlur(contraste_mejorado, (5, 5), 0)
            img_binaria = cv2.adaptiveThreshold(desenfoque, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY_INV, 11, 2)
            kernel = np.ones((1,1), np.uint8) 
            img_binaria = cv2.dilate(img_binaria, kernel, iterations=1)

            img_pil = Image.fromarray(img_binaria)
            
            lat, lon = reconocer_y_extraer(img_pil, config_ocr='--psm 8')

            if lat and lon:
                return lat, lon
            
            if i == len(ROI_LISTA) - 1:
                try:
                    ruta_salida = os.path.join('diagnostico_ocr', f"FALLO_FINAL_{os.path.basename(ruta_imagen)}")
                    img_pil.save(ruta_salida)
                except Exception as save_err:
                    pass

        except Exception as e:
            pass
            
    return None, None

# --------------------------------------------------------------------------
# III. FUNCIÓN PRINCIPAL DE EJECUCIÓN (CON RECORTE DINÁMICO)
# --------------------------------------------------------------------------

# 🛑 Acepta 'ciudad_seleccionada' como argumento
def procesar_carpeta(carpeta_path, recorte_porcentaje, ciudad_seleccionada):
    """Recorre la carpeta, aplica el recorte dinámico, la lógica de dos intentos y exporta a CSV."""
    
    verificar_e_instalar_librerias()
    
    # 🛑 PASO 1: SELECCIÓN DE CONTEXTO DE CIUDAD (Usando el valor recibido por sys.argv)
    ciudades_validas = list(RANGOS_CIUDADES.keys())
    print("\n\n--- CONTEXTO DE CIUDAD ---")
    print(f"[proceso1.py] Procesando con el contexto de CIUDAD: {ciudad_seleccionada}")
    
    rangos_especificos = obtener_rangos_por_ciudad(ciudad_seleccionada)
    
    if rangos_especificos:
        lat_min, lat_max, lon_min, lon_max = rangos_especificos
        print(f"✅ Contexto cargado: {ciudad_seleccionada}. Rango de Proximidad: Lat {lat_min}° a {lat_max}°")
    else:
        print(f"⚠️ Ciudad '{ciudad_seleccionada}' no reconocida. Usando rango general de México ({LAT_MIN_ESPERADA}° a {LAT_MAX_ESPERADA}°).")
        lat_min, lat_max, lon_min, lon_max = LAT_MIN_ESPERADA, LAT_MAX_ESPERADA, LON_MIN_ESPERADA, LON_MAX_ESPERADA

    print(f"\n📂 Procesando imágenes en la carpeta: {carpeta_path}")
    print(f"✂️ Aplicando Recorte Inicial del {int(recorte_porcentaje*100)}% Inferior.")
    print(f"⚙️ Primer Intento: Recorte del 30% + Reducción de Escala al {int(FACTOR_ESCALA_OCR*100)}%")

    if not os.path.isdir(carpeta_path):
        print(f"❌ Error: La carpeta '{carpeta_path}' no existe.")
        return

    datos_para_csv = []
    
    # --- PREPARAR LISTA DE ARCHIVOS ---
    archivos_a_procesar = [
        nombre_archivo_con_ext.strip()
        for nombre_archivo_con_ext in os.listdir(carpeta_path)
        if nombre_archivo_con_ext.strip().lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))
    ]

    # --- INICIO DE CONTADORES ---
    correctas_contadas = 0
    no_encontradas_contadas = 0
    
    # 🛑 NUEVO: Definir rangos de magnitud de Longitud de México para la corrección W.
    LON_MIN_MEX_MAG = min(abs(LON_MIN_ESPERADA), abs(LON_MAX_ESPERADA)) # 86.0
    LON_MAX_MEX_MAG = max(abs(LON_MIN_ESPERADA), abs(LON_MAX_ESPERADA)) # 119.0
    
    # --- INICIO DEL BUCLE CON BARRA DE PROGRESO (tqdm) ---
    total_archivos = len(archivos_a_procesar)
    
    for nombre_archivo_con_ext in tqdm(archivos_a_procesar, desc="Análisis OCR", unit="img"):
        
        ruta_completa = os.path.join(carpeta_path, nombre_archivo_con_ext)
        nombre_archivo_base = os.path.splitext(nombre_archivo_con_ext)[0]
        
        ot, resto_nombre = separar_por_posicion(nombre_archivo_base.strip())
        
        try:
            img_cv_original = cv2.imread(ruta_completa)
            if img_cv_original is None:
                # 🛑 ERROR DE LECTURA CRÍTICO
                tqdm.write(f"❌ Error de lectura en {nombre_archivo_con_ext}: No se pudo cargar la imagen.")
                raise FileNotFoundError("No se pudo cargar la imagen.")
        except Exception as e:
            # Ya se escribió el error, solo se añade el registro de fallo
            no_encontradas_contadas += 1
            # 🛑 REGISTRO DE FALLO EN CONSOLA (Error de lectura)
            tqdm.write(f"❌ Fallo OCR: OT:{ot} Elem:{resto_nombre}: Error al cargar la imagen.")

            datos_para_csv.append({
                'OT': ot, 'Resto_Nombre': resto_nombre, 'Latitud_Extraida': 'FALLO', 
                'Longitud_Extraida': 'FALLO', 'Latitud_Decimal': '', 
                'Longitud_Decimal': '', 'Estatus': "NO ENCONTRADO"
            })
            continue

        # RECORTE DINÁMICO (30% inferior)
        alto_total = img_cv_original.shape[0]
        y_inicio_recorte = int(alto_total * (1 - recorte_porcentaje)) 
        img_cv_recortada = img_cv_original[y_inicio_recorte:alto_total, 0:img_cv_original.shape[1]]
        
        if img_cv_recortada.size == 0:
              tqdm.write(f"⚠️ {ot} La imagen recortada está vacía. Saltando.")
              no_encontradas_contadas += 1
              datos_para_csv.append({
                'OT': ot, 'Resto_Nombre': resto_nombre, 'Latitud_Extraida': 'FALLO', 
                'Longitud_Extraida': 'FALLO', 'Latitud_Decimal': '', 
                'Longitud_Decimal': '', 'Estatus': "NO ENCONTRADO"
            })
              continue

        # INTENTO 1: Multi-Pass con Reducción de Escala
        lat_ext, lon_ext = intento1_multiple_passes(img_cv_recortada)
        
        # INTENTO 2: Fallback con ROIs predefinidos
        if lat_ext is None:
            lat_ext, lon_ext = intento2_fallback_detallado(ruta_completa, img_cv_recortada)
        
        # -------------------------------------------------------------
        # PREPARACIÓN DE DATOS PARA CSV Y VALIDACIÓN CONTEXTUAL
        # -------------------------------------------------------------
        
        estado = "NO ENCONTRADO"
        lat_dec_guardar = ''
        lon_dec_guardar = ''

        if lat_ext and lon_ext:
            
            # APLICAR CORRECCIÓN HEURÍSTICA (dígito inicial)
            lat_ext, lon_ext = corregir_latitud_ocr(lat_ext, lon_ext, rangos_especificos)
            
            # 🛑 INICIO: LÓGICA DE AJUSTE DE LONGITUD (W) 🛑
            # Usamos el rango amplio de México para forzar W si falta el signo
            lon_temp = lon_ext.strip().upper()
            lon_str_check = re.sub(r'[^0-9.]', '', lon_temp.replace(',', '.')).strip()
            
            if lon_str_check and not re.search(r'[NWSE-]', lon_temp):
                try:
                    valor_lon = float(lon_str_check)
                    
                    # USAR RANGO GLOBAL DE MÉXICO PARA EL CONTEXTO W
                    if LON_MIN_MEX_MAG <= valor_lon <= LON_MAX_MEX_MAG: 
                        lon_ext += 'W' # Agregamos W para forzar la conversión a negativo
                        tqdm.write(f" 	⚙️ AJUSTE DE CONTEXTO GLOBAL: Longitud '{lon_temp}' ajustada a '{lon_ext}' (W forzada, magnitud en rango México).")
                except ValueError:
                    pass
            # 🛑 FIN: LÓGICA DE AJUSTE 🛑

            lat_dec = convertir_a_decimal(lat_ext)
            lon_dec = convertir_a_decimal(lon_ext)
            
            # VALIDACIÓN 1: Rango Geográfico Estándar & Rango de Proximidad
            es_valido = validar_rango_geografico(lat_dec, lon_dec)
            es_proximo = validar_rango_proximidad(lat_dec, lon_dec, lat_min, lat_max, lon_min, lon_max)
            

            if es_valido and es_proximo:
                
                # CORRECTO: Pasó todas las validaciones (MUNDIAL + PROXIMIDAD)
                estado = "CORRECTO" 
                correctas_contadas += 1
                lat_dec_guardar = lat_dec
                lon_dec_guardar = lon_dec
                
            else:
                # Fallo en Validación (No es mundialmente válido o NO es próximo a la ciudad)
                no_encontradas_contadas += 1
                # 🛑 REGISTRO DE FALLO EN CONSOLA (Fallo de Validación)
                tqdm.write(f"❌ Fallo Valid: OT:{ot} Elem:{resto_nombre}: {lat_ext}, {lon_ext} (Fuera de Rango/Proximidad)")
                # Si falla, no guardamos los decimales para que sean reprocesados
                
            datos_para_csv.append({
                'OT': ot,
                'Resto_Nombre': resto_nombre,
                'Latitud_Extraida': lat_ext,
                'Longitud_Extraida': lon_ext,
                'Latitud_Decimal': lat_dec_guardar,
                'Longitud_Decimal': lon_dec_guardar,
                'Estatus': estado 
            })
        else:
            # Estatus si NO SE DETECTÓ nada
            no_encontradas_contadas += 1
            # 🛑 REGISTRO DE FALLO EN CONSOLA (Fallo de Detección OCR)
            tqdm.write(f"❌ Fallo OCR: OT:{ot} Elem:{resto_nombre}: No se detectaron coordenadas.")

            datos_para_csv.append({
                'OT': ot,
                'Resto_Nombre': resto_nombre,
                'Latitud_Extraida': 'FALLO',
                'Longitud_Extraida': 'FALLO',
                'Latitud_Decimal': '',
                'Longitud_Decimal': '',
                'Estatus': "NO ENCONTRADO" 
            })

    # EXPORTACIÓN FINAL
    exportar_a_csv(datos_para_csv)

    # --- RESUMEN FINAL DE ESTATUS ---
    print("\n" + "="*50)
    print("📋 RESUMEN FINAL DEL ANÁLISIS")
    print("-" * 50)
    print(f"📊 Total de Archivos Analizados: {total_archivos}")
    print(f"✅ Coordenadas CORRECTAS (Válidas): {correctas_contadas}")
    print(f"❌ Registros NO ENCONTRADOS (Fallo OCR/Validación): {no_encontradas_contadas}")
    print("="*50 + "\n")


# --------------------------------------------------------------------------
# IV. CONFIGURACIÓN DE EJECUCIÓN (MODIFICADA PARA RECIBIR ARGUMENTOS)
# --------------------------------------------------------------------------

CARPETA_IMAGENES = 'fotos' 

if __name__ == '__main__':
    # 🛑 Cambiamos la llamada a procesar_carpeta para usar sys.argv
    try:
        # sys.argv[1] es la Ciudad (Argumento 1)
        # sys.argv[2] es la Ruta de Tesseract (Argumento 2)
        if len(sys.argv) < 3:
            raise IndexError("Se esperaban los argumentos de CIUDAD y RUTA TESSERACT.")
            
        ciudad_input = sys.argv[1].upper() 
        tesseract_path = sys.argv[2] # Nuevo argumento
        
        # Configurar Tesseract con la ruta dinámica
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"[proceso1.py] Configurando Tesseract con la ruta dinámica: '{tesseract_path}'")
        
        print(f"[proceso1.py] Recibida CIUDAD desde el orquestador: '{ciudad_input}'")
        procesar_carpeta(CARPETA_IMAGENES, PORCENTAJE_RECORTE, ciudad_input)
    except IndexError as e:
        print(f"ERROR: {e} Se esperaba el argumento de la CIUDAD y la RUTA TESSERACT desde 'app.py'.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Fallo al ejecutar el proceso 1. {e}")
        sys.exit(1)