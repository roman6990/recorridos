import os
import re
import csv
import cv2 
import numpy as np 
from PIL import Image
import pytesseract
from collections import Counter
from tqdm import tqdm 
import sys # 🛑 Necesario para leer argumentos de línea de comandos

# =========================================================================
# 🛑 CONFIGURACIÓN DE TESSERACT OCR & RUTAS
# =========================================================================
# 🛑 RUTA HARDCODEADA ELIMINADA: La ruta se configura dinámicamente al inicio
# pytesseract.pytesseract.tesseract_cmd se configura en el bloque main()

CARPETA_IMAGENES = 'fotos'
CSV_ENTRADA = 'resultados_coordenadas.csv' 
# 🛑 Ajustado para sobrescribir el archivo principal de resultados, siguiendo el flujo iterativo
CSV_SALIDA = 'resultados_coordenadas.csv' 
ESTATUS_FALLO = 'NO ENCONTRADO' 

# --------------------------------------------------------------------------
# I. FUNCIONES AUXILIARES Y LÓGICA DE ANÁLISIS
# --------------------------------------------------------------------------

def extraer_solo_numeros_flexibles(texto):
    """
    Último recurso: Busca cualquier par de números largos que se parezcan a coordenadas.
    Asume N y W para asignar cardinales.
    """
    # 1. Limpiar el texto de caracteres no numéricos o de punto/coma, dejando solo números
    texto_limpio = re.sub(r'[^\d.,]', ' ', texto)
    
    # 2. Buscar números decimales largos (al menos 2 dígitos decimales)
    patron_decimal = re.compile(r'(\d{1,3}[.,]\d{2,})', re.IGNORECASE)
    numeros_encontrados = patron_decimal.findall(texto_limpio)
    
    if len(numeros_encontrados) >= 2:
        lat_str = numeros_encontrados[0].replace(',', '.')
        lon_str = numeros_encontrados[1].replace(',', '.')
        
        # Asumir cardinales N y W si no tienen, basado en el contexto geográfico
        if not lat_str[-1].isalpha():
            lat_str += 'N'
        if not lon_str[-1].isalpha():
            lon_str += 'W'

        # Verificar que el resultado sea válido (ej., no queremos 1.234N)
        # Reutilizamos convertir_a_decimal para la validación interna
        lat_dec = convertir_a_decimal(lat_str)
        lon_dec = convertir_a_decimal(lon_str)

        # Usamos una heurística simple para filtrar coordenadas muy pequeñas o no representativas de Long/Lat
        if lat_dec is not None and abs(lat_dec) > 10 and abs(lon_dec) > 70: 
            return lat_str, lon_str, "Patron_5_Numerico_Flexible"
            
    return None, None, None


def separar_por_posicion(nombre_archivo_base):
    """Separa el nombre del archivo en OT (primeros 8 caracteres) y el resto."""
    ot = nombre_archivo_base[:8]
    resto_nombre = nombre_archivo_base[8:]
    return ot, resto_nombre

def convertir_a_decimal(coord_con_cardinal):
    """Convierte una coordenada en formato Decimal con Cardinal a Decimal puro."""
    if not coord_con_cardinal or coord_con_cardinal == 'FALLO':
        return None
    coord_con_cardinal = str(coord_con_cardinal).strip()
    cardinal = coord_con_cardinal[-1].upper() if coord_con_cardinal and coord_con_cardinal[-1].isalpha() else None
    valor_str = coord_con_cardinal[:-1] if cardinal else coord_con_cardinal
    valor_str = valor_str.replace(',', '.').strip()
    try:
        valor = float(valor_str)
    except ValueError:
        return None
    if cardinal in ('S', 'W'):
        return -valor
    return valor

def exportar_a_csv(datos, nombre_archivo):
    """Exporta los resultados a un archivo CSV."""
    
    campos = ['OT', 'Resto_Nombre', 'Latitud_Extraida', 'Longitud_Extraida', 
              'Latitud_Decimal', 'Longitud_Decimal', 'Estatus', 'Metodo_Extraccion']
    
    try:
        with open(nombre_archivo, 'w', newline='', encoding='utf-8') as archivo_csv:
            escritor = csv.DictWriter(archivo_csv, fieldnames=campos, restval='') 
            escritor.writeheader()
            escritor.writerows(datos)
        
        print(f"\n✅ EXPORTACIÓN EXITOSA: Los resultados se guardaron en {nombre_archivo}")
    except Exception as e:
        print(f"\n❌ ERROR DE EXPORTACIÓN: No se pudo escribir el archivo CSV. {e}")
        
def reconocer_y_extraer_mejorado(img_pil, config_ocr='--psm 6'):
    """Aplica OCR y usa patrones específicos de coordenadas."""
    try:
        texto_extraido = pytesseract.image_to_string(img_pil, lang='eng', config=config_ocr)
        
        # 1. Intento de patrones específicos
        patron_coordenadas = re.compile(r'(\d{1,3}[.,]\d{1,}[NS])\s*[\s,-]?\s*(\d{1,3}[.,]\d{1,}[WE])', re.IGNORECASE)
        match = patron_coordenadas.search(texto_extraido)
        
        if match:
              lat, lon = match.groups()
              return lat, lon, "Patron_Exacto"
        
        # 2. Intento de patrones flexibles
        return extraer_solo_numeros_flexibles(texto_extraido)

    except Exception:
        return None, None, None

def preprocesar_imagen_optimizada(img_cv, metodo='inferior_optimizado'):
    """Aplica técnicas de preprocesamiento avanzadas."""
    gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    if metodo == 'inferior_optimizado':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gris = clahe.apply(gris)
        binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 15, 5)
    else: # Alto Contraste
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        gris = cv2.filter2D(gris, -1, kernel)
        _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    return binaria

def estrategia_30pct_inferior_intensiva(ruta_imagen, img_cv_original):
    """Estrategia intensiva que prueba múltiples preprocesamientos en el 30% inferior."""
    
    # 1. Recorte al 30% inferior
    height, width = img_cv_original.shape[:2]
    y_start = int(height * 0.70)
    img_30pct = img_cv_original[y_start:height, 0:width]
    
    if img_30pct.size == 0:
        return None, None, "Fallo_Recorte_Vacio"
    
    preprocesamientos = ['inferior_optimizado', 'alto_contraste']
    config_ocr_options = ['--psm 3', '--psm 6'] # Usar PSM 3 y 6
    
    for preproc in preprocesamientos:
        for config in config_ocr_options:
            try:
                img_procesada = preprocesar_imagen_optimizada(img_30pct, preproc)
                img_pil = Image.fromarray(img_procesada)
                lat, lon, metodo = reconocer_y_extraer_mejorado(img_pil, config)
                
                if lat and lon:
                    return lat, lon, f"Intensivo_{preproc}_{config.replace('--psm ', 'PSM')}_{metodo}"
            except Exception:
                continue
    
    # Fallo total: Guardar el diagnóstico
    try:
        img_diagnostico = preprocesar_imagen_optimizada(img_30pct, 'inferior_optimizado')
        ruta_salida = os.path.join('diagnostico_ocr', f"FALLO_FINAL_{os.path.basename(ruta_imagen)}")
        Image.fromarray(img_diagnostico).save(ruta_salida)
    except Exception:
        pass
    
    return None, None, "Fallo_Intensivo_Final"

# --------------------------------------------------------------------------
# II. FUNCIÓN PRINCIPAL DE PROCESAMIENTO DE FALLAS
# --------------------------------------------------------------------------

def procesar_fallas_csv():
    """Lee el CSV, identifica fallas y re-procesa las imágenes correspondientes."""
    
    print("--- Iniciando Reprocesamiento de Fallas (Intensivo y Flexible) ---")
    
    filas_originales = []
    archivos_a_reprocesar = []
    
    # 1. LEER CSV
    try:
        # Aseguramos que se lea el archivo de resultados actual
        with open(CSV_ENTRADA, 'r', encoding='utf-8') as archivo_csv:
            # Intentamos leer con DictReader, que es más seguro para archivos existentes
            try:
                lector = csv.DictReader(archivo_csv)
            except Exception as e:
                tqdm.write(f"Advertencia: No se pudo iniciar DictReader. {e}")
                return # Detiene si no se puede leer
                
            for fila in lector:
                # Aseguramos que el Metodo_Extraccion exista para evitar KeyError en P3/P4
                if 'Metodo_Extraccion' not in fila:
                    fila['Metodo_Extraccion'] = '' 
                    
                filas_originales.append(fila)
                # Verifica si la fila tiene el estatus de fallo
                if fila.get('Estatus') == ESTATUS_FALLO:
                    archivos_a_reprocesar.append(fila)
    except FileNotFoundError:
        print(f"❌ Error: El archivo de entrada '{CSV_ENTRADA}' no se encontró. No hay fallas que reprocesar.")
        return
    except Exception as e:
        print(f"❌ Error crítico leyendo CSV: {e}")
        return

    print(f"📊 Total registros: {len(filas_originales)} | Fallas a revisar: {len(archivos_a_reprocesar)}")
    
    exitos = 0
    total_fallas = len(archivos_a_reprocesar)

    # 2. PROCESAR FALLAS CON BARRA DE PROGRESO
    # ----------------------------------------------------------------------
    for fila in tqdm(archivos_a_reprocesar, desc="Reprocesando Fallas", unit="arch"):
    # ----------------------------------------------------------------------
        
        ot_raw = fila.get('OT', '').strip()
        resto = fila.get('Resto_Nombre', '').strip()
        
        # Generar candidatos de nombre de archivo
        ot_pad = ot_raw.zfill(8) 
        posibles_nombres_base = [ot_pad + resto, "00" + ot_raw + resto]

        ruta_imagen_encontrada = None
        nombre_archivo_debug = ot_pad + resto # Nombre base para mensajes y debug
        
        # Buscar la imagen en el disco
        for nombre_base in posibles_nombres_base:
            for ext in ('.jpg', '.jpeg', '.png', '.tiff'):
                ruta_candidata = os.path.join(CARPETA_IMAGENES, nombre_base + ext)
                if os.path.exists(ruta_candidata):
                    ruta_imagen_encontrada = ruta_candidata
                    break
            if ruta_imagen_encontrada:
                break
        
        if not ruta_imagen_encontrada:
            continue 
            
        try:
            img_cv_original = cv2.imread(ruta_imagen_encontrada)
            if img_cv_original is None: continue

            # EJECUTAR ANÁLISIS INTENSIVO
            lat_ext, lon_ext, metodo = estrategia_30pct_inferior_intensiva(ruta_imagen_encontrada, img_cv_original)

            if lat_ext and lon_ext:
                lat_dec = convertir_a_decimal(lat_ext)
                lon_dec = convertir_a_decimal(lon_ext)
                
                # Buscamos la fila original en la lista completa para actualizarla
                # Esto es crucial ya que el bucle itera sobre una sublista de fallas
                for original_row in filas_originales:
                    if original_row['OT'] == ot_raw and original_row['Resto_Nombre'] == resto:
                        original_row['OT'] = ot_pad 
                        original_row['Latitud_Extraida'] = lat_ext
                        original_row['Longitud_Extraida'] = lon_ext
                        original_row['Latitud_Decimal'] = lat_dec if lat_dec is not None else ''
                        original_row['Longitud_Decimal'] = lon_dec if lon_dec is not None else ''
                        original_row['Estatus'] = 'CORRECTO'
                        original_row['Metodo_Extraccion'] = metodo
                        
                        exitos += 1
                        tqdm.write(f"✔️ ¡ÉXITO! Coordenadas encontradas para {nombre_archivo_debug} (Método: {metodo})")
                        break
            else:
                # Si falló el reprocesamiento intensivo, actualizamos el método de extracción
                for original_row in filas_originales:
                    if original_row['OT'] == ot_raw and original_row['Resto_Nombre'] == resto and original_row['Estatus'] == ESTATUS_FALLO:
                        original_row['Metodo_Extraccion'] = metodo
                        break


        except Exception as e:
            # Si hay un error crítico durante el procesamiento de esta fila
            for original_row in filas_originales:
                if original_row['OT'] == ot_raw and original_row['Resto_Nombre'] == resto:
                    original_row['Metodo_Extraccion'] = f'ERROR_CRITICO: {e}'
                    tqdm.write(f"❌ Error en el procesamiento de {nombre_archivo_debug}: {e}")
                    break

    # 3. GUARDAR
    # Exportamos la lista completa de filas_originales (ya actualizadas)
    exportar_a_csv(filas_originales, CSV_SALIDA)

    # Contar los registros que todavía están en fallo después del reprocesamiento
    fallas_restantes = total_fallas - exitos
    
    print("\n" + "="*50)
    print("✨ REPORTE FINAL DEL REPROCESAMIENTO")
    print("-" * 50)
    print(f"📊 Total de Fallas Revisadas: {total_fallas}")
    print(f"✅ Coordenadas Recuperadas: {exitos}")
    print(f"❌ Fallos Persistentes: {fallas_restantes}")
    print("="*50 + "\n")


# --------------------------------------------------------------------------
# III. CONFIGURACIÓN DE EJECUCIÓN (MODIFICADA PARA RECIBIR ARGUMENTOS)
# --------------------------------------------------------------------------

def main():
    """Función principal para manejar argumentos y la ejecución."""
    # Crear carpeta de diagnóstico si no existe
    if not os.path.exists('diagnostico_ocr'):
        os.makedirs('diagnostico_ocr')

    try:
        # sys.argv[1] es el primer argumento. Si app.py pasa solo Tesseract, es Arg 1.
        if len(sys.argv) < 2:
            raise IndexError("El orquestador no ha pasado el argumento de la RUTA TESSERACT.")
        
        # Asumimos que el primer argumento es siempre la ruta de Tesseract,
        # ya que app.py fue actualizado para pasarla como Arg 1 a P2.
        tesseract_path = sys.argv[1] 
        
        # Configurar Tesseract con la ruta dinámica
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"[proceso2.py] Configurando Tesseract con la ruta dinámica: '{tesseract_path}'")
        
        procesar_fallas_csv()
    except IndexError as e:
        print(f"[proceso2.py] ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[proceso2.py] ERROR crítico en la ejecución principal: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()