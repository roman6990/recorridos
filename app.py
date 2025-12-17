import subprocess
import sys
import os
import shutil
import stat # Necesario para cambiar permisos de archivos bloqueados
import importlib.util # Para verificar librerías

# --- Configuración ---
# Lista de los scripts a ejecutar en orden.
# Formato: (nombre_del_script, requiere_ciudad, requiere_tesseract)
PROCESOS = [
    ("proceso1.py", True, True),  # Necesita ciudad y Tesseract path
    ("proceso2.py", False, True), # Necesita Tesseract path, NO ciudad
    ("proceso3.py", True, True),  # Necesita ciudad y Tesseract path
    ("proceso4.py", True, True),  # Necesita ciudad y Tesseract path
    ("proceso5.py", False, False), # No necesita nada
]

# Archivos y carpeta a eliminar al final
CARPETA_LIMPIEZA = "diagnostico_ocr"
# 🛑 ACTUALIZADO: resultados_coordenadas.csv se eliminará al final.
ARCHIVOS_LIMPIEZA = [
    "resultados_coordenadas.csv"
] 

# 🛑 RUTA HARDCODEADA DE TESSERACT (Usada como fallback)
# Esta ruta debe coincidir con la configuración de tus scripts de proceso.
TESSERACT_EXE_PATH = r'C:\Users\Roman Acolt\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


# =========================================================================
# I. FUNCIONES DE VALIDACIÓN E INSTALACIÓN DE LIBRERÍAS
# =========================================================================

def verificar_e_instalar_librerias_globales():
    """
    Verifica si las librerías externas requeridas por todos los procesos están instaladas.
    Si faltan, intenta instalarlas usando pip.
    """
    print("\n--- FASE DE VERIFICACIÓN E INSTALACIÓN DE LIBRERÍAS ---")
    # Mapeo de nombre de importación a nombre de paquete de pip
    REQUIRED_PACKAGES = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'cv2': 'opencv-python', # Importa cv2, instala opencv-python
        'PIL': 'Pillow',
        'pytesseract': 'pytesseract',
        'tqdm': 'tqdm'
    }
    
    paquetes_faltantes = []
    
    # 1. Verificar si las librerías están instaladas
    for import_name, package_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(import_name) is None:
            paquetes_faltantes.append(package_name)
    
    if not paquetes_faltantes:
        print("✅ Todas las librerías requeridas están instaladas. Continuar.")
        return True

    # 2. Instalar los paquetes faltantes
    print(f"⚠️ Librerías Faltantes: {', '.join(paquetes_faltantes)}. Intentando instalar...")

    for package in paquetes_faltantes:
        try:
            print(f"Instalando {package}...")
            # Usar sys.executable y el módulo -m pip para asegurar que la instalación 
            # se haga en el entorno correcto
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} instalado correctamente.")
        except subprocess.CalledProcessError:
            print(f"❌ ERROR CRÍTICO: No se pudo instalar {package}. Por favor, verifique su conexión a internet y el estado de pip.")
            return False
        except Exception as e:
            print(f"❌ ERROR DESCONOCIDO al instalar {package}: {e}")
            return False

    # 3. Re-verificación rápida después de la instalación
    for import_name in REQUIRED_PACKAGES.keys():
        if importlib.util.find_spec(import_name) is None:
             print(f"❌ Fallo de verificación: La librería '{import_name}' aún no se puede importar después de la instalación.")
             return False
             
    print("✅ Todas las librerías han sido instaladas y verificadas.")
    return True


# =========================================================================
# II. FUNCIONES DE EJECUCIÓN DEL ORQUESTADOR
# =========================================================================

def handle_remove_readonly(func, path, exc_info):
    """
    Manejador de errores para shutil.rmtree. Intenta cambiar permisos
    y reintentar la eliminación. Es crucial para eliminar archivos de
    sólo lectura o bloqueados por cv2/PIL en Windows.
    """
    # Si el error es un Error de Permiso (PermissionError o Access denied)
    if not os.access(path, os.W_OK):
        print(f"DEBUG: Intentando cambiar permisos en el archivo bloqueado: {path}")
        # Intenta cambiar los permisos para dar permiso de escritura al usuario
        os.chmod(path, stat.S_IWUSR)
        try:
            # Reintenta la función original (shutil.rmtree)
            func(path)
        except Exception as e:
            # Si aún falla, imprime el error
            print(f"ERROR: Fallo al eliminar {path} incluso después de cambiar permisos: {e}")
    else:
        # Para cualquier otro error, lanza la excepción original
        raise

def detectar_tesseract_path(hardcoded_path):
    """
    Intenta detectar Tesseract primero en el PATH del sistema, 
    luego en la ruta hardcodeada. Devuelve la ruta completa, o None si falla.
    """
    print("--- Intentando autodetectar Tesseract OCR ---")
    
    # 1. Intento a través del PATH del sistema (y obtener la ruta completa)
    try:
        # 1a. Intenta obtener la ruta completa usando shutil.which
        path_from_shutil = shutil.which('tesseract')
        if path_from_shutil:
            # 1b. Verifica que el ejecutable funcione
            subprocess.run([path_from_shutil, '-v'], check=True, text=True, capture_output=True, timeout=5)
            # 🛑 MENSAJE SOLICITADO POR EL USUARIO
            print("✅ Tesseract encontrado en el PATH del sistema y verificado.")
            return path_from_shutil # Devuelve la ruta completa
        else:
            print("⚠️ Tesseract NO encontrado en el PATH del sistema.")
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ Tesseract NO encontrado o no funcional en el PATH del sistema.")
        
    # 2. Intento en la ruta hardcodeada (FALLBACK)
    if os.path.exists(hardcoded_path):
        try:
            # Intenta ejecutar Tesseract en la ruta especificada para verificar que funciona
            subprocess.run([hardcoded_path, '-v'], check=True, text=True, capture_output=True, timeout=5)
            # 🛑 MENSAJE SOLICITADO POR EL USUARIO
            print(f"✅ Tesseract encontrado y verificado en la ruta hardcodeada.")
            return hardcoded_path
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: La ruta hardcodeada fue encontrada, pero la ejecución falló. {e}")
            return None
        except Exception as e:
            print(f"❌ ERROR: Falló la ejecución en la ruta hardcodeada. Excepción: {e}")
            return None

    print(f"❌ ERROR: El ejecutable de Tesseract NO fue encontrado ni en el PATH ni en la ruta hardcodeada.")
    return None


def ejecutar_script(script_name, ciudad=None, tesseract_path=None):
    """Ejecuta un script Python como un subproceso."""
    print(f"\n============================================================")
    print(f"[{script_name}] >> INICIANDO EJECUCIÓN...")
    print(f"============================================================")

    # El comando comienza con el ejecutable de Python
    comando = [sys.executable, script_name]

    # Lógica de pase de argumentos:
    if ciudad and tesseract_path:
        # Caso P1, P3, P4: Pasar Ciudad (Arg 1) y Tesseract (Arg 2)
        comando.append(ciudad)
        comando.append(tesseract_path)
        print(f"[{script_name}] Argumento 1 (Ciudad) Enviado: '{ciudad}'")
        # 🛑 RUTA OCULTA
        print(f"[{script_name}] Argumento 2 (Ruta Tesseract) Enviado: [Ruta Oculta]")
    elif tesseract_path and not ciudad:
        # Caso P2: Pasar solo Tesseract (Arg 1)
        comando.append(tesseract_path)
        # 🛑 RUTA OCULTA
        print(f"[{script_name}] Enviando UN SOLO ARGUMENTO (Ruta Tesseract): [Ruta Oculta]")
    # Si ninguno es requerido (P5), comando es solo [python, script_name]

    try:
        # Ejecutar el subproceso. La salida se imprime en tiempo real
        resultado = subprocess.run(
            comando,
            check=True,  
            text=True,
            encoding='utf-8',
            capture_output=False, # Muestra la salida en la consola de app.py
            cwd=os.path.dirname(os.path.abspath(__file__)) # Ejecuta desde el directorio actual
        )
        print(f"============================================================")
        print(f"[{script_name}] >> EJECUCIÓN COMPLETADA con código {resultado.returncode}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: [{script_name}] FALLÓ. Deteniendo el proceso.")
        # Muestra el error estándar si está disponible
        if e.stderr:
             print(f"Salida de Error del Subproceso:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"ERROR: No se encontró el script '{script_name}'. Asegúrate de que el archivo exista.")
        return False

def limpiar_archivos():
    """Elimina la carpeta y archivos especificados."""
    print(f"\n============================================================")
    print(f"--- FASE DE LIMPIEZA ---")
    print(f"============================================================")

    # 1. Eliminar la carpeta (usando el manejador de errores robusto)
    # 🛑 Cambiado para solo ADVERTIR que la carpeta de debug se queda
    if os.path.exists(CARPETA_LIMPIEZA):
        print(f"⚠️ La carpeta de diagnóstico '{CARPETA_LIMPIEZA}' se conservará para revisión. Debe ser eliminada manualmente.")
    else:
        print(f"Carpeta de diagnóstico no encontrada: '{CARPETA_LIMPIEZA}'.")

    # 2. Eliminar archivos
    if not ARCHIVOS_LIMPIEZA:
        print("✅ NO SE ELIMINÓ NINGÚN ARCHIVO CSV/TXT. Todos los resultados se conservan.")
        return
        
    for archivo in ARCHIVOS_LIMPIEZA:
        if os.path.exists(archivo):
            try:
                os.remove(archivo)
                print(f"Archivo eliminado exitosamente: '{archivo}'")
            except OSError as e:
                print(f"Error al eliminar el archivo '{archivo}' (asegúrate de que no esté abierto): {e}")
        else:
            print(f"Archivo no encontrado: '{archivo}'. Saltando eliminación.")

def main():
    """Función principal para solicitar input y orquestar la ejecución."""
    print("--- INICIO DEL ORQUESTADOR DE PROCESOS ---")
    
    # 🛑 1. VALIDACIÓN E INSTALACIÓN DE LIBRERÍAS
    if not verificar_e_instalar_librerias_globales():
        print("\n*** EJECUCIÓN DETENIDA DEBIDO AL FALLO EN LA INSTALACIÓN DE LIBRERÍAS. ***")
        return
    
    # 🛑 2. VALIDACIÓN Y DETECCIÓN DE TESSERACT
    tesseract_path_result = detectar_tesseract_path(TESSERACT_EXE_PATH)
    
    if not tesseract_path_result:
        print("\n*** EJECUCIÓN DETENIDA DEBIDO AL FALLO EN LA VERIFICACIÓN DE TESSERACT. ***")
        return

    # Solicitar el valor de la ciudad una sola vez
    ciudad_input = input("Por favor, ingresa el nombre de la CIUDAD para los procesos: ").strip()

    if not ciudad_input:
        print("La ciudad no puede estar vacía. Terminando el programa.")
        return

    # Convertir a mayúsculas aquí para pasarlo consistente a todos los scripts
    ciudad_upper = ciudad_input.upper() 
    print(f"CIUDAD SELECCIONADA: {ciudad_upper}")
    print("\n--- INICIO DE EJECUCIÓN SECUENCIAL ---")

    # Ejecutar cada proceso en orden
    for script_name, requiere_ciudad, requiere_tesseract in PROCESOS:
        ciudad_a_pasar = ciudad_upper if requiere_ciudad else None
        tesseract_a_pasar = tesseract_path_result if requiere_tesseract else None

        # Llamar a la función de ejecución que maneja la lógica de argumentos
        if not ejecutar_script(script_name, ciudad_a_pasar, tesseract_a_pasar):
            print("\n*** EJECUCIÓN DETENIDA DEBIDO A UN ERROR EN EL PROCESO ANTERIOR. ***")
            return

    # Realizar la limpieza final
    limpiar_archivos()

    print("\n--- TODOS LOS PROCESOS Y LA FASE DE LIMPIEZA HAN FINALIZADO EXITOSAMENTE ---")

if __name__ == "__main__":
    main()