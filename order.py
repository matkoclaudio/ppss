#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
###############################################################################

1. ELIMINA LOS PAQUETES ZIP VACIOS 
2. ORDENA ARCHIVOS POR AÑO 
3. UNIFICA PERÍODOS 
4. DESCOMPRIME
5. FUSIONA ARCHIVOS POR PARÁMETRO
6. REORDENA _raw.txt EN SUBCARPETAS

###############################################################################

'''
import os
import re
import glob

import shutil
import zipfile
import pandas as pd
import subprocess

import traceback

from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


DIRECTORIO = os.path.dirname(os.path.abspath(__file__))
# DIRECTORIO = '/home/clod/Descargas/07-VIEDMA' 


###############################################################################

# 1. ELIMINA LOS PAQUETES ZIP VACIOS

###############################################################################

#%%

def eliminar_zip_vacios(directorio):
   
    # Recorre el directorio y subdirectorios
    for carpeta_raiz, _, archivos in os.walk(directorio):
        for archivo in archivos:
            if archivo.endswith('.zip'):
                
                ruta_zip  = os.path.join(carpeta_raiz, archivo)
                zip_vacio = True  # Supongamos que el zip está vacío

                with zipfile.ZipFile(ruta_zip, 'r') as zip_file:
                    
                    # Itera sobre los archivos dentro del zip
                    for nombre_archivo in zip_file.namelist():
                        with zip_file.open(nombre_archivo) as f:
                            if f.read():  # Si el archivo no está vacío
                                
                                zip_vacio = False
                                break  # Sale del bucle si encuentra un archivo con contenido
                
                # Elimina el archivo zip si está vacío
                if zip_vacio:
                    print(f'*** {ruta_zip} vacío eliminado')
                    os.remove(ruta_zip)

eliminar_zip_vacios(DIRECTORIO)


###############################################################################

# 2. ORDENA ARCHIVOS POR AÑO

###############################################################################

#%%

base_dir = Path(DIRECTORIO)
# base_dir = '/home/clod/Descargas/07-VIEDMA'

# Expresión regular para extraer el año y la misión
pattern = re.compile(r'(\d{2})_(\d{4})_([A-Z0-9]+)\.zip')

# Función para crear el directorio si no existe
def create_year_dir(base_dir, year):
    year_dir = os.path.join(base_dir, year)
    if not os.path.exists(year_dir):
        os.makedirs(year_dir)
    return year_dir

# Función para ordenar los archivos por año
def ordenar_archivos_por_anio(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.zip'):
                match = pattern.match(file)
                if match:
                    zona, anio, mision = match.groups()
                    
                    # Crear el directorio del año si no existe
                    year_dir = create_year_dir(base_dir, anio)
                    
                    # Mover el archivo al nuevo directorio
                    src_file  = os.path.join(root, file)
                    dest_file = os.path.join(year_dir, file)
                    
                    print(f'{file} movido a {year_dir}')
                    shutil.move(src_file, dest_file)
    
    # Eliminar los subdirectorios originales después de mover los archivos
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not os.listdir(dir_path):  # Si el directorio está vacío
                print(f'carpeta {dir_path} *** eliminada ***')
                os.rmdir(dir_path)

# Función principal
if __name__ == "__main__":
    ordenar_archivos_por_anio(base_dir)
    print(f"\n*** ORDEN POR AÑO COMPLETO ***")

'''
###############################################################################

# 3. UNIFICA PERÍODOS 

###############################################################################

#%%

directorio_principal = Path(DIRECTORIO)  

cantidad_años  = 6     # Cuántos años por grupo
periodo_inicio = 2010  # Inicio del período
periodo_fin    = 2015  # Fin del período


def agrupar_archivos_por_periodo(directorio, periodo_inicio, periodo_fin, cantidad_años):
    
    # Generar la lista completa de años en el rango sin importar si tienen carpetas o no
    años_completos = list(range(periodo_inicio, periodo_fin + 1))

    # Filtrar solo los subdirectorios que existen dentro del directorio
    subdirectorios = sorted([d for d in os.listdir(directorio) if os.path.isdir(os.path.join(directorio, d))])
    
    # Filtrar los subdirectorios que son años válidos dentro del rango
    años_existentes = {int(d) for d in subdirectorios if d.isdigit() and periodo_inicio <= int(d) <= periodo_fin}

    # Agrupar por el intervalo de 'cantidad_años'
    for i in range(0, len(años_completos), cantidad_años):
        inicio_grupo = años_completos[i]
        fin_grupo    = min(inicio_grupo + cantidad_años - 1, periodo_fin)
        nombre_carpeta_grupo = f"{inicio_grupo}-{fin_grupo}"

        # Crear la nueva carpeta si no existe
        nuevo_directorio = os.path.join(directorio, nombre_carpeta_grupo)
        if not os.path.exists(nuevo_directorio):
            os.makedirs(nuevo_directorio)

        # Mover los archivos de cada año del grupo al nuevo directorio, si existen
        for año in range(inicio_grupo, fin_grupo + 1):
            año_str  = str(año)
            ruta_año = os.path.join(directorio, año_str)

            if año in años_existentes:
                archivos = os.listdir(ruta_año)
                
                for archivo in archivos:
                    ruta_archivo = os.path.join(ruta_año, archivo)
                    
                    if os.path.isfile(ruta_archivo):
                        shutil.move(ruta_archivo, nuevo_directorio)

                # Eliminar la carpeta del año si estaba en el rango
                shutil.rmtree(ruta_año)
                print(f"{ruta_año} *** eliminada ***")
            else:
                print(f"{directorio}/{año_str} *** no existe ***")

    print(f"\n*** PERÍODOS COMPLETOS ***")


# Llamar la función con los parámetros configurados
agrupar_archivos_por_periodo(directorio_principal, periodo_inicio, periodo_fin, cantidad_años)

'''
###############################################################################

# 4. DESCOMPRIME

###############################################################################

#%%

def extract_txt_from_zips(base_dir):
    
    # Iterar sobre cada carpeta dentro del directorio base
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            print(f"extracción {file}...")

            # Verificar si el archivo es un paquete zip
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Extraer todos los archivos a una carpeta temporal
                    extraction_path = os.path.join(root, 'extracted_txts')
                    os.makedirs(extraction_path, exist_ok=True)
                    zip_ref.extractall(extraction_path)

                # Mover solo los archivos .txt extraídos a la carpeta principal
                for member in os.listdir(extraction_path):
                    member_path = os.path.join(extraction_path, member)

                    # Si es un archivo, moverlo; si es un directorio, buscar .txt en él
                    if os.path.isfile(member_path) and member.endswith('.txt'):
                        shutil.move(member_path, os.path.join(root, member))
                    elif os.path.isdir(member_path):
                        for root_dir, _, txt_files in os.walk(member_path):
                            for txt_file in txt_files:
                                if txt_file.endswith('.txt'):
                                    txt_file_path = os.path.join(root_dir, txt_file)
                                    shutil.move(txt_file_path, os.path.join(root, txt_file))

                # Eliminar el archivo zip
                os.remove(zip_path)
                
                # Eliminar la carpeta temporal de extracción
                shutil.rmtree(extraction_path)

    # Eliminar carpetas vacías
    for root, dirs, _ in os.walk(base_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                os.rmdir(dir_path)  # Intentar eliminar la carpeta
            except OSError:
                pass  # Si la carpeta no está vacía, se omite

# directorio base donde están los paquetes zip
base_directory = DIRECTORIO
extract_txt_from_zips(base_directory)

print(f'\n*** EXTRACCIÓN COMPLETA ***')

###############################################################################

# 5. FUSIONA ARCHIVOS POR PARÁMETRO

###############################################################################

#%%

directorio_base = DIRECTORIO

# Ruta del archivo de errores (guardado en el mismo directorio del script)
ruta_errores = os.path.join(os.path.dirname(__file__), 'errores.txt')

# Función para manejar la escritura de errores en el archivo
def guardar_error(mensaje_error):
    with open(ruta_errores, 'a') as f:
        f.write(mensaje_error + '\n')

# Recorre cada carpeta dentro del directorio base
for año in os.listdir(directorio_base):
    ruta_año = os.path.join(directorio_base, año)

    # Verifica que sea un directorio
    if os.path.isdir(ruta_año):
        zona = os.path.basename(directorio_base).split('-')[0]  # Obtener solo el número de zona '07'
        archivos_region = {}
    
        # Recorre los archivos en la carpeta del año
        for archivo in os.listdir(ruta_año):
            # Filtra archivos que terminan en .txt
            if archivo.endswith('.txt'):
                archivo_path = os.path.join(ruta_año, archivo)
                
                # Verifica si el archivo no está vacío
                if os.path.getsize(archivo_path) == 0:
                    print(f"{archivo} está vacío, omitiendo...")
                    continue
                
                # Intenta leer el archivo con diferentes codificaciones
                try:
                    df = pd.read_csv(archivo_path, header=None, encoding='utf-8')  # Intenta leer como UTF-8
                except UnicodeDecodeError as e:
                    mensaje_error = f"Error de codificación (UnicodeDecodeError) en archivo {archivo}, línea {e.args[1]}. Omite el archivo."
                    print(mensaje_error)
                    guardar_error(mensaje_error)
                    continue
                except Exception as e:
                    mensaje_error = f"Error general al leer {archivo}: {e}, omitiendo...\n{traceback.format_exc()}"
                    print(mensaje_error)
                    guardar_error(mensaje_error)
                    continue
                
                # Obtiene el número de región del archivo
                parte_region = archivo.split('_')[-1].split('.')[0]  # '0' de '07_2021_L8_0.txt'
                
                if parte_region not in archivos_region:
                    archivos_region[parte_region] = []  # Crear una lista para la región
                
                # Añade el DataFrame a la lista correspondiente
                archivos_region[parte_region].append(df)

        # Procesa cada grupo de archivos con el mismo número de región
        for region, dfs in archivos_region.items():
            # Fusiona todos los DataFrames en uno solo, eliminando líneas con errores
            dfs_limpios = []
            for df in dfs:
                df_limpio = df.dropna()  # Elimina filas con valores NaN (errores)
                dfs_limpios.append(df_limpio)

            df_fusionado = pd.concat(dfs_limpios, ignore_index=True)
            
            # Genera el nombre del archivo resultante
            nombre_archivo_resultante = f"{zona}_{año}_{region}_raw.txt"
            ruta_resultante = os.path.join(ruta_año, nombre_archivo_resultante)
            
            # Guarda el DataFrame fusionado en un nuevo archivo
            try:
                df_fusionado.to_csv(ruta_resultante, index=False, header=False)
                print(f"Fusionado {nombre_archivo_resultante}...")
            except Exception as e:
                mensaje_error = f"Error al guardar {nombre_archivo_resultante}: {e}\n{traceback.format_exc()}"
                print(mensaje_error)
                guardar_error(mensaje_error)

        # Elimina archivos que no terminen en '_raw.txt'
        for archivo in os.listdir(ruta_año):
            if not archivo.endswith('_raw.txt') and archivo.endswith('.txt'):
                try:
                    os.remove(os.path.join(ruta_año, archivo))
                    print(f"{archivo} eliminado...")
                except Exception as e:
                    mensaje_error = f"Error al eliminar {archivo}: {e}\n{traceback.format_exc()}"
                    print(mensaje_error)
                    guardar_error(mensaje_error)

print(f'\n*** FUSIÓN COMPLETA ***')



###############################################################################

# 6. REORDENA 

###############################################################################

#%%

def reubicar_txt(directorio):
    
    # Recorre las subcarpetas del directorio proporcionado
    for subdir, _, archivos in os.walk(directorio):
        for archivo in archivos:
            
            # Verifica si el archivo es un .txt
            if archivo.endswith('.txt'):
            
                # Define la ruta completa del archivo
                ruta_archivo = os.path.join(subdir, archivo)
                
                # Extrae el nombre sin la extensión y elimina el sufijo '_raw' si está presente
                nombre_sin_raw = archivo.replace('_raw', '')
                nombre_carpeta = os.path.splitext(nombre_sin_raw)[0]
                
                # Crea la nueva carpeta con el nombre del archivo (sin la extensión y sin '_raw')
                nueva_carpeta = os.path.join(subdir, nombre_carpeta)
                os.makedirs(nueva_carpeta, exist_ok=True)
                
                # Mueve el archivo a la nueva carpeta
                shutil.move(ruta_archivo, os.path.join(nueva_carpeta, archivo))
                print(f'Movido: {ruta_archivo} a {nueva_carpeta}/{archivo}')

# directorio a procesar
directorio_principal = DIRECTORIO
reubicar_txt(directorio_principal)

print(f'\n*** LISTO PARA EL POSPROCESO ***')



if __name__ == "__main__":
    main()


