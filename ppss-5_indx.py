#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
###############################################################################

1. CALCULA EL INDICE DE CONFIANZA (IC) DE UNA SERIE DE DATOS
- tpem = raíz cuadrada de (((pixel/2) * (pixel/2)) + (coregistro * coregistro))
- IC : (tpem promedio * 100) / (raiz cuadrada(densidad))

2. INCERTA CAMPOS DE PERÍODO E IC EN LOS ARCHIVOS VECTORIALES

###############################################################################

'''

import os
import sys
import re
import math
import geopandas as gpd
from datetime import datetime

def procesar_datos(directorio_puntos):

    '''
    ###########################################################################

    1. CALCULA EL INDICE DE CONFIANZA (IC) DE UNA SERIE DE DATOS

    ###########################################################################

    '''

    #DIRECTORIO    = '/home/clod/Descargas/calculo'
    DIRECTORIO    = directorio_puntos
    COREGISTRO_S2 = 10.00  # Coregistro específico S2

    # Función para cargar el archivo que termina en _resumen.txt
    def cargar_densidad_resumen(directorio):
        resumen_path = None
        for archivo in os.listdir(directorio):
            if archivo.endswith('_resumen.txt'):
                resumen_path = os.path.join(directorio, archivo)
                break
        if not resumen_path:
            raise FileNotFoundError("Archivo _resumen.txt no encontrado.")
        
        with open(resumen_path, 'r') as f:
            contenido = f.read()
            densidad  = re.search(r"   - densidad   : (\d+)", contenido)
            if densidad:
                return float(densidad.group(1)), resumen_path
            else:
                raise ValueError("No se encontró la densidad en el archivo _resumen.txt.")

    # Función para procesar el archivo core.txt
    def procesar_archivo_core(directorio, coregistro_s2):
        archivo_core = None
        for archivo in os.listdir(directorio):
            if archivo.endswith('_core.txt'):
                archivo_core = os.path.join(directorio, archivo)
                break
        if not archivo_core:
            raise FileNotFoundError("Archivo _core.txt no encontrado.")

        imagenes = {}
        with open(archivo_core, 'r') as f:
            for linea in f:
                datos = linea.strip().split(',')
                if len(datos) < 9:
                    continue  # Saltar líneas incompletas

                lat, lon, fecha_str, satelite, nubes, coregistro, umbral, mndwi, _ = datos
                coregistro = round(float(coregistro), 2) if coregistro else coregistro_s2
                coregistro = f"{coregistro:05.2f}"  # Formato con dos decimales y cero inicial

                # Procesamiento de fecha
                fecha_dt     = datetime.strptime(fecha_str, "%d/%m/%Y %H:%M:%S")
                fecha_format = fecha_dt.strftime("%Y%m%d%H%M%S")

                # Calcular el tamaño de pixel
                if satelite == "S2":
                    pixel = 10
                elif satelite in ["L5", "L7", "L8", "L9"]:
                    pixel = 15
                else:
                    continue  # Satélite desconocido

                # Determinar el nombre único de la imagen
                nombre_imagen = f"{fecha_format}_{satelite}-{pixel}-{coregistro}"

                # Calcular tpem
                tpem = round(math.sqrt(((pixel / 2) ** 2) + (float(coregistro) ** 2)), 2)
                imagenes[nombre_imagen] = tpem

        # Ordenar las imágenes en orden descendente
        imagenes = dict(sorted(imagenes.items(), reverse=True))

        return imagenes

    # Función para calcular la media de tpem y el IC
    def calcular_ic_y_promedio(imagenes, densidad):
        tpem_total    = sum(imagenes.values())
        tpem_promedio = round(tpem_total / len(imagenes), 2)
        ic = round((tpem_promedio * 100) / math.sqrt(densidad), 2)
        return tpem_promedio, ic

    # Función para escribir los resultados en el archivo _resumen.txt
    def escribir_resultados(resumen_path, imagenes, tpem_promedio, ic):
        with open(resumen_path, 'a') as f:
            f.write("\n\nimágen_sat_pix_co_tpem :\n")
            for nombre, tpem in imagenes.items():
                f.write(f"{nombre}_{tpem}\n")
            f.write(f"tpem promedio : {tpem_promedio}\n")
            f.write(f"IC : {ic}\n")

    # Ejecución principal del programa
    def main():
        try:
            densidad, resumen_path = cargar_densidad_resumen(DIRECTORIO)
            imagenes = procesar_archivo_core(DIRECTORIO, COREGISTRO_S2)
            tpem_promedio, ic = calcular_ic_y_promedio(imagenes, densidad)
            escribir_resultados(resumen_path, imagenes, tpem_promedio, ic)
            print("***ic calculado***")
        except Exception as e:
            print(f"Ocurrió un error: {e}")

    if __name__ == "__main__":
        main()


    '''
    ###########################################################################

    2. INCERTA CAMPOS DE PERÍODO E IC EN LOS ARCHIVOS VECTORIALES

    ###########################################################################

    '''

    # Función para extraer el período del nombre del archivo
    def extraer_periodo(nombre_archivo):
        partes = nombre_archivo.split('_')
        if len(partes) >= 2:
            return partes[1]
        return None

    # Función para extraer el valor de IC del archivo resumen
    def extraer_ic(archivo_resumen):
        with open(archivo_resumen, 'r') as f:
            for linea in f:
                match = re.search(r'IC\s*:\s*(\d+(\.\d+)?)', linea)
                if match:
                    return float(match.group(1))
        return None

    # Procesamiento de archivos
    for archivo in os.listdir(DIRECTORIO):
        ruta_archivo = os.path.join(DIRECTORIO, archivo)
        
        # Procesar shapefiles
        if archivo.endswith('_lin.shp'):
            # Leer el shapefile
            gdf = gpd.read_file(ruta_archivo)
            
            # Extraer período del nombre del archivo
            periodo = extraer_periodo(archivo)
            
            # Buscar archivo de resumen correspondiente y extraer IC
            resumen_archivo = archivo.replace('_lin.shp', '_resumen.txt')
            resumen_ruta    = os.path.join(DIRECTORIO, resumen_archivo)
            if os.path.exists(resumen_ruta):
                ic = extraer_ic(resumen_ruta)
            else:
                ic = None
            
            # Agregar los campos "periodo" e "IC"
            gdf['periodo'] = periodo
            gdf['IC'] = ic
            
            # Guardar el shapefile modificado
            gdf.to_file(ruta_archivo)

        # Procesar GeoJSON
        elif archivo.endswith('_int.geojson'):
            # Leer el archivo GeoJSON
            gdf = gpd.read_file(ruta_archivo)
            
            # Extraer período del nombre del archivo
            periodo = extraer_periodo(archivo)
            
            # Buscar archivo de resumen correspondiente y extraer IC
            resumen_archivo = archivo.replace('_int.geojson', '_resumen.txt')
            resumen_ruta    = os.path.join(DIRECTORIO, resumen_archivo)
            if os.path.exists(resumen_ruta):
                ic = extraer_ic(resumen_ruta)
            else:
                ic = None
            
            # Agregar los campos "periodo" e "IC"
            gdf['periodo'] = periodo
            gdf['IC'] = ic
            
            # Guardar el archivo GeoJSON modificado
            gdf.to_file(ruta_archivo, driver='GeoJSON')

    print("***ic incertado***")
            
###############################################################################

# CIERRE DEL MAIN

###############################################################################

if __name__ == "__main__":
    # Toma la ruta de la carpeta pasada como argumento
    if len(sys.argv) > 1:
        directorio_puntos = sys.argv[1]
    else:
        print("Error: No se ha proporcionado una ruta de carpeta")
        sys.exit(1)
    
    # Ejecuta la función de procesamiento en la carpeta especificada
    procesar_datos(directorio_puntos)