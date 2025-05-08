#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
###############################################################################

CORRECCIÓN :

1. APLICA UN FILTRO DE ACUERDO A UNA DISTANCIA DETERMINADA 

###############################################################################

'''

import os
import sys
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.ops import unary_union
from shapely.geometry import Point

from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import DBSCAN
from tqdm import tqdm


###############################################################################
 
# 1. FILTRO DE DISTANCIA 

###############################################################################

def procesar_datos(directorio_puntos):
  

    def procesar_puntos_y_buffer(directorio_costa, directorio_puntos, buffer_dist_km):
        # Buscar automáticamente el archivo de puntos con terminación _raw.txt
        archivo_puntos = next((os.path.join(directorio_puntos, f) for f in os.listdir(directorio_puntos) if f.endswith('_raw.txt')), None)
        
        if archivo_puntos is None:
            print("No se encontró ningún archivo que termine en '_raw.txt' en el directorio:", directorio_puntos)
            return None  # Cambiado para devolver None en caso de error
        
        # Detectar automáticamente el archivo de costa correspondiente
        base_name = os.path.basename(archivo_puntos)
        partes_nombre = base_name.split("_")
        
        if len(partes_nombre) < 4:
            print("Nombre de archivo no tiene el formato esperado:", base_name)
            return None  # Cambiado para devolver None en caso de error
        
        zona, año, region, _ = partes_nombre[:4]
        region_formato = region.zfill(2)
        costa_shp = os.path.join(directorio_costa, f"{zona}_{region_formato}.shp")
        
        if not os.path.isfile(costa_shp):
            print("Archivo de costa no encontrado:", costa_shp)
            return None  # Cambiado para devolver None en caso de error
        
        # Generar el nombre del archivo buffer
        buffer_output_path = os.path.join(directorio_costa, f"{zona}_{año}_{region}_buff{buffer_dist_km}.shp")
        
        # Leer puntos desde el archivo .txt
        df = pd.read_csv(archivo_puntos, header=None, names=["long", "lat", "fecha_hora", "satelite", "nubes", "coregistro", "umbral", "mndwi"])
        
        # Crear objetos Point en el sistema de coordenadas geográficas
        puntos = [Point(lon, lat) for lon, lat in zip(df["long"], df["lat"])]
        
        # Crear un GeoDataFrame con los puntos
        puntos_gdf = gpd.GeoDataFrame(df, geometry=puntos, crs="EPSG:4326")
        
        # Transformar a la misma proyección que el archivo de costa
        puntos_gdf = puntos_gdf.to_crs(epsg=3395)
        
        # Cargar el archivo shapefile de la costa
        costa = gpd.read_file(costa_shp)
        
        # Transformar a la misma proyección
        costa_metrica = costa.to_crs(epsg=3395)
        
        # Crear el buffer en metros
        buffer = costa_metrica.buffer(buffer_dist_km * 1000)  # Conversión de km a metros
        buffer_unido = buffer.unary_union
        
        # Filtrar puntos que estén dentro del buffer
        puntos_en_buffer = puntos_gdf[puntos_gdf.geometry.within(buffer_unido)]
        
        # Eliminar la columna de geometría para evitar la aparición de la palabra "POINT"
        puntos_en_buffer = puntos_en_buffer.drop(columns='geometry')
        
        # Guardar el nuevo archivo con los puntos dentro del buffer (sin encabezados)
        output_file = archivo_puntos.replace('_raw.txt', '_dist.txt')
        puntos_en_buffer.to_csv(output_file, index=False, header=False)
        
        print(f"Se han guardado los puntos dentro del buffer en: {output_file}")
        
        return output_file

    # Ejemplo de uso:
    directorio_costa = "/home/clod/Descargas/costa"
    # directorio_puntos = "/home/clod/Descargas/proba_dist"
    buffer_dist_km = 2  # Distancia del buffer en km

    procesar_puntos_y_buffer(directorio_costa, directorio_puntos, buffer_dist_km)


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