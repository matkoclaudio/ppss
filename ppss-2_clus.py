#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
###############################################################################

CORRECCIÓN :

1. CALCULO DENSIDAD DE NUBE DE PUNTOS _dist 
2. APLICA DBSCAN COMO FILTRO 

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
 
# 1. CALCULO DENSIDAD DE NUBE DE PUNTOS _dist  
        
###############################################################################

def procesar_datos(directorio_puntos):
  
    
    # Configuración del buffer en km a partir de puntos
    BUFFER_DIST_KM = 0.5 
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        
        R = 6371  # Radio de la Tierra en km
        
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return 2 * R * atan2(sqrt(a), sqrt(1 - a))
    
    def calcular_superficie_buffer(costa_shp, buffer_output_path):
        
        # Cargar archivo shapefile de la línea costera
        costa = gpd.read_file(costa_shp)
        
        # Transformar a un sistema métrico (epsg:3395 - proyección mercator para medición de área en metros)
        costa_metrica = costa.to_crs(epsg=3395)
        
        # Crear el buffer en metros y unir en una sola geometría
        buffer = costa_metrica.buffer(BUFFER_DIST_KM * 1000)  # Conversión de km a metros
        buffer_unido = unary_union(buffer)
        
        # Exportar el buffer a shapefile para inspección visual
        # gpd.GeoDataFrame(geometry=[buffer_unido], crs=costa_metrica.crs).to_file(buffer_output_path)
        
        # Calcular el área en km²
        area_km2 = buffer_unido.area / 1e6
        return buffer_unido, area_km2
    
    def calcular_densidad_minima(puntos_txt, buffer_unido, area_km2):
        
        # Leer puntos desde el archivo .txt
        df = pd.read_csv(puntos_txt, header=None, names=["long", "lat", "fecha_hora", "satelite", "nubes", "coregistro", "umbral", "mndwi"])
        
        # Crear objetos Point en el sistema de coordenadas geográficas y convertir a la misma proyección que el buffer
        puntos     = [Point(lon, lat) for lon, lat in zip(df["long"], df["lat"])]
        puntos_gdf = gpd.GeoDataFrame(geometry=puntos, crs="EPSG:4326").to_crs(epsg=3395)
        
        # Contar puntos dentro del buffer
        puntos_en_buffer = sum(1 for p in puntos_gdf.geometry if buffer_unido.contains(p))
        
        # Calcular densidad
        densidad_minima = puntos_en_buffer / area_km2
        return densidad_minima
    
    def procesar_densidad(directorio_costa, directorio_puntos):
        
        # Buscar automáticamente el archivo de puntos con terminación _dist.txt
        archivo_puntos = next((os.path.join(directorio_puntos, f) for f in os.listdir(directorio_puntos) if f.endswith('_dist.txt')), None)
        
        if archivo_puntos is None:
            print("No se encontró ningún archivo que termine en '_dist.txt' en el directorio:", directorio_puntos)
            return None  # Cambiado para devolver None en caso de error
        
        # Detectar automáticamente el archivo de costa correspondiente
        base_name     = os.path.basename(archivo_puntos)
        partes_nombre = base_name.split("_")
        
        if len(partes_nombre) < 4:
            print("Nombre de archivo no tiene el formato esperado:", base_name)
            return None  # Cambiado para devolver None en caso de error
        
        zona, año, region, _ = partes_nombre[:4]
        region_formato       = region.zfill(2)
        costa_shp            = os.path.join(directorio_costa, f"{zona}_{region_formato}.shp")
        
        if not os.path.isfile(costa_shp):
            print("Archivo de costa no encontrado:", costa_shp)
            return None  # Cambiado para devolver None en caso de error
        
        # Generar el nombre del archivo buffer
        buffer_output_path = os.path.join(directorio_costa, f"{zona}_{año}_{region}_buff{BUFFER_DIST_KM}.shp")
    
        # Calcular superficie del buffer y densidad
        buffer_unido, area_km2 = calcular_superficie_buffer(costa_shp, buffer_output_path)
        densidad_minima        = calcular_densidad_minima(archivo_puntos, buffer_unido, area_km2)
        
        print(f"Densidad mínima de puntos para {archivo_puntos} con buffer de {BUFFER_DIST_KM} km: {densidad_minima:.2f} puntos/km²")
        
        return densidad_minima  # Devolver el valor de densidad mínima
        
    # DIRECTORIOS
    directorio_costa = "/home/clod/Descargas/costa"
    # directorio_puntos = "/home/clod/Descargas/dw/07_2022_0"  # Directorio donde buscará el archivo _dist.txt
    
    # Llamada a procesar_densidad y capturar el resultado
    densidad_minima_resultado = procesar_densidad(directorio_costa, directorio_puntos)
    
    calc_densidad = int(densidad_minima_resultado / 6)  
    # con buff de 1.0km 1/2 de la densidad calculada para DBSCAN gran densidad
    # con buff de 0.5km 1/4 de la densidad calculada para DBSCAN gran densidad
    # con buff de 0.5km 1/6 de la densidad calculada para DBSCAN baja densidad
    
    print(f'dens min calc {densidad_minima_resultado}')
    print(f'dens min DBSCAN {calc_densidad}')
    

    ###########################################################################    
    
    # 2. APLICA DBSCAN COMO FILTRO 
        
    ###########################################################################

    # parámetros para DBSCAN
    epsilon  = 200  # radio de búsqueda m
    # densidad = 2000 
    densidad = calc_densidad # N° mín pts toma calculo de densidad  
    
    # Constantes
    RADIUS_EARTH_METERS = 6371000
    
    # Listas para acumular datos
    core_points_list   = []
    border_points_list = []
    noise_points_list  = []
    
    # Convertir latitudes y longitudes a coordenadas en metros
    def convertir_a_metros(latitudes, longitudes):
        lat_central     = np.mean(latitudes)
        lat_central_rad = np.radians(lat_central)
    
        lat_rad = np.radians(latitudes)
        lon_rad = np.radians(longitudes)
    
        delta_lon = lon_rad - lon_rad[0]
        delta_lat = lat_rad - lat_rad[0]
    
        x = delta_lon * RADIUS_EARTH_METERS * np.cos(lat_central_rad)
        y = delta_lat * RADIUS_EARTH_METERS
        return np.column_stack((x, y))
    
    # Leer datos a analizar
    def leer_datos(file_path):
        dtypes = {
            0: 'float64',
            1: 'float64',
            2: 'str',
            3: 'str',
            4: 'float32',
            5: 'str',
            6: 'float32',
            7: 'float32'
        }

        data = pd.read_csv(file_path, header = None, dtype = dtypes)
        data.columns = ['longitud', 'latitud', 'fecha y hora', 'satelite', 'nubes', 'coregistro', 'umbral', 'mndwi']
        # data['coregistro'] = pd.to_numeric(data['coregistro'].replace("PASSED", np.nan), errors = 'coerce')
        data['coregistro'] = np.where(data['coregistro'] == "PASSED", np.nan, data['coregistro'])
        data['coregistro'] = pd.to_numeric(data['coregistro'], errors='coerce')
        return data
    
    # Crear un directorio si no existe
    def crear_directorio(directorio):
        os.makedirs(directorio, exist_ok = True)
    
    # toma parámetros 
    eps_metros  = epsilon   # epsilon m
    min_samples = densidad  # N° mínimo de puntos 
    
    # DIRECTORIOS 
    directorio_datos       = directorio_puntos
    directorio_base_salida = directorio_datos
    # directorio_base        = os.path.dirname(directorio_datos)
    # directorio_base_salida = os.path.join(directorio_base, '2.corr')
    # crear_directorio(directorio_base_salida)
     
    # Buscar archivo de datos a procesar
    archivo = next((f for f in os.listdir(directorio_datos) if f.endswith('_dist.txt')), None)
    
    if archivo is None:
        print("No se encontró ningún archivo que termine en '_dist.txt'.")
    else:
        # Inicializar contadores
        total_puntos_totales = 0
        total_puntos_core    = 0
        total_puntos_noise   = 0
        total_puntos_border  = 0
    
        # Leer datos a analizar
        datos_a_analizar      = leer_datos(os.path.join(directorio_datos, archivo))
        total_puntos_totales += len(datos_a_analizar)
        datos_metros          = convertir_a_metros(datos_a_analizar['latitud'], datos_a_analizar['longitud'])
    
        # Barra de progreso en colores
        with tqdm(total = total_puntos_totales, desc = "Procesando puntos", colour = "green") as pbar:
            
            # Aplicar DBSCAN
            min_samples = max(10, min_samples)  # Verifica/asegura que min_samples sea al menos ese N°
            db = DBSCAN(eps = eps_metros, min_samples = min_samples, metric = 'euclidean').fit(datos_metros)
            labels = db.labels_
            datos_a_analizar['cluster'] = labels
    
            # Identificar muestras centrales (core)
            core_samples_mask = np.zeros_like(labels, dtype = bool)
            core_samples_mask[db.core_sample_indices_] = True
    
            # Separar puntos en core, border y noise
            core_points   = datos_a_analizar[core_samples_mask]
            border_points = datos_a_analizar[~core_samples_mask & (labels != -1)]
            noise_points  = datos_a_analizar[labels == -1]
    
            # Actualizar contadores
            total_puntos_core   += len(core_points)
            total_puntos_border += len(border_points)
            total_puntos_noise  += len(noise_points)
    
            # Acumular puntos en las listas
            if not core_points.empty:
                core_points_list.append(core_points)
            if not border_points.empty:
                border_points_list.append(border_points)
            if not noise_points.empty:
                noise_points_list.append(noise_points)
    
            # Definir nombres de archivos con el prefijo
            prefijo = archivo.split('_dist')[0]
            archivo_core    = os.path.join(directorio_base_salida, f'{prefijo}_core.txt')
            archivo_border  = os.path.join(directorio_base_salida, f'{prefijo}_border.txt')
            archivo_noise   = os.path.join(directorio_base_salida, f'{prefijo}_noise.txt')
    
            # Guardar puntos en archivos correspondientes
            if not core_points.empty:
                core_points.to_csv(archivo_core, index=False, header=False, sep=',', mode='a')
            if not border_points.empty:
                border_points.to_csv(archivo_border, index=False, header=False, sep=',', mode='a')
            if not noise_points.empty:
                noise_points.to_csv(archivo_noise, index=False, header=False, sep=',', mode='a')
    
            # Actualizar la barra de progreso
            pbar.update(len(datos_a_analizar))
    
        # Crear el resumen para este archivo
        puntos_por_satelite = {}
        for satelite in datos_a_analizar['satelite'].unique():
            puntos = len(datos_a_analizar[datos_a_analizar['satelite'] == satelite])
            puntos_por_satelite[satelite] = puntos_por_satelite.get(satelite, 0) + puntos
    
        # Generar el resumen final
        archivo_resumen = os.path.join(directorio_base_salida, f'{prefijo}_resumen.txt')
        with open(archivo_resumen, 'w') as f:
            f.write("RESUMEN\n\n")
    
            f.write("dist :\n")
            for satelite, cantidad in puntos_por_satelite.items():
                porcentaje = (cantidad / total_puntos_totales) * 100
                f.write(f"       {satelite} : {cantidad} ({porcentaje:.2f}%)\n")
            f.write(f"    Total : {total_puntos_totales}\n")
                
            f.write("\ndbscan :\n")
            f.write(f"   - epsilon    : {epsilon}m\n")
            f.write(f"   - densidad   : {densidad}\n")
            f.write(f"   - core   : {total_puntos_core} ({(total_puntos_core / total_puntos_totales) * 100:.2f}%)\n")
            f.write(f"   - noise  : {total_puntos_noise} ({(total_puntos_noise / total_puntos_totales) * 100:.2f}%)\n")
            f.write(f"   - border : {total_puntos_border} ({(total_puntos_border / total_puntos_totales) * 100:.2f}%)\n")
    
        print("DBSCAN aplicado")

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