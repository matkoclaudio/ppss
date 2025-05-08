#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
###############################################################################

PARA EJECUTARSE DESDE main.py

###############################################################################

'''

import os
import re
import sys
import glob
import math
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

import cv2
import matplotlib.pyplot as plt
import rasterio
import pyproj

import colorcet as cc
import holoviews as hv
import datashader as ds

from shapely.ops import unary_union
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry import Polygon

from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from holoviews.element.tiles import EsriImagery
from holoviews.operation.datashader import datashade

from rasterio.transform import from_origin
from rasterio.warp import transform_geom
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.features import geometry_mask

from rich.progress import track
from bokeh.io import output_file, save
from scipy.ndimage import gaussian_filter
from datetime import datetime


# directorio_puntos = "/home/clod/Descargas/02-RDPLATA"  # Directorio donde buscará el archivo _raw.txt

def procesar_directorio(directorio_puntos):
    print(f"Procesando el directorio: {directorio_puntos}")

    
    ###############################################################################
     
    # 1. FILTRO DE DISTANCIA 
    
    ###############################################################################
    
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
    buffer_dist_km = 0.5  # Distancia del buffer en km
    procesar_puntos_y_buffer(directorio_costa, directorio_puntos, buffer_dist_km)
    
    
    ###############################################################################
     
    # 2. CALCULO DENSIDAD DE NUBE DE PUNTOS _dist  
            
    ###############################################################################
    
    
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
    
    # 3. APLICA DBSCAN COMO FILTRO 
        
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

    # 4. KERNEL-DENSITY

    ###############################################################################    


    hv.extension('bokeh')

    input_dir  = directorio_puntos
    output_dir = input_dir
    os.makedirs(output_dir, exist_ok = True)

    # Tamaño de píxel en el terreno m
    pixel_size = 5  

    # Configuración de sistemas de coordenadas
    proj_4326   = pyproj.CRS('EPSG:4326')  # WGS84
    proj_3857   = pyproj.CRS('EPSG:3857')  # Google Mercator
    transformer = pyproj.Transformer.from_crs(proj_4326, proj_3857, always_xy = True)

    # Identificar el archivo que termina en 'resumen.txt'
    resumen_file = None
    
    for file in os.listdir(input_dir):
        if file.endswith('resumen.txt'):
            resumen_file = file
            break

    if resumen_file is None:
        print("No se encontró ningún archivo que termine en 'resumen.txt'.")
        exit()

    # Agregar el término 'krnl' al final del archivo resumen
    file_path = os.path.join(input_dir, resumen_file)

    # Leer el contenido del archivo
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Agregar el nuevo contenido
    kernel_info = f"\nraster :\n   - resolución : {pixel_size}m\n"
    lines.append(kernel_info)

    # Escribir el contenido de vuelta al archivo
    with open(file_path, 'w') as file:
        file.writelines(lines)

    # Procedimientos restantes del código original
    # Identificar el archivo que termina en core.txt
    core_file = None
    for file in os.listdir(input_dir):
        if file.endswith('core.txt'):
            core_file = file
            break

    if core_file is None:
        print("No se encontró ningún archivo que termine en 'core.txt'.")
        exit()

    file_path = os.path.join(input_dir, core_file)
    df = pd.read_csv(file_path, header=None, usecols=[0, 1], names=['longitud', 'latitud'])

    # Convertir coordenadas de WGS84 a Google Mercator
    df['longitud'], df['latitud'] = transformer.transform(df['longitud'].values, df['latitud'].values)

    # Crear capas de mapa y puntos
    # Calcular el número de píxeles para el ancho y alto de la imagen en función del tamaño de píxel deseado
    x_min, y_min = df['longitud'].min(), df['latitud'].min()
    x_max, y_max = df['longitud'].max(), df['latitud'].max()
    resolution_x = int((x_max - x_min) / pixel_size)
    resolution_y = int((y_max - y_min) / pixel_size)

    map_tiles  = EsriImagery().opts(alpha=0.5, width=resolution_x, height=resolution_y, bgcolor='black')
    points     = hv.Points(df, ['longitud', 'latitud'])
    heatmap    = datashade(points, cmap=cc.fire, width=resolution_x, height=resolution_y)

    # Combinar el mapa base con el heatmap
    final_map = map_tiles * heatmap

    # Obtener el nombre base del archivo core
    base_name = os.path.splitext(core_file)[0].replace('core', 'krnl')  # Sin la extensión .txt y reemplazo

    # Guardar la salida en formato HTML
    # output_name_html = f'{base_name}.html'
    # output_path_html = os.path.join(output_dir, output_name_html)
    # output_file(output_path_html)
    # save(hv.render(final_map, backend='bokeh'))
    # print(f'Guardado en: {output_path_html}')

    # Generar el heatmap como un raster
    canvas = ds.Canvas(plot_width = resolution_x, plot_height = resolution_y)
    agg    = canvas.points(df, 'longitud', 'latitud')
    img    = ds.tf.shade(agg, cmap = cc.fire)

    # Convertir la imagen a escala de grises (para representarlo como un DEM)
    img_gray  = img.to_pil().convert('L')

    # Convertir la imagen PIL a un array NumPy
    img_array = np.array(img_gray)

    # Normalizar los valores para que representen elevaciones
    img_array = (img_array / img_array.max()) * 1000  # Normalización a un rango de 0 a 1000 (por ejemplo)

    # Definir las coordenadas del GeoTIFF
    x_min, y_min = df['longitud'].min(), df['latitud'].min()
    x_max, y_max = df['longitud'].max(), df['latitud'].max()

    # Crear la transformación para rasterio usando el tamaño de píxel fijo de 20 m
    transform = from_origin(x_min, y_max, pixel_size, pixel_size)

    # Guardar la imagen en formato GeoTIFF
    output_name_tiff = f'{base_name}.tif'
    output_path_tiff = os.path.join(output_dir, output_name_tiff)

    with rasterio.open(
        output_path_tiff,
        'w',
        driver = 'GTiff',
        height = img_array.shape[0],
        width  = img_array.shape[1],
        count  = 1,
        dtype  = img_array.dtype,
        crs    = proj_3857.to_string(),
        transform = transform,
    ) as dst:
        dst.write(img_array, 1)

    print(f'Guardado en: {output_path_tiff}')

    ###########################################################################
    
    # 5. OTSU-BINARIZACIÓN 
    # http://labbookpages.co.uk/software/imgProc/otsuThreshold.html
    
    ###########################################################################

    def otsu_threshold(image_path, output_path, apply_gaussian=False, sigma=0):
        # 3x3 sigma = 1 
        # 5x5 sigma = 2
        # 7x7 sigma = 3
        
        # Abrir la imagen y leer datos con rasterio
        with rasterio.open(image_path) as src:
            
            # Leer la primera banda de la imagen (suponiendo que es una imagen en escala de grises)
            band    = src.read(1, resampling=Resampling.nearest)
            profile = src.profile  # Guardar el perfil de metadatos de la imagen original
        
        # Aplicar filtro gaussiano si es necesario
        if apply_gaussian:
            band = gaussian_filter(band, sigma=sigma)
        
        # Calcular el histograma
        histData, bin_edges = np.histogram(band, bins=256, range=(0, 256))
        
        # Número total de píxeles
        total     = band.size
        sum_total = np.dot(np.arange(256), histData)
        sumB      = 0.0
        wB        = 0
        wF        = 0
        varMax    = 0.0
        threshold = 0

        for t in range(256):
            
            wB += histData[t]  # Fondo de Peso
            if wB == 0:
                continue
            
            wF = total - wB  # Peso Primer plano
            if wF == 0:
                break

            sumB += t * histData[t]
            mB    = sumB / wB  # Antecedentes Medios
            mF    = (sum_total - sumB) / wF  # Primer plano Medio
            
            # Calcular Entre la Varianza de Clase
            varBetween = wB * wF * (mB - mF) ** 2
            
            # Compruebe si se encuentra un nuevo máximo
            if varBetween > varMax:
                varMax    = varBetween
                threshold = t

        # Crear imagen binaria usando el umbral calculado
        binary_data = np.where(band > threshold, 255, 0).astype(np.uint8)
        
        # Actualizar el perfil para la imagen binaria
        profile.update(
            dtype    = rasterio.uint8,
            count    = 1,
            compress = 'lzw'
        )

        # Verificar y ajustar nodata si está presente
        if 'nodata' in profile and profile['nodata'] is not None:
            nodata = profile['nodata']
            if nodata < 0 or nodata > 255:
                profile.pop('nodata')

        # Guardar la imagen binaria manteniendo la georeferenciación
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(binary_data, 1)

        # Imprimir el histograma
        # plt.figure(figsize = (10, 6))
        # plt.bar(range(256), histData, width = 1, color = 'black')
        # plt.axvline(threshold, color = 'red', linestyle = 'dashed', linewidth = 2)
        # plt.title('Histograma de la imagen con umbral')
        # plt.xlabel('Intensidad de píxel')
        # plt.ylabel('Frecuencia')
        # plt.show()

        return threshold

    # Buscar el archivo que termina en 'krnl.tif' en el directorio de trabajo
    # input_dir = '/home/clod/Descargas/de/'  
    input_file = None

    for file in os.listdir(input_dir):
        if file.endswith('krnl.tif'):
            input_file = os.path.join(input_dir, file)
            break

    if input_file:
        # Definir la ruta de salida reemplazando 'krnl' por 'bin' en el nombre del archivo
        output_file = input_file.replace('krnl', 'bin')
        
        umbral = otsu_threshold(input_file, output_file, apply_gaussian = False, sigma = 0)
        print(f"El umbral óptimo es: {umbral}")
        print(f"La imagen binaria se ha guardado en: {output_file}")

        # Agregar umbral binario al archivo resumen
        resumen_file = None
        for file in os.listdir(input_dir):
            if file.endswith('resumen.txt'):
                resumen_file = os.path.join(input_dir, file)
                break

        if resumen_file:
            
            # Leer el contenido del archivo resumen
            with open(resumen_file, 'r') as f:
                lines = f.readlines()
            
            # Buscar la línea donde agregar el umbral binario
            for i, line in enumerate(lines):
                if '- resolución :' in line:
                    
                    # Insertar la nueva línea debajo
                    lines.insert(i + 1, f"\n   - umbral binario : {umbral}\n")
                    break

            # Escribir de nuevo el archivo resumen
            with open(resumen_file, 'w') as f:
                f.writelines(lines)
        else:
            print("No se encontró ningún archivo que termine en 'resumen.txt'.")
    else:
        print("No se encontró ningún archivo que termine en 'krnl.tif'.")

    ###########################################################################
    
    # 5. VECTORIZA
    
    ###########################################################################
    
    # input_dir = '/home/clod/Descargas/de'

    # Función para procesar cada archivo GeoTIFF
    def raster_to_vector(input_file):
        with rasterio.open(input_file) as src:
            
            # Leer la imagen como un array
            image = src.read(1)

            # Convertir a valores binarios (0 o 255)
            _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

            # Generar formas geométricas a partir del raster binario
            mask    = binary_image == 255  # Solo áreas blancas
            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) in enumerate(
                    shapes(binary_image, mask = mask, transform = src.transform))
            )
            
            # Crear una lista de polígonos a partir de las formas generadas
            polygons = []
            for result in results:
                geom = result['geometry']
                geom_transformed = transform_geom(src.crs, 'EPSG:4326', geom)
                polygons.append(Polygon(geom_transformed['coordinates'][0]))

            # Crear un GeoDataFrame
            if polygons:  # Solo proceder si hay polígonos válidos
                gdf = gpd.GeoDataFrame({'geometry': polygons}, crs = 'EPSG:4326')

                # Guardar como GeoJSON en coordenadas geográficas
                output_file = os.path.splitext(input_file)[0].replace('bin', 'int') + '.geojson'
                gdf.to_file(output_file, driver = 'GeoJSON')
                print(f'Guardado en: {output_file}')
            
            else:
                print(f'No se encontraron contornos válidos en {input_file}')

    # Procesar 'bin.tif' en el directorio de entrada
    for file in os.listdir(input_dir):
        if file.endswith('bin.tif'):
            input_file = os.path.join(input_dir, file)
            raster_to_vector(input_file)


    '''
    ###############################################################################
    
    # 6. DIGITALIZACIÓN AUTOMÁTICA 
    
    ###############################################################################
    '''

    # Configurar número de iteraciones
    iteraciones = 3  # más de 3 es demasiado, menos es un error ))

    directorio_trabajo = directorio_puntos

    # busca pseudodem 'krnl'
    dem_path = glob.glob(os.path.join(directorio_trabajo, '*krnl.tif'))[0]  # Toma el primer archivo encontrado

    # obtiene nombre base del 'krnl'
    nombre_base = os.path.basename(dem_path).replace('_krnl.tif', '')

    # Extrae el número de región
    numero_region = nombre_base.split('_')[0]

    # Busca el archivo .shp correspondiente a la región
    entrada = f'/home/clod/Descargas/costa2/{numero_region}.shp'  # línea costera base

    # salidas
    salida                 = os.path.join(directorio_trabajo, f"{nombre_base}_prf.shp")  # perfiles
    salida_puntos_max_elev = os.path.join(directorio_trabajo, f"{nombre_base}_pts.shp")  # puntos
    salida_linea_max_elev  = os.path.join(directorio_trabajo, f"{nombre_base}_lin.shp")  # línea

    largo_linea    = 700  # (700) en metros >> e/ 500 y 1000  
    interdistancia = 100  # (100) en metros >> e/ 100 y 200 
    grado_smooth   = 2    # más de 2 generaliza demasiado

    # Función para suavizar la línea
    def suavizar_linea(linea, grado):
        if grado < 1:
            return linea
        # Suavizado simple: toma un promedio de puntos alrededor de cada punto
        coords = list(linea.coords)
        smoothed_coords = []
        
        for i in range(len(coords)):
            if i < grado:
                # Promedia con puntos existentes al principio
                smoothed_point = (
                    sum(x for x, _ in coords[:i + grado + 1]) / (i + grado + 1),
                    sum(y for _, y in coords[:i + grado + 1]) / (i + grado + 1),
                )
            elif i + grado >= len(coords):
                # Promedia con puntos existentes al final
                smoothed_point = (
                    sum(x for x, _ in coords[i - grado:]) / (len(coords) - i + grado),
                    sum(y for _, y in coords[i - grado:]) / (len(coords) - i + grado),
                )
            else:
                # Promedia con puntos a la izquierda y derecha
                smoothed_point = (
                    sum(x for x, _ in coords[i - grado:i + grado + 1]) / (2 * grado + 1),
                    sum(y for _, y in coords[i - grado:i + grado + 1]) / (2 * grado + 1),
                )
            smoothed_coords.append(smoothed_point)
        
        return LineString(smoothed_coords)

    # Función para procesar una iteración
    def procesar_linea_costera(entrada_shp, salida_perfiles, salida_puntos, salida_linea):
        # carga línea costera
        gdf = gpd.read_file(entrada_shp)

        # verifica que haya solo una línea
        if len(gdf) != 1:
            raise ValueError("El shapefile debe contener exactamente una línea.")

        # reproyecta a GM
        gdf = gdf.to_crs(epsg=3857)
        linea_costera = gdf.geometry.iloc[0]

        # crea perpendiculares 
        def crear_linea_perpendicular(punto, largo, angulo):
            dx = math.cos(math.radians(angulo)) * largo / 2
            dy = math.sin(math.radians(angulo)) * largo / 2
            start = Point(punto.x - dx, punto.y - dy)
            end   = Point(punto.x + dx, punto.y + dy)
            return LineString([start, end])

        # lista perpendiculares
        lineas_perpendiculares = []

        # itera sobre puntos de la línea costera a intervalos de 'interdistancia'
        for dist in track(range(0, int(linea_costera.length), int(interdistancia)), description = "Generando perpendiculares..."):
            
            # obtiene puntos a la distancia especificada
            punto = linea_costera.interpolate(dist)
            
            # obtiene dirección de la línea costera en ese punto
            siguiente_punto = linea_costera.interpolate(min(dist + 1, linea_costera.length))
            dx = siguiente_punto.x - punto.x
            dy = siguiente_punto.y - punto.y
            direccion = math.degrees(math.atan2(dy, dx)) + 90  # perpendicular

            # crea línea perpendicular
            linea = crear_linea_perpendicular(punto, largo_linea, direccion)
            lineas_perpendiculares.append(linea)

        # crea GeoDF con perpendiculares
        gdf_perpendiculares = gpd.GeoDataFrame(geometry = lineas_perpendiculares, crs = gdf.crs)

        # guarda perpendiculares en nuevo shapefile
        # gdf_perpendiculares.to_file(salida_perfiles)

        # procesa cada perpendicular para encontrar el punto de máx densidad
        puntos_max_elevacion = []

        # carga pseudodem
        with rasterio.open(dem_path) as dem:
            for idx, linea in track(enumerate(gdf_perpendiculares.geometry), description = "Buscando puntos de máxima densidad..."):
                
                # crea máscara para la línea perpendicular en el pseudodem
                mask = geometry_mask([linea], transform = dem.transform, invert = True, out_shape = dem.shape)

                # extrae las elevaciones dentro de la máscara
                elevaciones = dem.read(1)[mask]

                # si hay elevaciones (es decir, línea intersecta pseudodem)
                if len(elevaciones) > 0:
                    max_elev = elevaciones.max()
                    
                    # encuentra coord del punto de max densidad
                    max_elev_index  = elevaciones.argmax()
                    max_elev_coords = list(zip(*mask.nonzero()))[max_elev_index]
                    max_elev_point  = dem.xy(max_elev_coords[0], max_elev_coords[1])

                    # crea punto a partir de coordenadas de max densidad
                    punto_max_elev = Point(max_elev_point[0], max_elev_point[1])

                    # añade punto al listado de puntos de max densidad, junto con su FID
                    puntos_max_elevacion.append((idx, punto_max_elev))

        # crea GeoDF con los puntos de máxima densidad
        gdf_puntos_max_elevacion = gpd.GeoDataFrame(
            [(i, p) for i, p in puntos_max_elevacion], 
            columns = ['FID', 'geometry'],
            crs = gdf.crs
        )

        # ordenar puntos por FID
        gdf_puntos_max_elevacion = gdf_puntos_max_elevacion.sort_values('FID')

        # genera la línea que conecta los puntos ordenados
        linea_max_elevacion = LineString(gdf_puntos_max_elevacion.geometry.tolist())
        
        # Suaviza la línea generada
        linea_max_elevacion = suavizar_linea(linea_max_elevacion, grado_smooth)

        # crea GeoDF para la línea
        gdf_linea_max_elevacion = gpd.GeoDataFrame(geometry=[linea_max_elevacion], crs=gdf.crs)

        # transformación inversa de coordenadas 
        gdf_puntos_max_elevacion = gdf_puntos_max_elevacion.to_crs(epsg=4326)
        gdf_linea_max_elevacion  = gdf_linea_max_elevacion.to_crs(epsg=4326)

        # guarda puntos de max densidad en nuevo shapefile
        # gdf_puntos_max_elevacion.to_file(salida_puntos)

        # guarda línea de max densidad en nuevo shapefile
        gdf_linea_max_elevacion.to_file(salida_linea)

        print(f" Puntos guardados en : {salida_puntos}")
        print(f" Línea guardada en : {salida_linea}")
        
        return salida_linea

    # Ejecuta el proceso con iteraciones
    for i in range(iteraciones + 1):
        print(f"Proceso {i + 1} de {iteraciones + 1}")
        
        if i == 0:
            # Primera iteración con la línea costera base
            linea_costera_inicial = entrada
        else:
            # Usa la línea generada como nueva línea costera para la siguiente iteración
            linea_costera_inicial = salida_linea_max_elev

        salida_linea_max_elev = procesar_linea_costera(linea_costera_inicial, salida, salida_puntos_max_elev, salida_linea_max_elev)



    '''
    ###########################################################################

    # 7. CALCULA EL INDICE DE CONFIANZA (IC) DE UNA SERIE DE DATOS

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

    # 8. INCERTA CAMPOS DE PERÍODO E IC EN LOS ARCHIVOS VECTORIALES

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



if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python SDS.py <directorio_puntos>")
        sys.exit(1)
    
    ruta_directorio = sys.argv[1]
    procesar_directorio(ruta_directorio)