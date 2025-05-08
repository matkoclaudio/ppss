#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
###############################################################################

DELIMITACIÓN : 

1. KERNEL-DENSITY 
2. OTSU-BINARIZACIÓN 
3. VECTORIZA

###############################################################################

'''

import os
import sys
import cv2

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
import rasterio
import pyproj

import colorcet as cc
import holoviews as hv
import datashader as ds

from holoviews.element.tiles import EsriImagery
from holoviews.operation.datashader import datashade

from rasterio.transform import from_origin
from rasterio.warp import transform_geom
from rasterio.enums import Resampling
from rasterio.features import shapes

from shapely.geometry import Polygon
from bokeh.io import output_file, save
from scipy.ndimage import gaussian_filter

###############################################################################

# 1. KERNEL-DENSITY new

###############################################################################

def procesar_datos(directorio_puntos):

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
    
    # 2. OTSU-BINARIZACIÓN 
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
    
    # 3. VECTORIZA
    
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