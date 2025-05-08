#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
###############################################################################

- DIGITALIZACIÓN AUTOMÁTICA 

Genera la linea costera correspondiente al año :

1. Traza perpendiculares a la linea de costa base
2. Maximas densidades = maximas elevaciones de un pseudodem (krnl)
3. Halla los puntos de max densidad entre las perpendiculares y el pseudodem
4. Traza una línea uniendo los puntos de max densidad
6. es posible iterar, sobre la línea resultante. 

-
pip install geopandas 
pip install shapely 
pip install haversine 
pip install rich 
pip install fiona

###############################################################################

'''

import os
import cv2
import sys

import glob
import math
import rasterio
import geopandas as gpd

from shapely.geometry import LineString, Point
from rasterio.features import geometry_mask
from rich.progress import track


'''
###############################################################################

- DIGITALIZACIÓN AUTOMÁTICA 

###############################################################################
'''

def procesar_datos(directorio_puntos):

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