#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
###############################################################################

EJECUTA SDS.py EN LAS RUTAS DE errores.txt 

###############################################################################

'''

import os
import subprocess
from tqdm import tqdm

def extraer_rutas(archivo_errores):
    # Extrae rutas del archivo errores.txt
    rutas = []
    inicio = "[ERROR] ppss-2_clus.py - "
    fin = ". COD -9"

    with open(archivo_errores, "r") as archivo:
        for linea in archivo:
            if inicio in linea and fin in linea:
                # Extraer la ruta usando el inicio y el fin especificados
                ruta = linea.split(inicio)[1].split(fin)[0].strip()
                rutas.append(ruta)
    return rutas

def ejecutar_sds_en_rutas(rutas):
    # Obtener la ruta absoluta de SDS.py
    ruta_sds = os.path.join(os.path.dirname(__file__), "SDS.py")
    
    # Verificar si el archivo SDS.py existe en el directorio 
    if not os.path.exists(ruta_sds):
        print(f"Error: No se encontró el archivo SDS.py en {ruta_sds}. Intentando buscar en la ruta absoluta.")
        ruta_sds = os.path.join(os.path.dirname(__file__), "SDS.py")
        
        if not os.path.exists(ruta_sds):
            print(f"Error: No se encontró SDS.py en la ruta especificada: {ruta_sds}")
            exit(1)

    # Ejecuta el script SDS.py en cada ruta extraída
    for ruta in tqdm(rutas, desc='>>', unit="ruta"):
        print(f"en:{ruta}")
        
        # Ejecutar el script SDS.py en la ruta correspondiente
        resultado = subprocess.run(["python3", ruta_sds, ruta], capture_output=True, text=True)
        
        # Mostrar el resultado de la ejecución
        if resultado.returncode == 0:
            print(f"*** procesado ***")
        else:
            print(f"Error al procesar {ruta}: {resultado.stderr}")

if __name__ == "__main__":
    # Obtener la ruta completa de errores.txt
    archivo_errores = os.path.join(os.path.dirname(__file__), "errores.txt")

    if not os.path.exists(archivo_errores):
        print(f"Error: No se encontró el archivo {archivo_errores}")
        exit(1)

    rutas_a_procesar = extraer_rutas(archivo_errores)
    if not rutas_a_procesar:
        print("No se encontraron rutas en el archivo de errores.")
        exit(1)

    # Ejecutar el proceso en las rutas extraídas
    ejecutar_sds_en_rutas(rutas_a_procesar)






