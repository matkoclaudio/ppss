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
7. MAIN DE POSPROCESOS

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

# 7. MAIN DE POSPROCESOS

###############################################################################

#%%
def ejecutar_posproceso(carpeta):
    # Obtiene el directorio donde se encuentra el archivo main.py
    directorio_scripts = os.path.dirname(os.path.abspath(__file__))
    
    # Ruta del archivo de errores
    archivo_errores = os.path.join(directorio_scripts, "errores.txt")
    
    # scripts de posproceso
    nombres_scripts = [
        'ppss-1_dist.py',
        'ppss-2_clus.py',
        'ppss-3_krnl.py',
        'ppss-4_dgit.py',
        'ppss-5_indx.py',
    ]
    
    # Inicia la barra de progreso para los scripts
    with tqdm(total=len(nombres_scripts), desc=f'{os.path.basename(carpeta)}', ncols=80, position=0, leave=True) as barra:
        for nombre_script in nombres_scripts:
            ruta_script = os.path.join(directorio_scripts, nombre_script)
            
            # Ejecuta el script actual y captura errores
            comando = ['python', ruta_script, carpeta]
            resultado = subprocess.run(comando)
            
            if resultado.returncode != 0:
                error_msg = f"[ERROR] {nombre_script} - {carpeta}. COD {resultado.returncode}\n"
                print(f"\n{error_msg}")
                
                # Guarda el error en el archivo de errores
                with open(archivo_errores, 'a') as f:
                    f.write(error_msg)
                
                break
            
            # Incrementa la barra de progreso
            barra.update(1)
            barra.set_postfix({"...": nombre_script})

def procesar_directorio(directorio_base):
    # Obtiene la lista de subdirectorios en el directorio base
    subdirectorios = [os.path.join(directorio_base, d) for d in os.listdir(directorio_base) if os.path.isdir(os.path.join(directorio_base, d))]
    
    # Procesa cada subdirectorio en paralelo usando ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(ejecutar_posproceso, subdirectorios), total=len(subdirectorios), desc=f"Progreso en {os.path.basename(directorio_base)}", ncols=80))

def main():
    # Define directorio base al directorio donde se encuentra main.py
    directorio_base = os.path.dirname(os.path.abspath(__file__))
    
    # periodos a procesar
    anos = ['1986-1991',
            '1992-1997',
            '1998-2003',
            '2004-2009',
            '2010-2015',
            '2016',
            '2017',
            '2018',
            '2019',
            '2020',
            '2021',
            '2022',
            '2023']
    
    # Genera la lista de rutas de directorios base para cada año
    directorios_anos = [os.path.join(directorio_base, ano) for ano in anos]
    
    # Procesa cada directorio de año en cadena
    for directorio_ano in directorios_anos:
        if os.path.isdir(directorio_ano):  # Verifica si el directorio del año existe
            print(f"\nprocesando : {directorio_ano}...")
            procesar_directorio(directorio_ano)
        else:
            print(f"\n[ADVERTENCIA] El directorio para el año {directorio_ano} no existe y será omitido.")

if __name__ == "__main__":
    main()


