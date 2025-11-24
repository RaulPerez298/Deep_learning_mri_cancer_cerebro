#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 22:23:23 2022

@author: ral298
"""

from skimage import io,color
import numpy as np
import glob
from agc_function import prom
import cv2 as cv
# import matplotlib.pyplot as plt
import random
import multiprocessing
import os
from tam_resize import resizeimage

try:
    os.makedirs('./mejora_imagen/test/no')
    os.makedirs('./mejora_imagen/test/yes')
    os.makedirs('./data_mejora/yes')
    os.makedirs('./data_mejora/no')
except:
    print('Ya existen')
test=1
if test==0:
    ruta='./data set/'
    ruta_mejora='./data_mejora/'
else:
    ruta='./test/'
    ruta_mejora='./mejora_imagen/test/'

ruta_yes=ruta+'yes/*'

files_yes=glob.glob(ruta_yes)

indice_yes=files_yes[0].index('yes/')

ruta_no=ruta+'no/*'
files_no=glob.glob(ruta_no)
indice_no=files_no[0].index('no/')        


def yes(i):
    
    
    print(i+1,'/',len(files_yes))
    
    img = resizeimage(files_yes[i])
    # img = io.imread(files_yes[i])[:,:,0]
    # print(img.dtype)
    imagen=np.copy(img)
    imagen_clahe=np.copy(img)
    final_image=np.zeros((img.shape[0],img.shape[1],3),dtype='uint8')
    
    # create a CLAHE object (Arguments are optional).
    newclahe=prom(imagen)
    final_image[:,:,0]=newclahe
    final_image[:,:,1] =newclahe 
    
    final_image[:,:,2]=newclahe
    
    new_name=ruta_mejora+files_yes[i][indice_yes::]
    io.imsave(new_name,final_image)

def no(i):
    print(i+1,'/',len(files_no))
    img = resizeimage(files_no[i])
    
    # img = io.imread(files_no[i])[:,:,0]
    imagen=np.copy(img)
    imagen_clahe=np.copy(img)
    final_image=np.zeros((img.shape[0],img.shape[1],3),dtype='uint8')
    
    # create a CLAHE object (Arguments are optional).
    
    newclahe=prom(imagen)
    final_image[:,:,0]=newclahe
    final_image[:,:,1] =newclahe 
    
    final_image[:,:,2]=newclahe
    new_name=ruta_mejora+files_no[i][indice_no::]
    io.imsave(new_name,final_image)

if __name__ == "__main__":
    print("cpu:",multiprocessing.cpu_count())
    multiprocessing.set_start_method("spawn", force=True)

    print("Procesando imágenes de YES...")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(yes, range(len(files_yes)))

    print("Procesando imágenes de NO...")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(no, range(len(files_no)))

    print("✅ Procesamiento completado.")
