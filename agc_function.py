
import numpy as np 
from skimage import data,color,io

import time
import glob
import cv2 as cv
# def his(imh):
#     hist=np.zeros(256)
#     p=0
#     ecu=np.zeros(256)
#     for i in range(256):
#         suma=np.sum(imh==i)
#         hist[i]=suma
#         p+=suma
#         ecu[i]=p
# #     return(hist/np.sum(hist),np.max(imh)*ecu/np.sum(hist))
#     return(hist/np.sum(hist))
def his(imh):
    hist=np.zeros(256)
    # p=0
    # ecu=np.zeros(256)
    for i in range(np.min(imh),np.max(imh)+1):
        suma=np.sum(imh==i)
        hist[i]=suma
        # p+=suma
        # ecu[i]=p
    return(hist/np.sum(hist))
def his(imh):
    hist=np.zeros(256)
    # p=0
    # ecu=np.zeros(256)
    for i in range(imh.shape[0]):
        for j in range(imh.shape[1]):
            hist[imh[i,j]]+=1
            # p+=suma
            # ecu[i]=p
    return(hist/np.sum(hist))

def cdf(vec):
    vecnew=np.zeros(256)
    proba=0
    # print(vec)
    vef=np.where(vec>0)[0]
    ini=vef[0]
    final=vef[-1]
    # print(ini)
    for i in range(ini,final+1):
        proba+=vec[i]
        vecnew[i]=(proba)
    vecnew[final:-1]=1
    # print(vecnew)
    return(vecnew)

def agc(im,porcnta):
    
    # porcnta=0.03
    ren=int(im.shape[0]*porcnta)
    col=int(im.shape[1]*porcnta)
    i=0
    j=0
    # while(im.shape[0]%(ren+i)!=0):
    #     i+=1
    # while(im.shape[1]%(col+j)!=0):
    #     j+=1
    # print(im.shape[0]/(ren+i)+1)
    # div_ren=np.linspace(0,im.shape[0],int(im.shape[0]/(ren+i)+1),dtype='int')
    # div_col=np.linspace(0,im.shape[1],int(im.shape[1]/(col+i)+1),dtype='int')
    div_ren=np.linspace(0,im.shape[0],porcnta,dtype='int')
    div_col=np.linspace(0,im.shape[1],porcnta,dtype='int')
    New_image_clahe=np.zeros((im.shape),dtype='uint8')
    # print(New_image_clahe)
    # plt.figure()
    
    # veu=np.linspace(0,255,256)
    veu=np.linspace(0,255,256,dtype='uint8')
    for i in range(div_ren.shape[0]-1):
        for j in range(div_col.shape[0]-1):
            
            # print(div_ren[i])
            # print(div_ren[i+1])
            # print(div_col[i])
            # print(div_col[i+1])
            
            iniren=div_ren[i]
            finren=div_ren[i+1]
            
            inicol=div_col[j]
            fincol=div_col[j+1]
            
            ipru=im[iniren:finren,inicol:fincol]
            maxnueva=np.max(ipru)

            
            pdf=his(ipru)

            maxpdf=np.max(pdf)
            minpdf=np.min(pdf)
            pdfw=maxpdf*(pdf-minpdf)/(maxpdf-minpdf)

            # pdfw=np.max(pdf)*(pdf-np.min(pdf))/(np.max(pdf)-np.min(pdf))
            
            # sumpdfw=np.sum(pdfw)
            # t=time.time()
            cdfw=cdf(pdfw)
            # print(time.time()-t)
            # print(cdfw.shape)
            # filnew,colnew=ipru.shape


            if maxnueva==0:
               maxnueva=0.00001
            
            # inicio = time.time()
            
            proba=maxnueva*(veu/maxnueva)**((1+cdfw[veu])/2)
            
            
            New_image_clahe[iniren:finren,inicol:fincol]=proba[ipru]
            # print(time.time()-inicio)
    # plt.plot(h)
    return New_image_clahe
        
    
    
    
def prom(mat):
    num=7
    imagen=np.zeros((mat.shape[0],mat.shape[1],num),dtype='uint8')
    
    for i in range(num):
        por=i+30
        
        imagen[:,:,i]=agc(mat,por)
        # plt.figure()
        # plt.imshow(imagen[:,:,i],cmap='gray')
    
    return np.mean(imagen,axis=(2))

    # print(fin-inicio)