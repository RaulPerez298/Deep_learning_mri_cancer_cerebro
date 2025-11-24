import tensorflow as tf
import sys
import os
# # from tensorflow.keras.preprocessing.image import ImageGenerator
# from tensorflow.keras import optimizers
# # from tensorflow.keras import optimizer
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
# # from tensorflow.python.keras.layers import Convolution2D, MaxPoolig2D
# from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
# from tensorflow.python.keras import backend as k


import time
import numpy as np
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from skimage import io

from nuestromodelo_v8 import nuestro_v8

from modelo_resnet50 import resnet50
from modelo_resnet50_elu import resnet50_elu

from inception_resnet_v2 import inception_resnet_v2
import matplotlib.pyplot as plt
plt.close('all')
clases=2
def lector(file):
    imagen=resizeimage(file)
    long,_=imagen.shape
    new_image=np.zeros((long,long,3),dtype='int')
    new_image[:,:,0]=imagen
    new_image[:,:,1]=imagen
    new_image[:,:,2]=imagen
    return new_image


def recorte_marco_black(imagen):
    lim=50
    
    if len(imagen.shape)==3:
        New_corte=imagen[:,:,0]>lim
        recorte=imagen[:,:,0]
    else:
        New_corte=imagen>lim
        recorte=imagen
    
    vec_renglones=np.sum(New_corte,axis=1)
    u_renglones=np.where(vec_renglones!=0)
    # print(u_renglones[0][0])
    # print(u_renglones[0][-1])
    vec_columnas=np.sum(New_corte,axis=0)
    u_columnas=np.where(vec_columnas!=0)
    # print(u_columnas[0][0])
    # print(u_columnas[0][-1])
    recorte=recorte[u_renglones[0][0]:u_renglones[0][-1],u_columnas[0][0]:u_columnas[0][-1]]
    
    # plt.figure()
    # plt.imshow(recorte)
    
    return recorte
def resizeimage(name_file):

    # imagen=io.imread(name_file)
    mat=io.imread(name_file)
    imagen=recorte_marco_black(mat)
    # if len(imagen.shape)==3:
    #     fil,col,_=imagen.shape
    #     imagen=imagen[:,:,0]
    # else:
    fil,col=imagen.shape
    # print(fil,col)
    
    dif=int(abs(fil-col))
    residuo=dif % 2
    # print(residuo)
    if dif!=0:
        dif=dif-residuo
        # print(dif)
        dif=int(dif/2)
        if fil>col:
            for j in range(dif):
                # print(imagen.shape)
                imagen=np.concatenate((imagen,np.zeros((fil,1))),axis=1)
                imagen=np.concatenate((np.zeros((fil,1)),imagen),axis=1)
            if residuo==1:
                imagen=np.concatenate((np.zeros((fil,1)),imagen),axis=1)
        elif fil<col:
            for j in range(dif):
                imagen=np.concatenate((np.zeros((1,col)),imagen),axis=0)
                imagen=np.concatenate((imagen,np.zeros((1,col))),axis=0)
            if residuo==1:
                imagen=np.concatenate((np.zeros((1,col)),imagen),axis=0)
    
    return(imagen)




red_a_usar=2


##########     0                      1           2             3
names_red=['inception_resnet_v2','resnet50','nuestro_v8','resnet50_elu']
for i in range(len(names_red)):
    try:
        os.makedirs('./Figuras/'+names_red[i])
    except:
        print('Direcciones ya existentes')
    

name_r=names_red[red_a_usar]


if red_a_usar==0:
    altura, longitud = 299,299
    x,input_im=inception_resnet_v2(altura, longitud,3,clases)
elif red_a_usar==1:
    altura, longitud = 256, 256
    # print(12)
    x,input_im=resnet50(altura, longitud,3,output_red=clases)
elif red_a_usar==2:
    alpha_final=7
    altura, longitud = 224, 224
    x,input_im=nuestro_v8(altura, longitud,3,output_red=clases)
    epocas_cambio=11

elif red_a_usar==3:
    altura, longitud = 299,299
    # epocas_cambio=8
    alpha_inicio=2
    alpha_final=4
    epocas_cambio=3
    epocas = 5

    x,input_im=inception_resnet_v2(altura, longitud,3,clases)


    # print('aqui')
cnn = Model(inputs=input_im, outputs=x, name=name_r)



# cnn = load_model('modelo.h5')
# cnn = load_model('modelo_clase.h5')


def predict(file):
  x = load_img(file, target_size=(altura, longitud), color_mode = 'rgb')#carga la imagen con una resolucion definida
  x = img_to_array(x)#se convierte a array la imagen
  x/=255 #Se normaliza de 0 a 1 los tonos del pixel
  x = np.expand_dims(x, axis=0) 
  arreglo = cnn.predict(x)# se evalua la imagen en la red neuronal
  resultado = arreglo[0] #se extrae el vectorr de los resultados
  respuesta = np.argmax(resultado) #se obtiene la ubicacion con la probabilidad mas alta
  return respuesta


# pesos = 'el_uff_best_cnn.hdf5'



for num_validation in range(5):
    # print(str(num_validation+1)+' busca '+name_vec_clases[j]+' evaluaciones:',i+1,'/',tam)
    # cnn.load_weights('./pesos/'+name_r+'/val_accuracy_'+name_r+'_best_cnn'+str(num_validation+1)+'.h5')
    cnn.load_weights('./c_v_pesos_'+name_r+'/val_accuracy_'+name_r+'_best_cnn'+str(num_validation+1)+'.weights.h5')
    
    cantidad_clases=[]
    
    num_clases=clases
    
    if num_clases==2:
        name_vec_clases=['negativo','positivo']
        path_glo = ['./test/no','./test/yes'] #la ubicacion de las imagenes a evaluar
        
    elif num_clases==3:
        path_glo = ['./Test_enfermedades/Glioma','./Test_enfermedades/Meninglioma',
                    './Test_enfermedades/Pituitary'] #la ubicacion de las imagenes a evaluar
        name_vec_clases=['Glioma','Meninglioma','Pituitary']
    
    Matriz_confucion=np.zeros((num_clases,num_clases))
    Matriz_confucion_precision=np.zeros((num_clases,num_clases))
    
    tam=0
    tama_clases=[]
    tiempo=[]

    for j in range(num_clases): #la clase que se desee evaluar, 0 para no y 1 para si #en el caso de enfermedades es 0 'Glioma'  1'Meninglioma' 2'Pituitary' 
        print(str(num_validation+1)+' busca '+name_vec_clases[j])
        path =path_glo[j]
        path2 = path+'/*.jpg'
        lista=glob.glob(path2)
        tam=len(lista)
        cantidad_clases.append(tam)
        
        for i in range(tam):
            filename=lista[i]
            # print(str(num_validation+1)+' busca '+name_vec_clases[j]+' evaluaciones:',i+1,'/',tam)
            
            imag_save=lector(filename)
            
            io.imsave('imag_save.jpg',np.uint8(imag_save))
            # time.sleep(0.001)
            inicialT=time.time()
            x= predict('imag_save.jpg')#se obtiene el resultado de la clase mas probable
            finalT=time.time()
            tiempo.append(finalT-inicialT)
            Matriz_confucion[x,j]+=1
            Matriz_confucion_precision[x,j]+=1
        tama_clases.append(tam)
        Matriz_confucion_precision[:,j]/=tam
    tiempo=np.array(tiempo)
    sumtiempo=np.sum(tiempo)
    promtiempo=np.mean(tiempo)
    print('tiempo promedio: ',promtiempo)
    print('tiempo total: ',sumtiempo)

    plt.figure(figsize=(20, 5))
    # plt.colorbar(pad=0.05)
    plt.subplot(1,3,1)
    mat_text=np.array([['Verdadero Positivo: ','Falso Positivo: '],[' Falso Negativo: ','Verdadero Negativo: ']])
    
    # plt.yticks(np.arange(num_clases),[name_vec_clases[1],name_vec_clases[0]])
    plt.yticks(np.arange(num_clases),[' ',' '])
    plt.xticks(np.arange(num_clases),[name_vec_clases[1],name_vec_clases[0]])
    Matriz_confucion_precision=Matriz_confucion_precision[::-1,::-1]
    plt.imshow(Matriz_confucion_precision)
    plt.title('Grupo validacion '+str(num_validation+1)+' red: '+name_r+' presicion')
    
    for i_clase in range(num_clases):
        for j_clase in range(num_clases):
            plt.text(j_clase,i_clase,mat_text[i_clase,j_clase]+str(round(Matriz_confucion_precision[i_clase,j_clase],4)),color="white" 
                     if Matriz_confucion_precision[i_clase,j_clase] < 0.5 else "black",horizontalalignment ='center')

    plt.subplot(1,3,2)
    
    # plt.yticks(np.arange(num_clases),[name_vec_clases[1],name_vec_clases[0]])
    plt.yticks(np.arange(num_clases),[' ',' '])
    plt.xticks(np.arange(num_clases),[name_vec_clases[1],name_vec_clases[0]])
    Matriz_confucion=Matriz_confucion[::-1,::-1]
    plt.imshow(Matriz_confucion)
    plt.title('Grupo validacion '+str(num_validation+1)+' red: '+name_r)
    
    for i_clase in range(num_clases):
        for j_clase in range(num_clases):
            plt.text(j_clase,i_clase,mat_text[i_clase,j_clase]+str(Matriz_confucion[i_clase,j_clase]),color="white" 
                     if Matriz_confucion[i_clase,j_clase] < tama_clases[j_clase]*0.5 else "black",horizontalalignment ='center')
    
    plt.subplot(1,3,3)
    plt.imshow(np.ones((3,1)),vmax=1,vmin=0,cmap='gray')
    # 'Sensitivity: '+str(round(Matriz_confucion[0,0]/(Matriz_confucion[0,0]+Matriz_confucion[1,0]),4))
    # 'Specificity: '+str(round(Matriz_confucion[1,1]/(Matriz_confucion[0,1]+Matriz_confucion[1,1]),4))
    # 'Accuracy: '+str(round((Matriz_confucion[0,0]+Matriz_confucion[1,1])/(np.sum(Matriz_confucion)),4))
    plt.text(0,0,'Sensitivity: '+str(round(Matriz_confucion[0,0]/(Matriz_confucion[0,0]+Matriz_confucion[1,0]),4)),horizontalalignment ='center')
    plt.text(0,1,'Specificity: '+str(round(Matriz_confucion[1,1]/(Matriz_confucion[0,1]+Matriz_confucion[1,1]),4)),horizontalalignment ='center')
    
    plt.text(0,2,'Accuracy: '+str(round((Matriz_confucion[0,0]+Matriz_confucion[1,1])/(np.sum(Matriz_confucion)),4)),horizontalalignment ='center')
    plt.xticks(np.arange(1),[' '])
    plt.yticks(np.arange(3),[' ',' ',' '])
    os.makedirs('./Figuras/'+names_red[red_a_usar]+'/', exist_ok=True)  
    plt.savefig('./Figuras/'+names_red[red_a_usar]+'/Figure_grupo_'+str(num_validation+1)+'.jpg')
    ima_new_fig=io.imread('./Figuras/'+names_red[red_a_usar]+'/Figure_grupo_'+str(num_validation+1)+'.jpg')
    ima_new_fig=np.concatenate((ima_new_fig[:,240:710],ima_new_fig[:,815:1240],ima_new_fig[:,1500:1650]),axis=1)
    io.imsave('./Figuras/'+names_red[red_a_usar]+'/Figure_grupo_'+str(num_validation+1)+'.jpg',ima_new_fig)
    
plt.show()
