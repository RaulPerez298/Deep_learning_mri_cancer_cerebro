
# from numba import cuda
# device = cuda.get_current_device()
# device.reset()

# import cv2
import matplotlib.pyplot as plt

# from google.colab.patches import cv2_imshow
# import pandas as pd
import glob
import os

# from sklearn.model_selection import KFold
import tensorflow as tf
import sys
import os
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as k

from inception_resnet_v2 import inception_resnet_v2
from tensorflow.keras.callbacks import ModelCheckpoint

from nuestromodelo_v8 import nuestro_v8

from modelo_resnet50 import resnet50

from modelo_resnet50_elu import resnet50_elu

from inception_resnet_v2 import inception_resnet_v2
# datos_entrenamiento = './train'
# datos_validacion = './validation'


    
# path_base='./'

###Aqui se puede colocar un for para la variables red_a_entrenar para asi hacer todos lso modelos a calcular
red_a_entrenar=0

# for red_a_entrenar in range(1,4):
##########     0                      1           2             3
names_red=['inception_resnet_v2','resnet50','nuestro_v8','resnet50_elu']
name_r=names_red[red_a_entrenar]
numeros_grupos=5
# interacion=1
try:
    matriz_evaluate=np.load('./c_v_pesos_'+name_r+'/matriz_c_v.npy')
    matriz_evaluate_train=np.load('./c_v_pesos_'+name_r+'/matriz_c_v_train.npy')
    matriz_evaluate_test=np.load('./c_v_pesos_'+name_r+'/matriz_c_v_test.npy')

except:
    matriz_evaluate=np.zeros((numeros_grupos,2))
    matriz_evaluate_train=np.zeros((numeros_grupos,2))
    matriz_evaluate_test=np.zeros((numeros_grupos,2))

os.makedirs("./c_v_pesos_"+name_r+"/", exist_ok=True)  
for grupo in range(numeros_grupos):

    print('entrenando modelo '+name_r)
    interacion=grupo+1
    datos_entrenamiento = './data_c_v/interaccion'+str(grupo+1)+'/train'
    datos_validacion='./data_c_v/interaccion'+str(grupo+1)+'/test'
    datos_test='./test'
    
    entrenamiento_datagen = ImageDataGenerator(rescale=(1.0/255))
    validation_datagen= ImageDataGenerator(rescale=(1.0/255))
    test_datagen= ImageDataGenerator(rescale=(1.0/255))
    # batch_size = 10
    
    batch_size=32
    
    clases = 2
    # tf.keras.backend.clear_session()
    
    
    
    
    alpha_final=7
    epocas_cambio=15
    alpha_inicio=4
    batch_fit=batch_size
    epocas = 10

    if red_a_entrenar==0:
        altura, longitud = 299,299
        
        alpha_final=6
        tf.keras.backend.clear_session()
        x,input_im=inception_resnet_v2(altura, longitud,3,clases)
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(True) # Enable XLA.
    elif red_a_entrenar==1:
        altura, longitud = 256, 256
        # print(12)
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(True) # Enable XLA.

        x,input_im=resnet50(altura, longitud,3,output_red=clases)
        

    elif red_a_entrenar==2:
        alpha_final=7
        altura, longitud = 224, 224
        x,input_im=nuestro_v8(altura, longitud,3,output_red=clases)
        epocas_cambio=11

    elif red_a_entrenar==3:
        altura, longitud = 256, 256
        # print(12)
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(True) # Enable XLA.

        x,input_im=resnet50_elu(altura, longitud,3,output_red=clases)
        
    # define the model 
    
    cnn = Model(inputs=input_im, outputs=x, name=name_r)
    
    
    
    # cnn.summary()
    # cnn.load_weights('resnet_best_cnn.h5')


    imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
                                        datos_entrenamiento,
                                        target_size = (altura, longitud),
                                        batch_size =batch_size,
                                        color_mode = 'rgb',
                                        class_mode = 'categorical',
                                        shuffle = True )
    
    imagen_validacion = validation_datagen.flow_from_directory(
                                            datos_validacion,
                                            target_size = (altura, longitud),
                                            batch_size = batch_size,
                                            color_mode = 'rgb',
                                            class_mode = 'categorical',
                                            shuffle = True )
    
    imagen_test = test_datagen.flow_from_directory(
                                            datos_test,
                                            target_size = (altura, longitud),
                                            batch_size = batch_size,
                                            color_mode = 'rgb',
                                            class_mode = 'categorical',
                                            shuffle = True )


    # print(len(imagen_validacion)+len(imagen_entrenamiento))
    # imagen_entrenamiento.class_indices #retorna las clases
    
    
    # print("TRAIN:", len(train_index), "TEST:", test_index)
    
    
    
    #Parametros de la arquitectura neuronal
    
    
    
    
    
    
    
    
    
    
    # checkpointer = ModelCheckpoint(filepath=os.path.join("./c_v_pesos_resnet50/val_loss_inception_resnet_v2_best_cnn"+str(interacion)+".h5"),
    #                               # monitor='val_accuracy',
    #                               verbose=1,
    #                               save_best_only=True,
    #                               save_weights_only=True,
    #                               mode='min')
    checkpointer_val = ModelCheckpoint(filepath=os.path.join("./c_v_pesos_"+name_r+"/val_accuracy_"+name_r+"_best_cnn"+str(interacion)+".weights.h5"),
                                  monitor='val_accuracy',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=True,
                                  mode='max')
    
  
    
    cnn.save('./c_v_pesos_'+name_r+'/modelo_'+name_r+'.weights.h5')
    # cnn.load_weights("./c_v_pesos_"+name_r+"/val_accuracy_"+name_r+"_best_cnn"+str(interacion)+".h5")
    
    for i in range(alpha_inicio,alpha_final):
            print(interacion,i)
            alpha_lr = 10**-i
            cnn.compile(loss='categorical_crossentropy', 
                    optimizer = optimizers.Adam(learning_rate=alpha_lr),
                    metrics = ['accuracy'])
    
            history=cnn.fit(imagen_entrenamiento, 
                    batch_size=batch_fit,
                    epochs = epocas, 
                    validation_data = imagen_validacion, 
                    verbose=1,
                    validation_freq=1,
                    # callbacks=[checkpointer,checkpointer_val])
                    callbacks=[checkpointer_val])
            
            his_accuracy=history.history['accuracy']
            his_val_accuracy=history.history['val_accuracy']
            his_loss=history.history['loss']
            his_val_loss=history.history['val_loss']
            
            os.makedirs('./c_v_history_'+name_r+'/', exist_ok=True)

            np.save('./c_v_history_'+name_r+'/c_v'+str(interacion)+'his_accuracy_'+str(i)+'.npy', his_accuracy)
            np.save('./c_v_history_'+name_r+'/c_v'+str(interacion)+'his_val_accuracy_'+str(i)+'.npy',his_val_accuracy)
            np.save('./c_v_history_'+name_r+'/c_v'+str(interacion)+'his_loss_'+str(i)+'.npy',his_loss)
            np.save('./c_v_history_'+name_r+'/c_v'+str(interacion)+'his_val_loss_'+str(i)+'.npy',his_val_loss)
            

            
            cnn.load_weights("./c_v_pesos_"+name_r+"/val_accuracy_"+name_r+"_best_cnn"+str(interacion)+".weights.h5")
            epocas =epocas_cambio
            if red_a_entrenar==0:
                time.sleep(4)
            
            # if i!=6:
            #     cnn.load_weights("./c_v_pesos_inception_resnet_v2/val_loss_inception_resnet_v2_best_cnn"+str(interacion)+".h5")

    results = cnn.evaluate(imagen_validacion)
    matriz_evaluate[interacion-1,:]=results
    # cnn.save_weights('./c_v_pesos_'+name_r+'/c_v'+str(interacion)+'best_val_accuracy_'+name_r+'_pesos.h5')
    
    print('espera 1')
    if red_a_entrenar==3:
        time.sleep(5)

    results_train = cnn.evaluate(imagen_entrenamiento)
    matriz_evaluate_train[interacion-1,:]=results_train
    print('espera 2')
    if red_a_entrenar==3:
        time.sleep(5)

    results_test = cnn.evaluate(imagen_test)
    matriz_evaluate_test[interacion-1,:]=results_test
    print('espera 3')
    if red_a_entrenar==3:
        time.sleep(5)


    np.save('./c_v_pesos_'+name_r+'/matriz_c_v.npy', matriz_evaluate)
    np.save('./c_v_pesos_'+name_r+'/matriz_c_v_train.npy', matriz_evaluate_train)
    np.save('./c_v_pesos_'+name_r+'/matriz_c_v_test.npy', matriz_evaluate_test)

    interacion+=1
# cnn.save_weights('./c_v_pesos_'+name_r+'_pesos.h5')
np.save('./c_v_pesos_'+name_r+'/matriz_c_v.npy', matriz_evaluate)
np.save('./c_v_pesos_'+name_r+'/matriz_c_v_train.npy', matriz_evaluate_train)
np.save('./c_v_pesos_'+name_r+'/matriz_c_v_test.npy', matriz_evaluate_test)




















# time.sleep(15)
# print('Se apagara el pc en dos minutos')
# os.system("shutdown +2")

