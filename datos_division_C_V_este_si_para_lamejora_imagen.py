

from skimage import io,color
import numpy as np
import glob
from sklearn.model_selection import KFold
import os
try:
    for i in range(5):
        os.makedirs('./mejora_imagen/data_c_v/interaccion'+str(i+1)+'/train/no', exist_ok=True)
        os.makedirs('./mejora_imagen/data_c_v/interaccion'+str(i+1)+'/train/yes', exist_ok=True)

        os.makedirs('./mejora_imagen/data_c_v/interaccion'+str(i+1)+'/test/no', exist_ok=True)
        os.makedirs('./mejora_imagen/data_c_v/interaccion'+str(i+1)+'/test/yes', exist_ok=True)

except:
    print('Ya existe el directorio')

numeros_grupos=5

kf = KFold(n_splits=numeros_grupos,shuffle=True)
ruta='./data_mejora/'
ruta_yes=ruta+'yes/*'
ruta_no=ruta+'no/*'
directorio_guardar='./mejora_imagen/data_c_v/interaccion'

files_yes=glob.glob(ruta_yes)
files_no=glob.glob(ruta_no)



# index_yes=files_yes[0].index('s/')+2
# index_no=files_no[0].index('o/')+2
interacion=1
for train_index, test_index in kf.split(np.zeros(len(files_yes))):
    dir_g=directorio_guardar+str(interacion)
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    imagen=1
    for j in range(len(train_index)):
        
        i=train_index[j]
        
        io.imsave(dir_g+'/train/no/'+str(imagen)+'.jpg',io.imread(files_no[i]))
        io.imsave(dir_g+'/train/yes/'+str(imagen)+'.jpg',io.imread(files_yes[i]))
        imagen+=1
    print('cargado datos de entrenamiento')
    
    for j in range(len(test_index)):
        
        i=test_index[j]
        # test_set_data[j,:,:,:]=imagen_entrenamiento[i][0]
        # label_test[j,:]=imagen_entrenamiento[i][1]
        io.imsave(dir_g+'/test/no/'+str(imagen)+'.jpg',io.imread(files_no[i]))
        io.imsave(dir_g+'/test/yes/'+str(imagen)+'.jpg',io.imread(files_yes[i]))
        imagen+=1
    print('cargado datos de testeo')
    interacion+=1
