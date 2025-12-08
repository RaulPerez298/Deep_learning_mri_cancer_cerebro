
from skimage import io,color
import numpy as np
import glob
from sklearn.model_selection import KFold
import os

numeros_grupos=5

kf = KFold(n_splits=numeros_grupos,shuffle=True)
ruta='./data set/'
ruta_yes=ruta+'yes/*'
ruta_no=ruta+'no/*'
directorio_guardar='./data_c_v/interaccion'

files_yes=glob.glob(ruta_yes)
files_no=glob.glob(ruta_no)



# index_yes=files_yes[0].index('s/')+2
# index_no=files_no[0].index('o/')+2
interacion=1
for train_index, test_index in kf.split(np.zeros(len(files_yes))):
    dir_g=directorio_guardar+str(interacion)
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    imagen=1
    if not os.path.exists(dir_g+'/train/no/'):
        os.makedirs(dir_g+'/train/no/', exist_ok=True)
    if not os.path.exists(dir_g+'/train/yes/'):
        os.makedirs(dir_g+'/train/yes/', exist_ok=True)
        
    if not os.path.exists(dir_g+'/test/no/'):
        os.makedirs(dir_g+'/test/no/', exist_ok=True)
    if not os.path.exists(dir_g+'/test/yes/'):
        os.makedirs(dir_g+'/test/yes/', exist_ok=True)
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
