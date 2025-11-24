from skimage import io,color
import numpy as np

def recorte_marco_black(imagen):
    lim=60
    
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
    imagen=recorte_marco_black(io.imread(name_file))
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
    imagen=np.array(imagen,dtype='uint8')
    return(imagen)
