# Deep_learning_mri_cancer_cerebro

## Instalación del proyecto.

### Instalación de CUDA.

El proyecto trata de la implementación de redes neuronales para la detección de cancer en el cerebro utilizando **MRI** la información de las especificaciones del proyecto y como fue entrenado se encuentran en [Deep_proyecto.pdf](Deep_proyecto.pdf), para poder hacer el entrenamiento de la red neuronal se recomienda usar una gpu RTX,  el archivo [instal_cuda.sh](instal_cuda.sh) contiene las instrucciones necesarias para utilizar dicha GPU en UBUNTU 2404. Donde los comandos a usar en termina son los siguientes:

```
sudo apt install git
git clone https://github.com/RaulPerez298/Deep_learning_mri_cancer_cerebro.git
cd Deep_learning_mri_cancer_cerebro
bash ./instal_cuda.sh
```

### Instalación de Python.

Se deberá utilizar Conda para poder utilizar Python 3.10, donde la instalación esta definida en las siguientes instrucciones:

```
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash ./Anaconda3-2024.10-1-Linux-x86_64.sh 
```

Para utilizar python 3.10 se utiliza la siguiente instrucción para instalar todas las dependencias necesarias:

```
bash ./datos_crear_python.sh
```

Al ejecutar el comando anterior se crea un nuevo entorno virtual el cual se tiene que activar usando el siguiente comando:

```
conda activate tfcpu
```

## Procesamiento de imágenes. 

La red neuronal fue entrada para 2 clases, si tiene o no la enfermedad, donde cada clase contiene 3418 imágenes en *Data set* y *test* contiene 180 imágenes por clase para testear el modelo (estas imágenes nunca son usadas para entrenar ni para validar los datos usando *"val_accuracy"* ). Si deseas agregar más imágenes para entrenar y testear deberás colocarlas en las direcciones siguientes (esto se debe que las imágenes se tienen que procesar para los modelos que usan el método **gamma**):

```text
Deep_learning_mri_cancer_cerebro/
├── data set/
│   ├── no/
│   └── yes
└── test/
    ├── no/
    └── yes/
```



[agc_function.py](agc_function.py)



```
python newdatasetGamma.py
```







```text
Deep_learning_mri_cancer_cerebro/
└── data_mejora/
   ├── no/
   └── yes
```







```
spyder datos_division_C_V_este_si_cx.py
```





```text
Deep_learning_mri_cancer_cerebro/
└── data_c_v/
   ├── interaccion1/
   │	├──test/
   │	│	├──no/
   │	│	└──yes/
   │	└──train/
   │		├──no/
   │		└──yes/
   ├── interaccion2/
   │	├──test/
   │	│	├──no/
   │	│	└──yes/
   │	└──train/
   │		├──no/
   │		└──yes/
   ├── interaccion3/
   │	├──test/
   │	│	├──no/
   │	│	└──yes/
   │	└──train/
   │		├──no/
   │		└──yes/
   ├── interaccion4/
   │	├──test/
   │	│	├──no/
   │	│	└──yes/
   │	└──train/
   │		├──no/
   │		└──yes/
   └── interaccion5/
   		├──test/
   		│	├──no/
   		│	└──yes/
   		└──train/
   			├──no/
   			└──yes/

```





```
spyder datos_division_C_V_este_si_para_lamejora_imagen.py
```





```text
Deep_learning_mri_cancer_cerebro/mejora_imagen/
└── data_c_v/
   ├── interaccion1/
   │	├──test/
   │	│	├──no/
   │	│	└──yes/
   │	└──train/
   │		├──no/
   │		└──yes/
   ├── interaccion2/
   │	├──test/
   │	│	├──no/
   │	│	└──yes/
   │	└──train/
   │		├──no/
   │		└──yes/
   ├── interaccion3/
   │	├──test/
   │	│	├──no/
   │	│	└──yes/
   │	└──train/
   │		├──no/
   │		└──yes/
   ├── interaccion4/
   │	├──test/
   │	│	├──no/
   │	│	└──yes/
   │	└──train/
   │		├──no/
   │		└──yes/
   └── interaccion5/
   		├──test/
   		│	├──no/
   		│	└──yes/
   		└──train/
   			├──no/
   			└──yes/

```







En [red_c_v_todas.py](red_c_v_todas.py) se deberá de modificar la variable "*red_a_entrenar*", la cual puede tomar valores de 0 hasta 3 dependiendo del modelo que se desea entrenar, donde la lista de los modelos esta definidos como *['inception_resnet_v2','resnet50','nuestro_v8','resnet50_elu']* .



```
python red_c_v_todas.py
```



```
python red_c_v_todas_gamma.py
```



```
spyder red_validacion_impro.py
```



```
cd mejora_imagen/
spyder red_validacion_impro.py
```



```
python kivy_proyecto.py
```

