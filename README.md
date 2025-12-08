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



Se deberá de crear un nuevo dataset con el ya existente, esto se debe a que se tiene que aplicar una mejora de imágenes usando la funcion *prom* del archivo [agc_function.py](agc_function.py), donde se aplica el método gamma propuesto, donde el código siguiente es el utilizado para crear el nuevo data set:

```
python newdatasetGamma.py
```

Al ser ejecutado dicho comando ahora tendremos estos directorios, donde se encuentran las imágenes ya procesadas usando el método propuesto basado en gamma. 



```text
Deep_learning_mri_cancer_cerebro/
└── data_mejora/
│  ├── no/
│  └── yes
└── mejora_imagen/test/
   ├── no/
   └── yes
```



### Utilización de validación cruzada.

Este metodo es utilizado como tal para evaluar el modelo para hacer una tarea especifica, para ello el data set se divide en 5, donde por cada interacción el subdata set seleccionado sera usado para calcular *"val_accuracy"*, y los 4 restantes para hacer el entrenamiento del modelo, teniendo un total de 5 pesos sinapticos, para crear el data set es con la siguiente instrucción (se deberá de darle run con el software de spyder *F5*):

```
spyder datos_division_C_V_este_si_cx.py
```

Donde ahora los datos nuevos son los siguientes:

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



Ahora se deberá aplicar la creación de los nuevos datos de validación cruzada, pero para las imágenes con un pre-procesamiento (se deberá de darle run con el software de spyder *F5*): 

```
spyder datos_division_C_V_este_si_para_lamejora_imagen.py
```

Siendo ahora este árbol de direcciones creados:

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



## Entrenamiento de las redes neuronales.



En [red_c_v_todas.py](red_c_v_todas.py) se deberá de modificar la variable "*red_a_entrenar*", la cual puede tomar valores de 0 hasta 3 dependiendo del modelo que se desea entrenar, donde la lista de los modelos esta definidos como *['inception_resnet_v2','resnet50','nuestro_v8','resnet50_elu']* . Donde se entrenara 5 veces el modelo seleccionado, para entrenarlo con los datos sin procesar es con el siguiente comando:

```
python red_c_v_todas.py
```

Para entrenar los modelos con los datos ya procesados es con el siguiente comando:

```
python red_c_v_todas_gamma.py
```



(se deberá de darle run con el software de spyder *F5*):

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

