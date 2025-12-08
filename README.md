# Deep_learning_mri_cancer_cerebro



[Instalador de Cuda](instal_cuda.sh)





```
sudo apt install git
git clone https://github.com/RaulPerez298/Deep_learning_mri_cancer_cerebro.git
cd Deep_learning_mri_cancer_cerebro
bash ./instal_cuda.sh
```





```
bash ./datos_crear_python.sh
```



```
conda activate tfcpu
```







```
python newdatasetGamma.py
```



[agc_function.py](agc_function.py)



```
spyder datos_division_C_V_este_si_cx.py
```





```
spyder datos_division_C_V_este_si_para_lamejora_imagen.py
```



En [red_c_v_todas.py](red_c_v_todas.py) se debera de modificar la variable "*red_a_entrenar*", la cual puede tomar valores de 0 hasta 3 dependiendo del modelo que se desea entrenar, donde la lista de los modelos esta definidos como *['inception_resnet_v2','resnet50','nuestro_v8','resnet50_elu']* .



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



