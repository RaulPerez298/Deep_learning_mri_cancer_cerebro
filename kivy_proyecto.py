import os
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silencia logs ruidosos
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from tensorflow import device
from modelo_resnet50 import resnet50
from nuestromodelo_v8 import nuestro_v8
from modelo_resnet50_elu import resnet50_elu

from inception_resnet_v2 import inception_resnet_v2
import tensorflow as tf
# print("Dispositivo a usarse",tf.config.list_physical_devices())  # fuerza la inicialización


class ImageApp(App):
    def build(self):
        self.selected_image = None
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

        main_layout = BoxLayout(orientation='vertical', padding=30, spacing=10)

        # Selector de modelo
        self.spinner = Spinner(
            text='Selecciona modelo',
            values=('ResNet50 Modificado', 'Modelo v8', "Inception resnet v2","Resnet 50"),
            size_hint=(1, None),
            height=44
        )
        main_layout.add_widget(self.spinner)

        # Botones principales
        button_layout = BoxLayout(size_hint=(1, None), height=50, spacing=10)
        search_btn = Button(text="Buscar imagen")
        apply_btn = Button(text="Aplicar")
        search_btn.bind(on_press=self.open_file_chooser)
        apply_btn.bind(on_press=self.apply_model)
        button_layout.add_widget(search_btn)
        button_layout.add_widget(apply_btn)
        main_layout.add_widget(button_layout)

        # Área de imágenes
        img_layout = BoxLayout(orientation='horizontal', spacing=10)
        self.img1 = Image()
        self.img2 = Image()
        img_layout.add_widget(self.img1)
        img_layout.add_widget(self.img2)
        main_layout.add_widget(img_layout)

        # Texto de resultado
        self.result_label = Label(text='', size_hint=(1, None), height=40)
        main_layout.add_widget(self.result_label)

        return main_layout

    def open_file_chooser(self, instance):
        """Abre un popup con FileChooser a la izquierda y vista previa a la derecha."""
        # File chooser (panel izquierdo)
        filechooser = FileChooserIconView(
            path=self.current_dir,
            filters=['*.png', '*.jpg', '*.jpeg'],
            size_hint=(0.7, 1)
        )

        # Panel derecho: preview + botones
        right_box = BoxLayout(orientation='vertical', size_hint=(0.3, 1), spacing=10, padding=10)
        preview = Image(allow_stretch=True, keep_ratio=True)  # vista previa
        preview.source = ''  # vacío al inicio
        preview_box = BoxLayout(size_hint=(1, 0.9))
        preview_box.add_widget(preview)
        right_box.add_widget(preview_box)

        # Botón de seleccionar
        btn_select = Button(text='Seleccionar', size_hint=(1, 0.1))
        right_box.add_widget(btn_select)

        # Layout del popup
        content = BoxLayout(orientation='horizontal')
        content.add_widget(filechooser)
        content.add_widget(right_box)

        popup = Popup(title="Seleccionar imagen", content=content, size_hint=(0.9, 0.9))

        # Cuando cambie la selección, actualiza la preview
        def on_selection(instance, selection):
            if selection:
                # usa AsyncImage para que no bloquee si la imagen es grande
                preview.source = selection[0]
                preview.reload()
            else:
                preview.source = ''
        filechooser.bind(selection=on_selection)

        # Cuando el usuario haga doble click (on_submit) o presione el botón "Seleccionar"
        def do_select(*args):
            sel = filechooser.selection
            if sel:
                self.select_image(sel, popup)

        filechooser.bind(on_submit=lambda chooser, selection, touch: do_select())
        btn_select.bind(on_press=lambda inst: do_select())

        popup.open()
    def select_image(self, selection, popup):
        """Guarda la ruta de la imagen seleccionada y la muestra."""
        if selection:
            self.selected_image = selection[0]
            self.img1.source = self.selected_image
            self.img1.reload()
            self.result_label.text = "Imagen cargada correctamente."
        popup.dismiss()

    def apply_model(self, instance):
        """Carga el modelo seleccionado, aplica la inferencia y muestra el resultado."""
        if not self.selected_image:
            self.result_label.text = "Primero selecciona una imagen."
            return
        with device('/CPU:0'):
            resultado=""
            clases=2
            names_red=['inception_resnet_v2','resnet50','nuestro_v8','resnet50_elu']
            selected_model = self.spinner.text
            
            
            if selected_model == 'ResNet50 Modificado':
                altura, longitud = 256, 256
                x,input_im=resnet50_elu(altura, longitud,3,output_red=clases)
                name_r=names_red[3]
                model=Model(inputs=input_im, outputs=x, name=name_r)
                # model.load_weights(os.path.join(self.current_dir, './c_v_pesos_resnet50_elu/val_accuracy_resnet50_elu_best_cnn1.weights.h5'))
            elif selected_model == 'Modelo v8':
                altura, longitud = 224, 224
                x,input_im=nuestro_v8(altura, longitud,3,output_red=clases)
                name_r=names_red[2]
                model=Model(inputs=input_im, outputs=x, name=name_r)
                # model.load_weights(os.path.join(self.current_dir, './c_v_pesos_nuestro_v8/val_accuracy_nuestro_v8_best_cnn1.weights.h5'))  		    
            elif selected_model == 'Inception resnet v2':
                altura, longitud = 299,299
                name_r=names_red[0]
                x,input_im=inception_resnet_v2(altura, longitud,3,clases)
                model=Model(inputs=input_im, outputs=x, name=name_r)
                # model.load_weights(os.path.join(self.current_dir, './c_v_pesos_inception_resnet_v2/val_accuracy_inception_resnet_v2_best_cnn1.weights.h5'))
            elif selected_model == 'Resnet 50':
                altura, longitud = 256, 256
                x,input_im=resnet50(altura, longitud,3,output_red=clases)
                name_r=names_red[1]
                model=Model(inputs=input_im, outputs=x, name=name_r)
                # model.load_weights(os.path.join(self.current_dir, './c_v_pesos_resnet50/val_accuracy_resnet50_best_cnn1.weights.h5'))
            else:
                self.result_label.text = "Selecciona un modelo válido."
                return
            target_size = (altura, longitud)
      		# Cargar y preprocesar imagen
            img = keras_image.load_img(self.selected_image, target_size=target_size)
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            for index_model in range(5):
                if selected_model == 'ResNet50 Modificado':
                    
                    model.load_weights(os.path.join(self.current_dir, './c_v_pesos_resnet50_elu/val_accuracy_resnet50_elu_best_cnn'+str(1+index_model)+'.weights.h5'))
                elif selected_model == 'Modelo v8':
                    model.load_weights(os.path.join(self.current_dir, './c_v_pesos_nuestro_v8/val_accuracy_nuestro_v8_best_cnn'+str(1+index_model)+'.weights.h5'))  		    
                elif selected_model == 'Inception resnet v2':
                    model.load_weights(os.path.join(self.current_dir, './c_v_pesos_inception_resnet_v2/val_accuracy_inception_resnet_v2_best_cnn'+str(1+index_model)+'.weights.h5'))
                elif selected_model == 'Resnet 50':
                    model.load_weights(os.path.join(self.current_dir, './c_v_pesos_resnet50/val_accuracy_resnet50_best_cnn'+str(1+index_model)+'.weights.h5'))
          		# Inferencia
                pred = model.predict(img_array)
                clase = np.argmax(pred[0])
                resultado += "Tiene enfermedad para modelo "+str(1+index_model)+"\n" if clase == 1 else "No tiene enfermedad para modelo "+str(1+index_model)+"\n"

        # Mostrar imagen procesada (mismo archivo por ahora)
        self.img2.source = self.selected_image
        self.img2.reload()
        self.result_label.text = f"Imagen cargada — Resultado: \n{resultado}"

class FileChooserPopup(BoxLayout):
    pass

if __name__ == '__main__':
    ImageApp().run()

