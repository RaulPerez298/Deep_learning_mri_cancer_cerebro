from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization,Concatenate, Flatten,Dense, Activation, Add ,GlobalAveragePooling2D,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam

# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.regularizers import l2
def res_conv(x, s, filters):
    x_skip = x
    f1, f2 = filters
    
    
    x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.elu)(x)
    
    
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.elu)(x)
    
    
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    
    
    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='same')(x_skip)
    x_skip = BatchNormalization()(x_skip)
    
    
    x = Add()([x, x_skip])
    x = Activation(activations.elu)(x)

    return x

def res_identity(x, filters): 
    
    x_skip = x 
    f1, f2 = filters
    
    #first block 
    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x) #, kernel_regularizer=l2(0.001)
    x = BatchNormalization()(x)
    x = Activation(activations.elu)(x)
    
    
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.elu)(x)
    
    
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    
    
    
    x = Add()([x, x_skip])
    x = Activation(activations.elu)(x)

    return x


def reduction_B(x):
    x_a=MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    x_b=Conv2D(192, kernel_size=(1,1), strides=(1, 1), padding='same')(x)
    x_b = BatchNormalization()(x_b)
    x_b = Activation(activations.elu)(x_b)

    x_b=Conv2D(256, kernel_size=(3,3), strides=(2, 2), padding='valid')(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation(activations.elu)(x_b)

    x_c=Conv2D(192, kernel_size=(1,1), strides=(1, 1), padding='same')(x)
    x_c = BatchNormalization()(x_c)
    x_c = Activation(activations.elu)(x_c)

    x_c=Conv2D(256, kernel_size=(3,3), strides=(2, 2), padding='valid')(x_c)
    x_c = BatchNormalization()(x_c)
    x_c = Activation(activations.elu)(x_c)

    x_d=Conv2D(192, kernel_size=(1,1), strides=(1, 1), padding='same')(x)
    x_d = BatchNormalization()(x_d)
    x_d = Activation(activations.elu)(x_d)
    x_d=Conv2D(224, kernel_size=(3,3), strides=(1, 1), padding='same')(x_d)
    x_d = BatchNormalization()(x_d)
    x_d = Activation(activations.elu)(x_d)
    x_d=Conv2D(256, kernel_size=(3,3), strides=(2, 2), padding='valid')(x_d)
    x_d = BatchNormalization()(x_d)
    x_d = Activation(activations.elu)(x_d)

    x=Concatenate()([x_a,x_b,x_c,x_d])
    
    # x = Activation(activations.relu)(x)
    
    return x



def nuestro_v8(input_x,input_y,input_z,output_red):
    
    
    input_im = Input(shape=(input_x,input_y,input_z)) 
    
    ####Capas adicionales
    x=Conv2D(64,kernel_size=(3,3),strides=(2,2),padding='same')(input_im)
    
    
    x=Conv2D(64,kernel_size=(3,3),padding='same')(x)

    x=MaxPooling2D((2, 2))(x)
    
    
    x=Conv2D(128,kernel_size=(3,3),padding='same')(x)

    x=MaxPooling2D((2,2))(x)
    
    ###########Aqui sigue el modelo que conocemos por el paper ############

    x=res_conv(x, s=2, filters=(192,256))
    x=res_identity(x, filters=(192,256))
    x=MaxPooling2D((2, 2))(x)
    
    x=reduction_B(x)



    x=Flatten()(x)
    
    x=Dense(2048)(x)
    x = Activation(activations.relu)(x)
    
    x=Dense(1024)(x)
    x=BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x=Dropout(0.4)(x)
    x=Dense(1024)(x)
    x = Activation(activations.relu)(x)
    
    x = Dense(output_red, activation='softmax')(x) #multi-class
    
    # model = Model(inputs=input_im, outputs=x, name='nuestra')
    # model.summary()
    return x,input_im
# x,y=nuestro_v6(224,224,3,2)
# cnn = Model(inputs=y, outputs=x)
# cnn.summary()

