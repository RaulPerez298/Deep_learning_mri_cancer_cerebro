from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D,Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add ,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2




def res_identity(x, filters): 
    
    x_skip = x 
    f1, f2 = filters
    
    #first block 
    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x) #, kernel_regularizer=l2(0.001)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    
    
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    
    
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    
    
    
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x

def res_conv(x, s, filters):
    x_skip = x
    f1, f2 = filters
    
    
    x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    
    
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    
    
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    
    
    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='same')(x_skip)
    x_skip = BatchNormalization()(x_skip)
    
    
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x


def resnet50(input_x,input_y,input_z,output_red):
    
    input_im = Input(shape=(input_x,input_y,input_z)) 
    x = ZeroPadding2D(padding=(3, 3))(input_im)
    
    
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    
    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    
    
    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    
    
    x = res_conv(x, s=2, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    
    
    x = res_conv(x, s=2, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    
    x=GlobalAveragePooling2D()(x)
    x = Dense(output_red, activation='softmax')(x) #multi-class
    
    
    return x,input_im
