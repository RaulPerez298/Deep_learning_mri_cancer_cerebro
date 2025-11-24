from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D,Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add ,GlobalAveragePooling2D,Concatenate,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

def reduction_B(x):
    x_a=MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    x_b=Conv2D(256, kernel_size=(1,1), strides=(1, 1), padding='same')(x)
    x_b = BatchNormalization()(x_b)
    x_b = Activation(activations.relu)(x_b)

    x_b=Conv2D(288, kernel_size=(3,3), strides=(2, 2), padding='valid')(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation(activations.relu)(x_b)

    x_c=Conv2D(256, kernel_size=(1,1), strides=(1, 1), padding='same')(x)
    x_c = BatchNormalization()(x_c)
    x_c = Activation(activations.relu)(x_c)

    x_c=Conv2D(288, kernel_size=(3,3), strides=(2, 2), padding='valid')(x_c)
    x_c = BatchNormalization()(x_c)
    x_c = Activation(activations.relu)(x_c)

    x_d=Conv2D(256, kernel_size=(1,1), strides=(1, 1), padding='same')(x)
    x_d = BatchNormalization()(x_d)
    x_d = Activation(activations.relu)(x_d)
    x_d=Conv2D(288, kernel_size=(3,3), strides=(1, 1), padding='same')(x_d)
    x_d = BatchNormalization()(x_d)
    x_d = Activation(activations.relu)(x_d)
    x_d=Conv2D(320, kernel_size=(3,3), strides=(2, 2), padding='valid')(x_d)
    x_d = BatchNormalization()(x_d)
    x_d = Activation(activations.relu)(x_d)

    x=Concatenate()([x_a,x_b,x_c,x_d])
    
    # x = Activation(activations.relu)(x)
    
    return x


def reduction_A(x,k,l,m,n):

    
    x_a=MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    
    x_b=Conv2D(n, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x_b = BatchNormalization()(x_b)
    x_b = Activation(activations.relu)(x_b)

    
    x_c=Conv2D(k, kernel_size=(1,1), strides=(1, 1), padding='same')(x)
    x_c = BatchNormalization()(x_c)
    x_c = Activation(activations.relu)(x_c)
    x_c=Conv2D(l, kernel_size=(3,3), strides=(1, 1), padding='same')(x_c)
    x_c = BatchNormalization()(x_c)
    x_c = Activation(activations.relu)(x_c)
    x_c=Conv2D(m, kernel_size=(3,3), strides=(2, 2), padding='valid')(x_c)
    x_c = BatchNormalization()(x_c)
    x_c = Activation(activations.relu)(x_c)
    
    x=Concatenate()([x_a,x_b,x_c])
    
    # x = Activation(activations.relu)(x)
    
    
    
    return (x)

def Inception_ResNet_C(x):
    x_skip=x
    x_a=Conv2D(192, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x_a = BatchNormalization()(x_a)
    x_a = Activation(activations.relu)(x_a)

    
    x_b=Conv2D(192, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x_b = BatchNormalization()(x_b)
    x_b = Activation(activations.relu)(x_b)
    x_b=Conv2D(224, kernel_size=(1, 3), strides=(1, 1), padding='same')(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation(activations.relu)(x_b)
    x_b=Conv2D(224, kernel_size=(3, 1), strides=(1, 1), padding='same')(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation(activations.relu)(x_b)
    
    x_r=Concatenate()([x_a,x_b])
    
    x_r=Conv2D(2048, kernel_size=(1, 1), strides=(1, 1), padding='same')(x_r)
    
    x = Add()([x_r, x_skip])
    x = Activation(activations.relu)(x)
    
    return (x)

def Inception_ResNet_A(x):
    x_skip=x
    x1=Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation(activations.relu)(x1)
    
    x2=Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation(activations.relu)(x2)
    x2=Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation(activations.relu)(x2)
    
    x3=Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = Activation(activations.relu)(x3)
    x3=Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation(activations.relu)(x3)
    x3=Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation(activations.relu)(x3)
    
    x_r=Concatenate()([x1,x2,x3])
    x_r=Conv2D(384, kernel_size=(1, 1), strides=(1, 1), padding='same')(x_r)
    
    
    x = Add()([x_r, x_skip])
    x = Activation(activations.relu)(x)
    
    return (x)


def Inception_ResNet_B(x):
    
    x_skip=x#De aqui se obtienen 1152 capas no 1154 como dice el paper.
    x1=Conv2D(192, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation(activations.relu)(x1)
    
    x2=Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation(activations.relu)(x2)
    x2=Conv2D(160, kernel_size=(1, 7), strides=(1, 1), padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation(activations.relu)(x2)
    x2=Conv2D(192, kernel_size=(7, 1), strides=(1, 1), padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation(activations.relu)(x2)
    
    x_r=Concatenate()([x1,x2])
    
    x_r=Conv2D(1152, kernel_size=(1, 1), strides=(1, 1), padding='same')(x_r)
    
    
    x = Add()([x_r, x_skip])
    x = Activation(activations.relu)(x)
    
    return(x)

def stem (input_im):
      
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='valid')(input_im)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    
    
    x1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x2=Conv2D(96, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation(activations.relu)(x2)

    x=Concatenate()([x1,x2])
    
    x1=Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation(activations.relu)(x1)
    x1=Conv2D(96, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation(activations.relu)(x1)
    
    x2=Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation(activations.relu)(x2)
    x2=Conv2D(64, kernel_size=(7, 1), strides=(1, 1), padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation(activations.relu)(x2)
    x2=Conv2D(64, kernel_size=(1, 7), strides=(1, 1), padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation(activations.relu)(x2)
    x2=Conv2D(96, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation(activations.relu)(x2)

    x=Concatenate()([x1,x2])
    
    x1=Conv2D(192, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation(activations.relu)(x1)

    x2 = MaxPooling2D( strides=(2, 2), padding='valid')(x)

    x=Concatenate()([x1,x2])
    # x = Activation(activations.relu)(x)
    return x


def inception_resnet_v2(input_x,input_y,input_z,output_red):
    
    
    input_im = Input(shape=(input_x,input_y,input_z)) # cifar 10 images size
    x=stem (input_im)
    
    x=Inception_ResNet_A(x)
    x=Inception_ResNet_A(x)
    x=Inception_ResNet_A(x)
    x=Inception_ResNet_A(x)
    x=Inception_ResNet_A(x)
    
    x=reduction_A(x,k=256,l=256,m=384,n=384) 
    
    x=Inception_ResNet_B(x)
    x=Inception_ResNet_B(x)
    x=Inception_ResNet_B(x)
    x=Inception_ResNet_B(x)
    x=Inception_ResNet_B(x)
    x=Inception_ResNet_B(x)
    x=Inception_ResNet_B(x)
    x=Inception_ResNet_B(x)
    x=Inception_ResNet_B(x)
    x=Inception_ResNet_B(x)

    x=reduction_B(x)
    
    x=Inception_ResNet_C(x)
    x=Inception_ResNet_C(x)
    x=Inception_ResNet_C(x)
    x=Inception_ResNet_C(x)
    x=Inception_ResNet_C(x)
    
    x=GlobalAveragePooling2D()(x)
    
    x=Dropout(0.8)(x)
    x = Dense(output_red, activation='softmax')(x) #multi-class
    
    # model = Model(inputs=input_im, outputs=x, name='inception_resnet_v2')
    
    return x,input_im
