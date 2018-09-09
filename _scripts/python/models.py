#from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
#from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, Flatten, Dense, Activation, ZeroPadding2D, BatchNormalization

from keras import backend as K

from keras.layers import Input, Dense, Conv3D, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization,MaxPooling3D,UpSampling3D,ZeroPadding3D,Dropout,Flatten,Reshape,Lambda
from keras.layers.merge import concatenate as concat
from keras.models import Model,Sequential
#from keras import backend as K
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
# ------------------------------------------------------------------------------
def get_mrSabzi_net():
# ------------------------------------------------------------------------------
    def vae_loss(y_true, y_pred):
      recon = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
      kl = 0.5 * K.sum(K.exp(sigma) + K.square(mu) - 1. - sigma, axis=-1)
      return recon + kl
    
    def KL_loss(y_true, y_pred):
    	return(0.5 * K.sum(K.exp(sigma) + K.square(mu) - 1. - sigma, axis=1))
    
    def recon_loss(y_true, y_pred):
    	return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
    
    def sampling(args):
      mu, l_sigma=args
      eps = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
      return mu + K.exp(l_sigma / 2) * eps
    
    def Encoder3D(input_3d,ouput_size):
      conv=Conv3D(8, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='3D_conv1_1')(input_3d)
      conv=Conv3D(8, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='3D_conv1_2')(conv)
      pool=MaxPooling3D(name='3d_pool_1')(conv)
      conv=Conv3D(16, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='3D_conv2_1')(pool)
      conv=Conv3D(16, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='3D_conv2_2')(conv)
      pool=MaxPooling3D(name='3d_pool_2')(conv)
      conv=Conv3D(32, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='3D_conv3_1')(pool)
      conv=Conv3D(32, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='3D_conv3_2')(conv)
      pool=MaxPooling3D(name='3d_pool_3')(conv)
      conv=Conv3D(64, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='3D_conv4_1')(pool)
      conv=Conv3D(64, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='3D_conv4_2')(conv)
      flat=Flatten(name='3d_flatten')(conv)
      dense=Dense(256,activation='relu',name='3d_dense_1')(flat)
      dense=Dense(ouput_size,activation='relu',name='3d_dense_2')(dense)
      return dense
    
    def Encoder2D(input_2d,ouput_size):
      conv=Conv2D(8, (3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='2D_conv1_1')(input_2d)
      conv=Conv2D(8, (3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='2D_conv1_2')(conv)
      pool=MaxPooling2D(name='2d_pool_1')(conv)
      conv=Conv2D(16, (3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='2D_conv2_1')(pool)
      conv=Conv2D(16, (3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='2D_conv2_2')(conv)
      pool=MaxPooling2D(name='2d_pool_2')(conv)
      conv=Conv2D(32, (3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='2D_conv3_1')(pool)
      conv=Conv2D(32, (3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='2D_conv3_2')(conv)
      pool=MaxPooling2D(name='2d_pool_3')(conv)
      conv=Conv2D(64, (3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='2D_conv4_1')(pool)
      conv=Conv2D(64, (3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='2D_conv4_2')(conv)
      flat=Flatten(name='2d_flatten')(conv)
      dense=Dense(256,activation='relu',name='2d_dense_1')(flat)
      dense=Dense(ouput_size,activation='relu',name='2d_dense_2')(dense)
      return dense
    
    def Decoder3D(latent_output,decoder3D_dense_shape,decoder3D_reshape_shape):
      dense = Dense(decoder3D_dense_shape, activation='relu')(latent_output)
      reshape =Reshape(decoder3D_reshape_shape)(dense)
      conv=Conv3D(64, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',activation='relu',name='3D_conv5_1')(reshape)
      conv=Conv3D(64, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',activation='relu',name='3D_conv5_2')(conv)
      up=UpSampling3D(size = (2, 2, 2),name='3D_up1')(conv)
      conv=Conv3D(32, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',activation='relu',name='3D_conv6_1')(up)
      conv=Conv3D(32, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',activation='relu',name='3D_conv6_2')(conv)
      up=UpSampling3D(size = (2, 2, 2),name='3D_up2')(conv)
      conv=Conv3D(16, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',activation='relu',name='3D_conv7_1')(up)
      conv=Conv3D(16, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',activation='relu',name='3D_conv7_2')(conv)
      up=UpSampling3D(size = (2, 2, 2),name='3D_up3')(conv)
      conv=Conv3D(8, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',activation='relu',name='3D_conv8_1')(up)
      conv=Conv3D(8, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',activation='relu',name='3D_conv8_2')(conv)
      conv=Conv3D(2, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',activation='relu',name='3D_conv8_3')(conv)
      conv=Conv3D(1, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',activation='sigmoid',name='3D_conv_output')(conv)
      return conv
    
    input3D_shape=(32,32,32,1)
    input2D_shape=(32,32,1)
    output2D_dim=32
    output3D_dim=32
    latent_dim = 10
    decoder3D_dense_shape=(4*4*4*64)
    decoder3D_reshape_shape=(4,4,4,64)
    batch_size=16
    
    input_3d=Input(input3D_shape,name='3D_Encoder_input')
    input_2d=Input(input2D_shape,name='2D_Encoder_input')
    
    Code_3d=Encoder3D(input_3d,output3D_dim)
    Code_2d=Encoder2D(input_2d,output2D_dim)
    
    inputs = concat([Code_3d, Code_2d],name='input_concat')
    
    encoder = Dense(512, activation='relu',name='encoder_dense')(inputs)
    mu = Dense(latent_dim, activation='linear',name='mu')(encoder)
    sigma = Dense(latent_dim, activation='linear',name='sigma')(encoder)
    latent = Lambda(sampling, output_shape = (latent_dim, ),name='latent')([mu, sigma])
    latent_concat = concat([latent, Code_2d],name='latent_concat')
    output = Decoder3D(latent_concat,decoder3D_dense_shape,decoder3D_reshape_shape)
    
    cvae = Model([input_3d, input_2d], output)
    
#    encoder = Model([input_3d, input_2d], mu)
#    
#    d_in = Input(shape=(latent_dim+output2D_dim,))
#    d_out = Decoder3D(d_in,decoder3D_dense_shape,decoder3D_reshape_shape)
#    decoder = Model(d_in, d_out)
#                                                               
    cvae.compile(optimizer='Adam', loss=vae_loss, metrics = [KL_loss, recon_loss])
#    cvae.summary()
#    
#    cvae.save('cvae.h5')
    return cvae


# ------------------------------------------------------------------------------