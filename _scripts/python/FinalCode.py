from keras.models import Model
from keras import backend as K
from keras.layers import *
from keras.layers.merge import concatenate as concat
import numpy as np
import scipy.io as sio
from os import listdir
from os.path import isfile, join
from keras.callbacks import ModelCheckpoint

img_path = 'E:\\RS\\inverse rendering\\MODELNET_FINAL_DATA\\IMG_2D\\Train\\'
vox_path = 'E:\\RS\\inverse rendering\\MODELNET_FINAL_DATA\\VOX_GT\\Train\\'

img_path_Test = 'E:\\RS\\inverse rendering\\MODELNET_FINAL_DATA\\IMG_2D\\Test\\'
vox_path_Test = 'E:\\RS\\inverse rendering\\MODELNET_FINAL_DATA\\VOX_GT\\Test\\'
cc=0
vox_list = [f for f in listdir(vox_path) if isfile(join(vox_path, f))]
vox_list_test = [f for f in listdir(vox_path_Test) if isfile(join(vox_path_Test, f))]
def Generator(batch_size,vox_path,img_path):
    while True:
        #img_list=[f for f in listdir(img_path) if isfile(join(img_path, f))]
        vox_files=np.random.choice(vox_list, size=batch_size, replace=False)
        #img_files=np.random.choice(img_list, size=batch_size, replace=False)
        img2d=[sio.loadmat(img_path+str(f))['cropped_input_image'] for f in vox_files]
        voxels=[sio.loadmat(vox_path+str(f))['tmp'] for f in vox_files]
        img2d=np.array(img2d)
        voxels=np.array(voxels)
        img2d=np.expand_dims(img2d, axis=-1)
        voxels=np.expand_dims(voxels, axis=-1)
        #print('============>',img2d.shape)
        #print('============>',voxels.shape)
        yield ([voxels,img2d],voxels)

def Generator_Test(batch_size,vox_path_Test,img_path_Test,vox_list_test):
    while True:
        if(len(vox_list_test)<batch_size):
            vox_list_test = [f for f in listdir(vox_path_Test) if isfile(join(vox_path_Test, f))]
        #img_list=[f for f in listdir(img_path) if isfile(join(img_path, f))]
        vox_files=np.random.choice(vox_list_test, size=batch_size, replace=False)
        vox_list_test=[e for e in vox_list_test if e not in vox_files]
        #img_files=np.random.choice(img_list, size=batch_size, replace=False)
        img2d=[sio.loadmat(img_path_Test+str(f))['cropped_input_image'] for f in vox_files]
        voxels=[sio.loadmat(vox_path_Test+str(f))['tmp'] for f in vox_files]
        img2d=np.array(img2d)
        voxels=np.array(voxels)
        img2d=np.expand_dims(img2d, axis=-1)
        voxels=np.expand_dims(voxels, axis=-1)
        #print('============>',img2d.shape)
        #print('============>',voxels.shape)
        yield ([voxels,img2d],voxels)

def vae_loss(y_true, y_pred):
    recon = K.mean(K.binary_crossentropy(y_true, y_pred))
    kl = 0.5 * K.sum(K.exp(sigma) + K.square(mu) - 1. - sigma, axis=-1)
    print(recon.shape,kl.shape)
    return recon + kl

def KL_loss(y_true, y_pred):
    a=(0.5 * K.sum(K.exp(sigma) + K.square(mu) - 1. - sigma, axis=-1))
    print(a.shape)
    return a

def recon_loss(y_true, y_pred):
    a=K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
    print(a.shape)
    return a
def sampling(args):
    mu, sigma=args
    eps = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    return mu + K.exp(sigma / 2) * eps

def Encoder3D(input_3d,ouput_size):
    conv=Conv3D(8, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',name='3D_conv1_1')(input_3d)
    #conv=Conv3D(8, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='3D_conv1_2')(conv)
    actv=LeakyReLU(alpha=0.1,name='lrelu1')(conv)
    pool=MaxPooling3D(name='3d_pool_1')(actv)
    conv=Conv3D(64, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',name='3D_conv2_1')(pool)
    #conv=Conv3D(16, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='3D_conv2_2')(conv)
    actv=LeakyReLU(alpha=0.1,name='lrelu2')(conv)
    x = BatchNormalization()(actv)
    pool=MaxPooling3D(name='3d_pool_2')(x)
    conv=Conv3D(128, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',name='3D_conv3_1')(pool)
    #conv=Conv3D(32, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='3D_conv3_2')(conv)
    actv=LeakyReLU(alpha=0.1,name='lrelu3')(conv)
    x = BatchNormalization()(actv)
    pool=MaxPooling3D(name='3d_pool_3')(x)
    conv=Conv3D(256, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',name='3D_conv4_1')(pool)
    #conv=Conv3D(64, (3, 3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='3D_conv4_2')(conv)
    flat=Flatten(name='3d_flatten')(conv)
    dense=Dense(256,name='3d_dense_1')(flat)
    actv=LeakyReLU(alpha=0.1,name='lreluD1')(dense)
    x = BatchNormalization()(actv)
    x = Dropout(0.2)(x)
    dense=Dense(ouput_size,activation='relu',name='3d_dense_2')(x)
    return dense

def Encoder2D(input_2d,ouput_size):
    conv=Conv2D(8, (3, 3),padding='same',kernel_initializer = 'he_normal',name='2D_conv1_1')(input_2d)
    #conv=Conv2D(8, (3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='2D_conv1_2')(conv)
    actv=LeakyReLU(alpha=0.1,name='lrelu4')(conv)
    pool=MaxPooling2D(name='2d_pool_1')(actv)
    conv=Conv2D(64, (3, 3),padding='same',kernel_initializer = 'he_normal',name='2D_conv2_1')(pool)
    #conv=Conv2D(16, (3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='2D_conv2_2')(conv)
    actv=LeakyReLU(alpha=0.1,name='lrelu5')(conv)
    x = BatchNormalization()(actv)
    pool=MaxPooling2D(name='2d_pool_2')(x)
    conv=Conv2D(128, (3, 3),padding='same',kernel_initializer = 'he_normal',name='2D_conv3_1')(pool)
    #conv=Conv2D(32, (3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='2D_conv3_2')(conv)
    actv=LeakyReLU(alpha=0.1,name='lrelu6')(conv)
    x = BatchNormalization()(actv)
    pool=MaxPooling2D(name='2d_pool_3')(actv)
    conv=Conv2D(256, (3, 3),padding='same',kernel_initializer = 'he_normal',name='2D_conv4_1')(pool)
    #conv=Conv2D(64, (3, 3),padding='same',kernel_initializer = 'he_normal',activation='relu',name='2D_conv4_2')(conv)
    flat=Flatten(name='2d_flatten')(conv)
    dense=Dense(256,activation='relu',name='2d_dense_1')(flat)
    actv=LeakyReLU(alpha=0.1,name='lreluD3')(dense)
    x = BatchNormalization()(actv)
    dense=Dense(ouput_size,activation='relu',name='2d_dense_2')(x)
    return dense

def Decoder3D(latent_output,decoder3D_dense_shape,decoder3D_reshape_shape):
    dense = Dense(decoder3D_dense_shape, activation='relu')(latent_output)
    x = BatchNormalization()(dense)
    x = Dropout(0.2)(x)
    reshape =Reshape(decoder3D_reshape_shape)(x)
    conv=Conv3D(256, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',name='3D_conv5_1')(reshape)
    #conv=Conv3D(64, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',activation='relu',name='3D_conv5_2')(conv)
    actv=LeakyReLU(alpha=0.1,name='lrelu7')(conv)
    up=UpSampling3D(size = (2, 2, 2),name='3D_up1')(actv)
    conv=Conv3D(128, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',name='3D_conv6_1')(up)
    #conv=Conv3D(32, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',activation='relu',name='3D_conv6_2')(conv)
    actv=LeakyReLU(alpha=0.1,name='lrelu8')(conv)
    up=UpSampling3D(size = (2, 2, 2),name='3D_up2')(actv)
    conv=Conv3D(16, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',name='3D_conv7_1')(up)
    #conv=Conv3D(64, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',name='3D_conv7_2')(conv)
    actv=LeakyReLU(alpha=0.1,name='lrelu9')(conv)
    up=UpSampling3D(size = (2, 2, 2),name='3D_up3')(actv)
    conv=Conv3D(8, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',name='3D_conv8_1')(up)
    actv=LeakyReLU(alpha=0.1,name='lrelu10')(conv)
    #conv=Conv3D(8, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',activation='relu',name='3D_conv8_2')(conv)
    #conv=Conv3D(2, (3, 3, 3),kernel_initializer = 'he_normal',padding='same',activation='relu',name='3D_conv8_3')(conv)
    conv=Conv3D(1, (3, 3, 3),activation='sigmoid',kernel_initializer = 'he_normal',padding='same',name='3D_conv_output')(actv)
    return conv

def Test(encoder2d,decoder,noise_size,img_path_Test):
    img_list_test = [f for f in listdir(img_path_Test) if isfile(join(img_path_Test, f))]
    save_path='E:\\RS\\inverse rendering\\MODELNET_FINAL_DATA\\VOX_PRED\\Test\\'
    for img in img_list_test:
        img2d=sio.loadmat(img_path_Test+str(img))['cropped_input_image']
        img2d=np.expand_dims(img2d, axis=-1)
        img2d=np.expand_dims(img2d, axis=0)
        condition=encoder2d.predict(img2d)
        noise=np.random.normal(size=noise_size)
        #print(noise.shape,condition.shape)
        decoder_in=np.concatenate((noise.reshape((1,noise_size)),condition),axis=-1)
        pred=decoder.predict(decoder_in)   
        pred=pred.reshape(32,32,32)
        sio.savemat(save_path+str(img), {'pred_vox':pred})
        print(img)
   

    

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

#encoder = Dense(512, activation='relu',name='encoder_dense')(inputs)
mu = Dense(latent_dim, activation='linear',name='mu')(inputs)
sigma = Dense(latent_dim, activation='linear',name='sigma')(inputs)
latent = Lambda(sampling, output_shape = (latent_dim, ),name='latent')([mu, sigma])
latent_concat = concat([latent, Code_2d],name='latent_concat')
output = Decoder3D(latent_concat,decoder3D_dense_shape,decoder3D_reshape_shape)

cvae = Model(inputs=[input_3d, input_2d], outputs= output)

encoder = Model([input_3d, input_2d], mu)

d_in = Input(shape=(latent_dim+output2D_dim,))
d_out = Decoder3D(d_in,decoder3D_dense_shape,decoder3D_reshape_shape)
decoder = Model(d_in, d_out)
encoder2d=Model(input_2d,Code_2d)
cvae.compile(optimizer='Adam', loss=vae_loss, metrics = [KL_loss, recon_loss])
cvae.summary()

checkpointer = ModelCheckpoint(
   filepath = 'CVAE_checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5',\
   verbose = 1, \
   monitor = 'val_loss', \
   mode = 'auto', \
   save_best_only = True) # save at each epoch if the validation decreased
history =cvae.fit_generator(Generator(batch_size,vox_path,img_path), steps_per_epoch =1000, epochs =50,validation_data=Generator_Test(16,vox_path_Test,img_path_Test,vox_list_test), validation_steps=454,callbacks=[checkpointer])
#model.save(model_weights_path + '/' + 'saved_model_%s.h5' % (experiment_name))
#model.save_weights(model_weights_path + '/' + 'last_weights_%s.h5' % (experiment_name), overwrite=True)
cvae.save('cvae.h5')
Test(encoder2d,decoder,10,img_path_Test)