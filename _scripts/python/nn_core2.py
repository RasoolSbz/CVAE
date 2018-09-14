import os
import sys
# import cv2
import h5py
#import glob
#import itertools
import scipy.misc
import scipy.io as sio
import skimage.io
import skimage.color
import numpy as np

from keras import backend as K
from keras.models import load_model

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model as plot
from sklearn.metrics import average_precision_score, recall_score, f1_score
import models, paths, performance


def Generator(batch_size,DataPath):
    while True:
        index=np.random.randint(low=200,high=1000,size=batch_size)
        img2d=[]
        voxels=[]
        for i in index:
            img=sio.loadmat(DataPath+'\\2d'+str(i)+'.mat')['I']
            voxel=sio.loadmat(DataPath+'\\vox'+str(i)+'.mat')['OUTPUTgrid']
            img2d.append(img)
            voxels.append(voxel)
           # print(i)
        img2d=np.array(img2d)
        voxels=np.array(voxels)
        depths=np.expand_dims(img2d, axis=-1)
        voxels=np.expand_dims(voxels, axis=-1)
        #print('============>',depths.shape)
        #print('============>',voxels.shape)
        yield depths,voxels


        
def getExperimentName(dataset, network_type, image_size):
  return '%s_cvae_%s_class_%d' % (network_type, dataset, image_size)

def trainAndEvaluate(Args):
    experiment_name = getExperimentName(Args["dataset"], Args["network_type"], Args["image_size"])
    model_weights_path = paths.getModelWeightsDirectory(Args["username"])
    data_directory = paths.getDatasetDirectory(Args["username"])
    if (Args["Train_Mode"]=="Direct"):
        if Args["dataset"] == 'Modelnet10':
            tmp_file = h5py.File(os.path.join(data_directory, 'modelNet_data_X_vox_train_10class.mat'))
            training_voxels = tmp_file.get('modelNet_data_X_vox')
            #training_voxels = np.transpose(training_voxels)
            #training_voxels = np.transpose(training_voxels, (0, 3, 1, 2))
            training_voxels = np.array(training_voxels, dtype = '<f8')
            training_voxels = np.expand_dims(training_voxels, axis=4)
            print('[INFO] training voxel size: ', training_voxels.shape)

            voxel_depth = training_voxels.shape[1]
            voxel_height = training_voxels.shape[2]
            voxel_width = training_voxels.shape[3]

            tmp_file = h5py.File(os.path.join(data_directory, 'modelNet_data_X_2d_train_10class.mat'))
            training_2dimage = tmp_file.get('modelNet_data_X_2d')
            #training_2dimage = np.transpose(training_2dimage)
            training_2dimage = np.array(training_2dimage, dtype = '<f8')
            training_2dimage = np.expand_dims(training_2dimage, axis=3)
            print('[INFO] training 2D image size: ', training_2dimage.shape)
        if Args["network_type"] == 'mrSabzi':
            model_loader = models.get_mrSabzi_net

        if Args["load_saved_model"]:

            model = load_model(model_weights_path + '/' + 'saved_model_%s.h5' % (experiment_name))

        else:
            model = model_loader()#(voxel_depth, voxel_height, voxel_width, dropout_ratio, learning_rate, net_optimizer, loss_fcn)

        print('[INFO] network input size: ', model.input_shape)
        print('[INFO] network output size: ', model.output_shape)
        checkpointer = ModelCheckpoint(
            filepath = model_weights_path + '/' + 'best_weights_%s.h5' % (experiment_name), \
            verbose = 1, \
            monitor = 'val_loss', \
            mode = 'auto', \
            save_best_only = True) # save at each epoch if the validation decreased
        model.fit(
            [training_voxels, \
            training_2dimage], \
            training_voxels, \
            nb_epoch = Args["number_of_epochs"], \
            batch_size = Args["batch_size"], \
            verbose = 1, \
            shuffle = True, \
            validation_split = 0.1, \
            callbacks = [checkpointer])
        model.save(model_weights_path + '/' + 'saved_model_%s.h5' % (experiment_name))
        model.save_weights(model_weights_path + '/' + 'last_weights_%s.h5' % (experiment_name), overwrite=True)
        saved_model = load_model(model_weights_path + '/' + 'saved_model_%s.h5' % (experiment_name))
    elif (Args["Train_Mode"]=="Generate"):
        checkpointer = ModelCheckpoint(
            filepath = model_weights_path + '\\' + 'best_weights_%s.h5' % (experiment_name), \
            verbose = 1, \
            monitor = 'val_loss', \
            mode = 'auto', \
            save_best_only = True) 
        history =model.fit_generator(Generator(Args["batch_size"],data_directory), samples_per_epoch =Args["TrainData_num"]//Args["number_of_epochs"], nb_epoch = Args["number_of_epochs"], validation_data=None, class_weight=None)
        model.save(model_weights_path + '/' + 'saved_model_%s.h5' % (experiment_name))
        model.save_weights(model_weights_path + '/' + 'last_weights_%s.h5' % (experiment_name), overwrite=True)
        saved_model = load_model(model_weights_path + '/' + 'saved_model_%s.h5' % (experiment_name))


