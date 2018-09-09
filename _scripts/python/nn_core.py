import os
import sys
# import cv2
import h5py
import glob
import itertools
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
import models, paths, patches, performance

sys.path.insert(0, './lib/')


def evaluateResults(model, test_2dimage, test_voxels, debug_flag=False):
    score = model.evaluate(test_2dimage, test_voxels, verbose=0)
    
    voxel_preds_test = model.predict(test_2dimage)
    for i in range(voxel_preds_test.shape[0]):
        for j in range(voxel_preds_test.shape[1]):
            for k in range(voxel_preds_test.shape[2]):
                if voxel_preds_test[i,j,k] > 0.5:
                  voxel_preds_test[i,j,k] = 1
                else:
                  voxel_preds_test[i,j,k] = 0
            
    test_voxels_vectorized = np.reshape(test_voxels , (1,-1))
    voxel_preds_test_vectorized = np.reshape(voxel_preds_test, (1,-1))
    TP, FP, TN, FN = performance.get_performance(test_voxels_vectorized,   voxel_preds_test_vectorized)

    if debug_flag:
        print('Test score =', score[0])
        print('Test accuracy =', score[1])
        print('TP =', TP)
        print('FP =', FP)
        print('TN =', TN)
        print('FN =', FN)

    return voxel_preds_test, TP, FP, TN, FN



def getExperimentName(dataset, network_type, image_size):
  return '%s_cvae_%s_class_%d' % (network_type, dataset, image_size)

def trainAndEvaluate(username ,dataset, network_type, image_size, load_saved_model, number_of_epochs, batch_size, dropout_ratio, learning_rate, net_optimizer, loss_fcn):

   # ============================================================================
   #                                                                        Meta
   # ============================================================================
   experiment_name = getExperimentName(dataset, network_type, image_size)
   model_weights_path = paths.getModelWeightsDirectory(username)
   data_directory = paths.getDatasetDirectory(username)


   # ============================================================================
   #                                        Load the data and divided in patches
   # ============================================================================
   if dataset == 'Modelnet10':
       tmp_file = h5py.File(os.path.join(data_directory, 'modelNet_data_X_vox_train_10class.mat'))
       training_voxels = tmp_file.get('modelNet_data_X_vox')
       training_voxels = np.transpose(training_voxels)
       training_voxels = np.transpose(training_voxels, (0, 3, 1, 2))
       # patches_image = np.transpose(patches_image, (3, 0, 1, 2))
       training_voxels = np.array(training_voxels, dtype = '<f8')
       # = patches_image[0:samples_split] / 255
       #patches_image_test = patches_image[samples_split:] / 255
       print('[INFO] training voxel size: ', training_voxels.shape)
    
       # number_of_patches = patches_image_train.shape[0]
       voxel_depth = training_voxels.shape[1]
       voxel_height = training_voxels.shape[2]
       voxel_width = training_voxels.shape[3]
    
       tmp_file = h5py.File(os.path.join(data_directory, 'modelNet_data_X_2d_train_10class.mat'))
       training_2dimage = tmp_file.get('modelNet_data_X_2d')
       training_2dimage = np.transpose(training_2dimage)
       training_2dimage = np.array(training_2dimage, dtype = '<f8')
#       patches_class_train = to_categorical(patches_class[0:samples_split], num_classes = 2)
#       patches_class_test = to_categorical(patches_class[samples_split:], num_classes = 2)
       print('[INFO] training 2D image size: ', training_2dimage.shape)


   # ============================================================================
   #                                     Construct or load the model arcitecture
   # ============================================================================
   if network_type == 'mrSabzi':
     model_loader = models.get_mrSabzi_net
   

   if load_saved_model:

      model = load_model(model_weights_path + '/' + 'saved_model_%s.h5' % (experiment_name))

   else:

     model = model_loader(voxel_depth, voxel_height, voxel_width, dropout_ratio, learning_rate, net_optimizer, loss_fcn)

     print('[INFO] network input size: ', model.input_shape)
     print('[INFO] network output size: ', model.output_shape)


   # ============================================================================
   #                                                                    Training
   # ============================================================================
   checkpointer = ModelCheckpoint(
       filepath = model_weights_path + '/' + 'best_weights_%s.h5' % (experiment_name), \
       verbose = 1, \
       monitor = 'val_loss', \
       mode = 'auto', \
       save_best_only = True) # save at each epoch if the validation decreased
   model.fit(
       training_voxels, \
       training_2dimage, \
       nb_epoch = number_of_epochs, \
       batch_size = batch_size, \
       verbose = 1, \
       shuffle = True, \
       validation_split = 0.1, \
       callbacks = [checkpointer])
   model.save(model_weights_path + '/' + 'saved_model_%s.h5' % (experiment_name))
   model.save_weights(model_weights_path + '/' + 'last_weights_%s.h5' % (experiment_name), overwrite=True)
   saved_model = load_model(model_weights_path + '/' + 'saved_model_%s.h5' % (experiment_name))


   # ============================================================================
   #                                                       Testing & Performance
   # ============================================================================
   if dataset == 'Modelnet10':
       tmp_file = h5py.File(os.path.join(data_directory, 'modelNet_data_X_vox_test_10class.mat'))
       test_voxels = tmp_file.get('modelNet_data_X_vox')
       test_voxels = np.transpose(test_voxels)
       test_voxels = np.transpose(test_voxels, (0, 3, 1, 2))
       test_voxels = np.array(test_voxels, dtype = '<f8')
       print('[INFO] test voxel size: ', test_voxels.shape)
    
       voxel_depth = test_voxels.shape[1]
       voxel_height = test_voxels.shape[2]
       voxel_width = test_voxels.shape[3]
    
       tmp_file = h5py.File(os.path.join(data_directory, 'modelNet_data_X_2d_test_10class.mat'))
       test_2dimage = tmp_file.get('modelNet_data_X_2d')
       test_2dimage = np.transpose(test_2dimage)
       test_2dimage = np.array(test_2dimage, dtype = '<f8')
       print('[INFO] test 2D image size: ', test_2dimage.shape)

   voxel_preds_test, TP, FP, TN, FN = evaluateResults(saved_model, test_2dimage, test_voxels, debug_flag = True)
