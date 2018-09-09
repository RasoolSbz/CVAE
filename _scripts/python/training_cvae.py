import nn_core

dataset = 'Modelnet10'
network_type = 'mrSabzi'
image_size = 32
load_saved_model = False
number_of_epochs = 10
batch_size = 10
dropout_ratio = 0.2
learning_rate = 0.0001
net_optimizer = 'SGD' # ADAM, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
loss_fcn = 'categorical_crossentropy' #'weighted_ours'#'categorical_crossentropy' # 'categorical_hinge' # 'categorical_crossentropy' 'mean_squared_error' mean_absolute_error
username = 'kamyab'

nn_core.trainAndEvaluate( \
username, \
dataset, \
network_type, \
image_size, \
load_saved_model, \
number_of_epochs, \
batch_size, \
dropout_ratio, \
learning_rate, \
net_optimizer, \
loss_fcn)
