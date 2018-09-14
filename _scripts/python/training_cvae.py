import nn_core
from enum import Enum
from keras import optimizers

class optimizer(Enum):
    SGD=optimizers.SGD()
    ADAM=optimizers.Adagrad()
    SGD=optimizers.SGD()
    RMSprop=optimizers.RMSprop()
    Adagrad=optimizers.Adagrad()
    Adadelta=optimizers.Adadelta()
    Adamax=optimizers.Adamax()
    Nadam=optimizers.Nadam()

class losses(Enum):
    categorical_crossentropy='categorical_crossentropy'
    weighted_ours='weighted_ours'
    categorical_crossentropy='categorical_crossentropy'
    categorical_hinge='categorical_hinge'
    categorical_crossentropy='categorical_crossentropy'
    mean_squared_error='mean_squared_error' 
    mean_absolute_error='mean_absolute_error'

Args =	{
    "dataset" : 'Modelnet10',
    "network_type" : 'mrSabzi',
    "image_size" : 32,
    "load_saved_model" : False,
    "number_of_epochs" : 10,
    "batch_size" : 10,
    "dropout_ratio" : 0.2,
    "learning_rate" : 0.0001,
    "net_optimizer" : optimizer.SGD ,
    "loss_fcn" : losses.categorical_crossentropy,
    "username" : 'kamyab'
}

nn_core.trainAndEvaluate(Args)
