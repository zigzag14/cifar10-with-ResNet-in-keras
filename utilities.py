import numpy as np
import os

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau


def lr_decay(epoch):
    """
    The learning rate scheme
    """
    lr = 1e-2
    
    if epoch > 180:
        lr = 1e-6
    elif epoch > 140:
        lr = 1e-5
    elif epoch > 100:
        lr = 1e-4
    elif epoch > 60:
        lr = 1e-3
    print('The learning rate is: ', lr)
    return lr


def lr_callbacks(model_name):   
    """
    Input:
    model_name:  the name of the model, it will be part of the saved model name of h5 file
    
    Output:
    callbacks functions
    """
    
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_decay)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=1e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

