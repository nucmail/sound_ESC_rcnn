import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import librosa
import soundfile as sound
from keras.preprocessing.image import ImageDataGenerator

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

# from test_new import model_resnet
# from DCASE2019_network import model_resnet
from my_network import model_resnet
from DCASE_training_functions import LR_WarmRestart, MixupGenerator
import matplotlib.pyplot as plt


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.metrics import confusion_matrix
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0],True)

def train(batch_Size, epoch, tra_path, val_path):
# train = np.load(r'E:\声源识别\声源例程\sound\train.npy')
# x_train = np.expand_dims(train, -1)
# print(x_train.shape)
    x_train = np.load(tra_path + '//' + 'tra_data.npy')
    x_train = np.transpose(x_train,(0,3,2,1))
    print(x_train.shape)
    x_train = x_train[:,0:64,:,:]
    # print(x_train.shape)
    # x_train = np.log(x_train+1e-8)
    y_train = np.load(tra_path + '//' + 'tra_targets.npy')
    print(y_train.shape)

    # val = np.load(r'E:\声源识别\声源例程\sound\val.npy')
    # x_val = np.expand_dims(val, -1)
    # print(x_val.shape)
    x_val = np.load(val_path + '//' + 'val_data.npy')
    x_val = np.transpose(x_val,(0,3,2,1))
    x_val = x_val[:,0:64,:,:]
    # x_val = np.log(x_val+1e-8)
    y_val = np.load(val_path + '//' + 'val_targets.npy')


    # test = np.load(r'E:\声源识别\声源例程\sound\test.npy')
    # x_test = np.expand_dims(test, -1)
    # print(x_test.shape)
    max_lr = 0.1
    batch_size = batch_Size
    num_epochs = epoch
    mixup_alpha = 0.4
    crop_length = 400
    NumClasses = 50

    save_path = 'best_model.h5'

    model = model_resnet(NumClasses,
                         input_shape=[64, None, 3],
                         num_filters=24,
                         wd=1e-3)
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.0001, decay=0, momentum=0.9, nesterov=False),
                  metrics=['accuracy'])

    model.summary()

    # lr_scheduler = LR_WarmRestart(nbatch=np.ceil(x_train.shape[0]/batch_size), Tmult=2,
    #                              initial_lr=max_lr, min_lr=max_lr*1e-4,
    #                              epochs_restart=[3.0, 7.0, 15.0, 31.0, 63.0,127.0,255.0,511.0])

    lr_scheduler = LR_WarmRestart(nbatch=np.ceil(x_train.shape[0]/batch_size), Tmult=2,
                                  initial_lr=max_lr, min_lr=max_lr*1e-5,
                                  epochs_restart=[3.0, 7.0, 15.0, 31.0, 63.0,127.0,255.0,511.0,1023.0,2047.0])
                          
    cp_callback = keras.callbacks.ModelCheckpoint(save_path, 
                    monitor='val_accuracy', 
                    verbose=1, 
                    save_best_only=True, 
                    save_weights_only=False, 
                    mode='auto', 
                    period=1)                             
    callbacks = [lr_scheduler, cp_callback]
    #callbacks = [lr_scheduler]
    
    #create data generator

    TrainDataGen = MixupGenerator(x_train,
                                  y_train,
                                  batch_size=batch_size,
                                  alpha=mixup_alpha,
                                  crop_length=crop_length)()

    history = model.fit(TrainDataGen,
                                  validation_data=(x_val, y_val),
                                  epochs=num_epochs,
                                  verbose=1,
                                  workers=4,
                                  max_queue_size=50,
                                  callbacks=callbacks,
                                  steps_per_epoch=np.ceil(x_train.shape[0]/batch_size)
                                  )
    model.save('bestModel22.h5')

    best_model = keras.models.load_model('best_model.h5')
    loss, accuracy = best_model.evaluate(x_val, y_val, batch_size=batch_Size)
    return accuracy

if __name__ == '__main__':
    i = 0
    batch_Size = 64
    epoch = 2046
    path = ['i=0', 'i=1','i=2','i=3','i=4']
    tra_path = path[i]
    val_path = path[i]
    acc_last = 0.87
    while(1):
        acc_new = train(batch_Size, epoch, './', './')
        if acc_new > acc_last:
            best_model = keras.models.load_model('best_model.h5')
            best_model.save('./123'+'/'+'i=' +str(i)+'_'+str(acc_new)+'_'+'weight.h5')
            acc_last = acc_new
    # acc = train(batch_Size, epoch, tra_path, val_path)

