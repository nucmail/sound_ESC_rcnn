import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import librosa
import soundfile as sound
from keras.preprocessing.image import ImageDataGenerator

from tensorflow import keras
import tensorflow
from tensorflow.keras.optimizers import SGD

#from test_new import model_resnet
from DCASE2019_network import model_resnet
from DCASE_training_functions import LR_WarmRestart, MixupGenerator
import matplotlib.pyplot as plt


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.metrics import confusion_matrix

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def train(batch_Size, epoch, tra_path, val_path):
# train = np.load(r'E:\声源识别\声源例程\sound\train.npy')
# x_train = np.expand_dims(train, -1)
# print(x_train.shape)
    x_train = np.load(tra_path + '//' + 'tra_data.npy')
    x_train = np.transpose(x_train,(0,2,3,1))
    #x_train = x_train[:,:,:,0:1]
    print(x_train.shape)
    # x_train = np.log(x_train+1e-8)
    y_train = np.load(tra_path + '//' + 'tra_targets.npy')

    # val = np.load(r'E:\声源识别\声源例程\sound\val.npy')
    # x_val = np.expand_dims(val, -1)
    # print(x_val.shape)
    x_val = np.load(val_path + '//' + 'val_data.npy')
    x_val = np.transpose(x_val,(0,2,3,1))
    #x_val = x_val[:,:,:,0:1]
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
    NumClasses = 10

    save_path = 'best_model.h5'

    model = model_resnet(NumClasses,
                         input_shape=[64, None, 3],
                         num_filters=24,
                         wd=1e-3)
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.001, decay=0, momentum=0.9, nesterov=False),
                  metrics=['accuracy'])

    model.summary()

    # lr_scheduler = LR_WarmRestart(nbatch=np.ceil(x_train.shape[0]/batch_size), Tmult=2,
    #                              initial_lr=max_lr, min_lr=max_lr*1e-4,
    #                              epochs_restart=[3.0, 7.0, 15.0, 31.0, 63.0,127.0,255.0,511.0])

                          
    cp_callback = keras.callbacks.ModelCheckpoint(save_path, 
                    monitor='val_accuracy', 
                    verbose=1, 
                    save_best_only=True, 
                    save_weights_only=False, 
                    mode='auto', 
                    period=1)                             
    callbacks = [cp_callback]
    # callbacks = [cp_callback]
    
    #create data generator

    TrainDataGen = MixupGenerator(x_train,
                                  y_train,
                                  batch_size=batch_size,
                                  alpha=mixup_alpha,
                                  crop_length=crop_length)()
    #train the model

    history = model.fit(TrainDataGen,
                                  validation_data=(x_val, y_val),
                                  epochs=num_epochs,
                                  verbose=1,
                                  workers=4,
                                  max_queue_size=50,
                                  callbacks=callbacks,
                                  steps_per_epoch=np.ceil(x_train.shape[0]/batch_size)
                                  )
    # model.save('bestModel22.h5')

    best_model = keras.models.load_model('best_model.h5')
    loss, accuracy = best_model.evaluate(x_val, y_val, batch_size=batch_Size)
    return history

if __name__ == '__main__':
    i = 0
    batch_Size = 64
    epoch = 1024
    path = ['i=0', 'i=1','i=2','i=3','i=4']
    tra_path = path[i]
    val_path = path[i]
    history = train(batch_Size, epoch, './', './')
    
    tra_loss = history.history['loss']
    tra_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    plt.figure()
    plt.plot(val_loss, 'g', label='val_loss')
    plt.plot(tra_loss, 'r', label='Train_loss')
    # plt.plot( val_loss, 'b', label='Validation loss')
    plt.title('Training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    # 准确率变化
    plt.figure()
    plt.plot(val_accuracy, 'b', label='val_accuracy')
    plt.plot(tra_accuracy, 'y', label='Train_accuracy')
    # plt.plot( val_accuracy, 'b', label='Validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    # acc = train(batch_Size, epoch, tra_path, val_path)

