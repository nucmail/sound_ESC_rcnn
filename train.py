import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import librosa
import soundfile as sound

from tensorflow import keras
import tensorflow
from tensorflow.keras.optimizers import SGD

from DCASE2019_network import model_resnet
from DCASE_training_functions import LR_WarmRestart, MixupGenerator
import matplotlib.pyplot as plt


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# train = np.load(r'E:\声源识别\声源例程\sound\train.npy')
# x_train = np.expand_dims(train, -1)
# print(x_train.shape)
x_train = np.load('tra_data.npy')
# x_train = np.log(x_train+1e-8)
y_train = np.load('tra_targets.npy')

# val = np.load(r'E:\声源识别\声源例程\sound\val.npy')
# x_val = np.expand_dims(val, -1)
# print(x_val.shape)
x_val = np.load('val_data.npy')
# x_val = np.log(x_val+1e-8)
y_val = np.load('val_targes.npy')

# test = np.load(r'E:\声源识别\声源例程\sound\test.npy')
# x_test = np.expand_dims(test, -1)
# print(x_test.shape)
x_test = np.load('x_test.npy')
# x_test = np.log(x_test+1e-8)
y_test = np.load('y_test.npy')
def deltas(X_in):
    X_out = (X_in[:,:,2:,:]-X_in[:,:,:-2,:])/10.0
    X_out = X_out[:,:,1:-1,:]+(X_in[:,:,4:,:]-X_in[:,:,:-4,:])/5.0
    return X_out

LM_train = np.log(x_train+1e-8)
LM_deltas_train = deltas(LM_train)
LM_deltas_deltas_train = deltas(LM_deltas_train)
x_train = np.concatenate((LM_train[:,:,4:-4,:],LM_deltas_train[:,:,2:-2,:],LM_deltas_deltas_train),axis=-1)

LM_val = np.log(x_val+1e-8)
LM_deltas_val = deltas(LM_val)
LM_deltas_deltas_val = deltas(LM_deltas_val)
x_val = np.concatenate((LM_val[:,:,4:-4,:],LM_deltas_val[:,:,2:-2,:],LM_deltas_deltas_val),axis=-1)

LM_test = np.log(x_test+1e-8)
LM_deltas_test = deltas(LM_test)
LM_deltas_deltas_test = deltas(LM_deltas_test)
x_test = np.concatenate((LM_test[:,:,4:-4,:],LM_deltas_test[:,:,2:-2,:],LM_deltas_deltas_test),axis=-1)

# x_train = deltas(x_train)
# x_val = deltas(x_val)
max_lr = 0.1
batch_size = 32
num_epochs = 2048
mixup_alpha = 0.4
crop_length = 400
NumClasses = 50
model = model_resnet(NumClasses,
                     input_shape=[128, None, 3],
                     num_filters=24,
                     wd=1e-3)
model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.0001, decay=0, momentum=0.9, nesterov=False),
              metrics=['accuracy'])

model.summary()

# lr_scheduler = LR_WarmRestart(nbatch=np.ceil(x_train.shape[0]/batch_size), Tmult=2,
#                              initial_lr=max_lr, min_lr=max_lr*1e-4,
#                              epochs_restart=[3.0, 7.0, 15.0, 31.0, 63.0,127.0,255.0,511.0])

lr_scheduler = LR_WarmRestart(nbatch=np.ceil(x_train.shape[0]/batch_size), Tmult=2,
                              initial_lr=max_lr, min_lr=max_lr*1e-4,
                              epochs_restart=[3.0, 7.0, 15.0, 31.0, 63.0,127.0,255.0,511.0])
callbacks = [lr_scheduler]

#create data generator
TrainDataGen = MixupGenerator(x_train,
                              y_train,
                              batch_size=batch_size,
                              alpha=mixup_alpha,
                              crop_length=crop_length)()

#train the model
history = model.fit_generator(TrainDataGen,
                              validation_data=(x_val, y_val),
                              epochs=num_epochs,
                              verbose=1,
                              workers=4,
                              max_queue_size=50,
                              callbacks=callbacks,
                              steps_per_epoch=np.ceil(x_train.shape[0]/batch_size)
                              )
model.save('bestModel22.h5')
loss, accuracy = model.evaluate(x_val, y_val, batch_size=1)
print(loss, accuracy)

accuracy = history.history['accuracy']
# val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
plt.figure()
plt.plot(loss, 'r', label='Train loss')
# plt.plot( val_loss, 'b', label='Validation loss')
plt.title('Training loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
# 准确率变化
plt.figure()
plt.plot(accuracy, 'b', label='Train acc')
# plt.plot( val_accuracy, 'b', label='Validation accuracy')
plt.title('Training accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()


plt.show()

# 测试
score = model.evaluate(x_test, y_test, verbose=1)
print('准确率', score[1])
