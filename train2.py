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

save_path = 'best_model.h5'
# train = np.load(r'E:\声源识别\声源例程\sound\train.npy')
# x_train = np.expand_dims(train, -1)
# print(x_train.shape)
x_train = np.load('tra_data.npy')
#x_train = np.transpose(x_train,(0,3,2,1))
x_train = np.transpose(x_train,(0,2,3,1))
print(x_train.shape)
# x_train = np.log(x_train+1e-8)
y_train = np.load('tra_targets.npy')

# val = np.load(r'E:\声源识别\声源例程\sound\val.npy')
# x_val = np.expand_dims(val, -1)
# print(x_val.shape)
x_val = np.load('val_data.npy')
#x_val = np.transpose(x_val,(0,3,2,1))
x_val = np.transpose(x_val,(0,2,3,1))
# x_val = np.log(x_val+1e-8)
y_val = np.load('val_targets.npy')


# test = np.load(r'E:\声源识别\声源例程\sound\test.npy')
# x_test = np.expand_dims(test, -1)
# print(x_test.shape)
max_lr = 0.1
batch_size = 64
num_epochs = 510
mixup_alpha = 0.4
crop_length = 400
NumClasses = 10
model = model_resnet(NumClasses,
                     input_shape=[40, None, 1],
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
                              initial_lr=max_lr, min_lr=max_lr*1e-3,
                              epochs_restart=[3.0, 7.0, 15.0, 31.0, 63.0,127.0,255.0,511.0,1023.0,2047.0])
cp_callback = keras.callbacks.ModelCheckpoint(save_path, 
                    monitor='val_accuracy', 
                    verbose=1, 
                    save_best_only=True, 
                    save_weights_only=False, 
                    mode='auto', 
                    period=1)                             
callbacks = [lr_scheduler, cp_callback]

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
best_model = keras.models.load_model('best_model.h5')
loss, accuracy = best_model.evaluate(x_val, y_val, batch_size=1)
print(loss, accuracy)

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
plt.plot(tra_accuracy, 'y', label='Train_accuracy')
plt.plot(val_accuracy, 'b', label='val_accuracy')
# plt.plot( val_accuracy, 'b', label='Validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()


plt.show()

