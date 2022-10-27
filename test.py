import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from DCASE_plots import plot_confusion_matrix
import matplotlib.pyplot as plt
import keras
import tensorflow

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
pred_audio_category = []

def deltas(X_in):
    X_out = (X_in[:,:,2:,:]-X_in[:,:,:-2,:])/10.0
    X_out = X_out[:,:,1:-1,:]+(X_in[:,:,4:,:]-X_in[:,:,:-4,:])/5.0
    return X_out

annotation_data = pd.read_csv('./meta/esc50.csv', low_memory=False)

audio_category = annotation_data['category'].unique().tolist()

x_test = np.load(r'E:\声源识别\声源例程\sound_ESC\x_test.npy')
y_test = np.load(r'E:\声源识别\声源例程\sound_ESC\y_test.npy')
LM_test = np.log(x_test+1e-8)
LM_deltas_test = deltas(LM_test)
LM_deltas_deltas_test = deltas(LM_deltas_test)
x_test = np.concatenate((LM_test[:,:,4:-4,:],LM_deltas_test[:,:,2:-2,:],LM_deltas_deltas_test),axis=-1)

best_model = keras.models.load_model('bestModel22.h5')
y_pred_val = np.argmax(best_model.predict(x_test), axis=1)
for i in y_pred_val:
    pred_lable = audio_category[i]
    pred_audio_category.append(pred_lable)
print(pred_audio_category)

score = best_model.evaluate(x_test, y_test, verbose=1)
print('准确率', score[1])
