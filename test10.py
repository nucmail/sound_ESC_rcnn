import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from DCASE_plots import plot_confusion_matrix
import matplotlib.pyplot as plt
import keras
import tensorflow
from sklearn.metrics import confusion_matrix
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
pred_audio_category = []


from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,#这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
                          normalize=True):
   
 
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    cmap = plt.get_cmap('Blues')
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #这里这个savefig是保存图片，如果想把图存在什么地方就改一下下面的路径，然后dpi设一下分辨率即可。
	#plt.savefig('/content/drive/My Drive/Colab Notebooks/confusionmatrix32.png',dpi=350)
    plt.show()
# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
    # predictions = model.predict_classes(x_val,batch_size=batch)
    predictions = np.argmax(best_model.predict(x_test), axis=1)
    truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, normalize=False,target_names=labels,title='Confusion Matrix')



audio_category = ['Dog bark', 'Rain', 'Sea_waves', 'Baby_cry', 'Clock_tick', 'Person_sneeze', 'Hellicopter', 'Chainsaw', 'Rooster', 'Fire_crackling']

x_test = np.load('./123/val_data.npy') 
x_test = np.transpose(x_test,(0,2,3,1))

print(x_test.shape)
y_test = np.load('./123/val_targets.npy')
print(y_test)

best_model = keras.models.load_model('./10_best_model/_i=4_0.9750000238418579_weight.h5')
# y_pred_val = np.argmax(best_model.predict(x_test), axis=1)
print(audio_category)
loss, acc=best_model.evaluate(x_test,y_test,batch_size=64)
print(acc)
labels = audio_category
plot_confuse(best_model, x_test, y_test)


