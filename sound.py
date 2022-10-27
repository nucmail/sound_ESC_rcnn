import pandas as pd
import csv
import os
import numpy as np
import librosa
import librosa.display
import soundfile as sound
from sklearn.model_selection import KFold

def numdir(path):
    fileNum = 0
    # global fileNum
    for lists in os.listdir(path):
        sub_path = os.path.join(path, lists)
        if os.path.isfile(sub_path):
            fileNum = fileNum + 1
    return fileNum

def label_maker(label):
    chars = sorted(list(set(label)))  ##函数对所有可迭代的对象进行排序操作。
    char_indices = dict((char, chars.index(char)) for char in chars)
    x = np.zeros((len(label), len(chars)), dtype='float64')
    for i, char in enumerate(label):
        x[i, char_indices[char]] = 1
    return x

def data_split(x):
    train = []
    val = []
    test = []
    for i in range(len(x)):
        sample = x[i]
        bi = int(0.6*len(sample))
        bj = int(0.2*len(sample))
        train.append(sample[0:bi])
        val.append(sample[bi:bi+bj])
        test.append(sample[bi+bj:])
    train = np.concatenate(train,axis=0)
    val = np.concatenate(val,axis=0)
    test = np.concatenate(test,axis=0)
    return train, val, test

sr = 48000
NumFreqBins = 128
NumFFTPoints = 2048
HopLength = int(NumFFTPoints/2)
audio_name = []
audio_train = []
audio_train_label = []

audio_path = r'E:\声源识别\声源例程\sound_rnn\audio'
meta_path = r'E:\声源识别\声源例程\sound_rnn\meta'
annotation_data = pd.read_csv('./meta/esc50.csv', low_memory=False)

audio_category = annotation_data['category'].unique().tolist()
# audio_train_label = label_maker(audio_category)
print(audio_category)
