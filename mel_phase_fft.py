import pandas as pd
import csv
import os
import numpy as np
from scipy import misc
import imageio
import pickle,pprint
import librosa
import librosa.display
import soundfile as sound
import scipy
import tensorflow as tf

def mel_phase_fft(k_i, save_path):
    audio_path = r'./audio'
    meta_path = r'./meta'
    data = []
    label2 = []
    label_list = []
    label3 = []
    annotation_data = pd.read_csv('./meta/esc50.csv', low_memory=False)

    annotation_list = pd.read_csv('./meta/esc50.csv', low_memory=False)
    audio_list = annotation_list['category'].unique().tolist()

    category_list = np.array(annotation_data['category'])
    filename_list = np.array(annotation_data['filename'])
    # label = np.zeros((2,filename_list.shape[0]))

    tra_files = open(save_path + '//' + 'tra_targes.pkl','wb')
    val_files = open(save_path + '//' + 'val_targes.pkl','wb')


    all_feature = []
    sr = 441000
    window_length = 1024
    for i in range(filename_list.shape[0]):
        label1 = []
        label1.append(filename_list[i])
        label1.append(audio_list.index(category_list[i]))
        
        label2.append(filename_list[i])
        label3.append(audio_list.index(category_list[i]))
        label_list.append(label1)
        nosum_signal = []
        one_feature = []
        stereo, fs = sound.read('./audio/' + filename_list[i])
        print(stereo.shape)
        
        #log_mel
        melspec = librosa.feature.melspectrogram(stereo, fs, n_fft=1024, hop_length=1024, n_mels=36)
        logmelspec = librosa.power_to_db(melspec)
        print(logmelspec.shape)
        one_feature.append(logmelspec)
        '''
        mfcc = librosa.feature.mfcc(stereo, fs, n_fft=1024, hop_length=1024, n_mfcc=36)
        print(mfcc.shape)
        one_feature.append(mfcc)
        '''
        '''
        #phase
        phase_signal = []
        for j in range(0, len(stereo), window_length):
            win_signal = stereo[j:j+window_length]
            # win_signal *= np.hamming(window_length)
            ft_signal = np.angle(np.fft.rfft(win_signal, 72))
            phase_signal.append(ft_signal)
        phase_signal = np.array(phase_signal).T
        phase_signal = np.delete(phase_signal, -1, 0)
        print(np.array(phase_signal).shape)
        one_feature.append(phase_signal)
        '''
        '''
        #fft
        nosum_signal = []
        for j in range(0, len(stereo), window_length):
            win_signal = stereo[j:j+window_length]
            # win_signal *= np.hamming(window_length)
            ft_signal = np.absolute(np.fft.rfft(win_signal, 72))
            nosum_signal.append(ft_signal)
        nosum_signal = np.array(nosum_signal).T
        nosum_signal = np.delete(nosum_signal, -1, 0)
        print(np.array(nosum_signal).shape)
        one_feature.append(nosum_signal)
        '''
        '''
        #cqt
        cq_lib = librosa.feature.chroma_cqt(stereo, sr=44100, hop_length=1024, n_chroma=36)
        log_cqt = librosa.power_to_db(cq_lib)
        print(np.array(log_cqt).shape)
        one_feature.append(log_cqt)
        '''
        data.append(one_feature)
        print(i)
    print(np.array(data).shape)
    k = 5
    num_val_samples = len(data) // k
    # for i in range(k):
    #k_i = 3

    train_label_dict = []
    val_label_dict = []
    i = k_i
    # label2 = np.array(label2)
    # label3 = np.array(label3)
    # data = np.array(data)
    # print(np.array(label2).shape)
    # print(np.array(label3).shape)

    #one-hot
    audio_label = tf.one_hot(label3, depth=50, axis=0)
    audio_label = np.transpose(audio_label,(1,0))
    # print(label3)
    #not one-hot
    # audio_label = label3

    # print(audio_label)
    if i != 0 and i != 4:
        val_data = data[i*num_val_samples:(i+1)*num_val_samples]
        val_targets = label_list[i*num_val_samples:(i+1)*num_val_samples]
        val_label = audio_label[i*num_val_samples:(i+1)*num_val_samples]
        
        train_data = np.concatenate((data[:i*num_val_samples],data[(i+1)*num_val_samples:]),axis=0)
        train_targes = np.concatenate((label_list[:i*num_val_samples],label_list[(i+1)*num_val_samples:]),axis=0)
        train_label = np.concatenate((audio_label[:i*num_val_samples],audio_label[(i+1)*num_val_samples:]),axis=0)

        train_targes2 = np.concatenate((label2[:i*num_val_samples],label2[(i+1)*num_val_samples:]),axis=0)
        val_targets2 = label2[i*num_val_samples:(i+1)*num_val_samples]

        train_targes3 = np.concatenate((label3[:i*num_val_samples],label3[(i+1)*num_val_samples:]),axis=0)
        val_targets3 = label3[i*num_val_samples:(i+1)*num_val_samples]

        train_label_dict.append(train_targes2)
        train_label_dict.append(train_targes3)

        val_label_dict.append(val_targets2)
        val_label_dict.append(val_targets3)

    if i == 0:
        val_data = data[i*num_val_samples:(i+1)*num_val_samples]
        val_targets = label_list[i*num_val_samples:(i+1)*num_val_samples]
        val_label = audio_label[i*num_val_samples:(i+1)*num_val_samples]
        
        train_data = data[(i+1)*num_val_samples:]
        train_targes = label_list[(i+1)*num_val_samples:]
        train_label = audio_label[(i+1)*num_val_samples:]

        train_targes2 = label2[(i+1)*num_val_samples:]
        val_targets2 = label2[i*num_val_samples:(i+1)*num_val_samples]

        train_targes3 = label3[(i+1)*num_val_samples:]
        val_targets3 = label3[i*num_val_samples:(i+1)*num_val_samples]

        train_label_dict.append(train_targes2)
        train_label_dict.append(train_targes3)

        val_label_dict.append(val_targets2)
        val_label_dict.append(val_targets3)

    if i == 4:
        val_data = data[i*num_val_samples:]
        val_targets = label_list[i*num_val_samples:]
        val_label = audio_label[i*num_val_samples:]
        
        train_data = data[:i*num_val_samples]
        train_targes = label_list[:i*num_val_samples]
        train_label = audio_label[:i*num_val_samples]

        train_targes2 = label2[:i*num_val_samples]
        val_targets2 = label2[i*num_val_samples:]

        train_targes3 = label3[:i*num_val_samples]
        val_targets3 = label3[i*num_val_samples:]

        train_label_dict.append(train_targes2)
        train_label_dict.append(train_targes3)

        val_label_dict.append(val_targets2)
        val_label_dict.append(val_targets3)
    '''
    train_label_dict = dict(zip(train_targes2, train_targes3))
    val_label_dict = dict(zip(val_targets2, val_targets3))
    '''

    pickle.dump(train_label_dict, tra_files)
    pickle.dump(val_label_dict, val_files)

    tra_files.close()
    val_files.close()
    print('===============================')
    print(np.array(train_data).shape)
    train_data = np.transpose(train_data,(0,1,3,2))
    np.save(save_path + '//' +'tra_data.npy', train_data)
    np.save(save_path + '//' +'tra_targets.npy', train_label)

    print(np.array(train_data).shape)
    print(np.array(train_targes).shape)
    val_data = np.transpose(val_data,(0,1,3,2))
    np.save(save_path + '//' +'val_data.npy', val_data)
    np.save(save_path + '//' +'val_targets.npy', val_label)

if __name__ == '__main__':
    mel_phase_fft(0,'./')

