import os
import numpy as np
import librosa
import librosa.display
import soundfile as sound
import scipy
import tensorflow as tf
from DCASE2019_network import model_resnet

data_zores = np.zeros((64,64,216,3), dtype=float)

model = model_resnet(50, [64,216,3])

acc = model(data_zores)


