import numpy as np
from test_new import model_resnet

x_train = np.load('tra_data.npy')
x_train = np.transpose(x_train,(0,3,2,1))

model_resnet(50,
             input_shape=[64, 216, 3],
             num_filters=24,
             wd=1e-3)(x_train[:10])
                         

