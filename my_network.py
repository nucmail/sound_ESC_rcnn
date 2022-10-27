import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Reshape, GlobalMaxPooling2D, Flatten,Dropout
from tensorflow.keras.layers import AveragePooling2D, Input, concatenate, Lambda, Dense, BatchNormalization, Add, multiply, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


#network definition
def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,learn_bn = True,wd=1e-4,use_relu=True):

    x = inputs
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',kernel_initializer='he_normal',
                  kernel_regularizer=l2(wd),use_bias=False)(x)
    return x

def pad_depth(inputs, desired_channels):
    from keras import backend as K
    y = K.zeros_like(inputs, name='pad_depth1')
    return y

def My_freq_split1(x):
    #from keras import backend as K
    return x[:,0:64,:,:]



def channel_attention(input_feature, ratio=8):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature.shape[-1]
	
	shared_layer_one = Dense(channel//ratio,
							 kernel_initializer='he_normal',
							 activation = 'relu',
							 use_bias=True,
							 bias_initializer='zeros')

	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	avg_pool = Reshape((1,1,channel))(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	
	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool.shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('hard_sigmoid')(cbam_feature)
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
	
	return multiply([input_feature, cbam_feature])

def spatial_attention1(input_feature):
	kernel_size = 3
	if K.image_data_format() == "channels_first":
		channel = input_feature.shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature.shape[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool.shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool.shape[-1] == 1
	avg_pool_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					activation = 'hard_sigmoid',
					strides=1,
					padding='same',
					kernel_initializer='he_normal',
					use_bias=False)(avg_pool)
	max_pool_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					activation = 'hard_sigmoid',
					strides=1,
					padding='same',
					kernel_initializer='he_normal',
					use_bias=False)(max_pool)				
	concat = Concatenate(axis=3)([avg_pool_feature, max_pool_feature])
	assert concat.shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					activation = 'hard_sigmoid',
					strides=1,
					padding='same',
					kernel_initializer='he_normal',
					use_bias=False)(concat)
	assert cbam_feature.shape[-1] == 1
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
		
	return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
	kernel_size = 3
	if K.image_data_format() == "channels_first":
		channel = input_feature.shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature.shape[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	avg_pool = Conv2D(filters = 1,
					kernel_size=kernel_size,
					activation = 'hard_sigmoid',
					strides=1,
					padding='same',
					kernel_initializer='he_normal',
					use_bias=False)(avg_pool)
	assert avg_pool.shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	max_pool = Conv2D(filters = 1,
					kernel_size=kernel_size,
					activation = 'hard_sigmoid',
					strides=1,
					padding='same',
					kernel_initializer='he_normal',
					use_bias=False)(max_pool)
	assert max_pool.shape[-1] == 1			
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	assert concat.shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					activation = 'hard_sigmoid',
					strides=1,
					padding='same',
					kernel_initializer='he_normal',
					use_bias=False)(concat)
	assert cbam_feature.shape[-1] == 1
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
		
	return multiply([input_feature, cbam_feature])

def cbam_block(cbam_feature,ratio=8):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in CBAM: Convolutional Block Attention Module.
	"""
	
	cbam_feature = spatial_attention(cbam_feature, )
	cbam_feature = channel_attention(cbam_feature, ratio)
	return cbam_feature

def model_resnet(num_classes,input_shape =[128,None,6], num_filters =24,wd=1e-3):
    
    My_wd = wd #this is 5e-3 in matlab, so quite large
    num_res_blocks=2
    
    
    inputs = Input(shape=input_shape)
    
    residual = Conv2D(1, kernel_size = (1, 1), padding = 'same')(inputs)
    residual = BatchNormalization(axis = 3)(residual)
    cbam = cbam_block(residual)
    inputw = keras.layers.add([inputs, residual, cbam])
    #split up frequency into two branches
    # Split1=  Lambda(My_freq_split1)(inputw)


    ResidualPath1 = resnet_layer(inputs=inputw,
                     num_filters=num_filters,
                     strides=[1,2],
                     learn_bn = True,
                     wd=My_wd,
                     use_relu = False)

    # Instantiate the stack of residual units
    for stack in range(4):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = [1,2]  # downsample
            ConvPath1 = resnet_layer(inputs=ResidualPath1,
                             num_filters=num_filters,
                             strides=strides,
                             learn_bn = False,
                             wd=My_wd,
                             use_relu = True)
            ConvPath1 = resnet_layer(inputs=ConvPath1,
                             num_filters=num_filters,
                             strides=1,
                             learn_bn = False,
                             wd=My_wd,
                             use_relu = True)


            if stack > 0 and res_block == 0:  
                # first layer but not first stack: this is where we have gone up in channels and down in feature map size
                #so need to account for this in the residual path
                #average pool and downsample the residual path
                
                ResidualPath1 = AveragePooling2D(pool_size=(3, 3), strides=[1,2], padding='same')(ResidualPath1)
                
                #zero pad to increase channels
                desired_channels = ConvPath1.shape.as_list()[-1]

                Padding1=Lambda(pad_depth,arguments={'desired_channels':desired_channels})(ResidualPath1)
                ResidualPath1 = keras.layers.Concatenate(axis=-1)([ResidualPath1,Padding1])

            ResidualPath1 = keras.layers.add([ConvPath1,ResidualPath1])
        #when we are here, we double the number of filters    
        num_filters *= 2
        
    OutputPath = resnet_layer(inputs=ResidualPath1,
                             num_filters=2*num_filters,
                              kernel_size=1,
                             strides=1,
                             learn_bn = False,
                             wd=My_wd,
                             use_relu = True)
        
    #output layers after last sum
    OutputPath = resnet_layer(inputs=OutputPath,
                     num_filters=50,
                     strides = 1,
                     kernel_size=1,
                     learn_bn = False,
                     wd=My_wd,
                      use_relu=False)
    OutputPath = BatchNormalization(center=False, scale=False)(OutputPath)
    print(OutputPath.shape)
    OutputPath = GlobalAveragePooling2D()(OutputPath)
    OutputPath = Activation('softmax')(OutputPath)
    print(OutputPath.shape)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=OutputPath)
    return model
