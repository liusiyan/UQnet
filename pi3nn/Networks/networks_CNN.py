''' Network structure and training steps --- Time series data '''

import tensorflow as tf
import numpy as np
import random
# Force using CPU globally by hiding GPU(s)
# tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.layers import Dense, Conv3D, Conv3DTranspose, MaxPool3D, Flatten
from tensorflow.keras import Model, layers
import warnings
import copy
from sklearn.metrics import r2_score
from pi3nn.Utils.Utils import CL_losses


import os
num_threads = 8
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)

# Force using CPU globally by hiding GPU(s), comment the line of code below to enable GPU
# tf.config.set_visible_devices([], 'GPU')



# ### PyTroch Conv3d
# in_channels (int) – Number of channels in the input image
# out_channels (int) – Number of channels produced by the convolution
# kernel_size (int or tuple) – Size of the convolving kernel
# stride (int or tuple, optional) – Stride of the convolution. Default: 1
# padding (int, tuple or str, optional) – Padding added to all six sides of the input. Default: 0
# padding_mode (string, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
# dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
# groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
# bias (bool, optional) – If True, adds a learnable bias to the output. Default: True

# ### PyTorch MaxPool3d
# kernel_size – the size of the window to take a max over
# stride – the stride of the window. Default value is kernel_size
# padding – implicit zero padding to be added on all three sides
# dilation – a parameter that controls the stride of elements in the window
# return_indices – if True, will return the max indices along with the outputs. Useful for torch.nn.MaxUnpool3d later
# ceil_mode – when True, will use ceil instead of floor to compute the output shape

# tf.keras.layers.Conv3D(
#     filters,
#     kernel_size,
#     strides=(1, 1, 1),
#     padding='valid',
#     data_format=None,
#     dilation_rate=(1, 1, 1),
#     groups=1,
#     activation=None,
#     use_bias=True,
#     kernel_initializer='glorot_uniform',
#     bias_initializer='zeros',
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs
# )

# tf.keras.layers.MaxPool3D(
#     pool_size=(2, 2, 2),
#     strides=None,
#     padding='valid',
#     data_format=None,
#     **kwargs
# )


## Pure CNN networks
class CNN_pure_TF2(Model):
	def __init__(self, configs, num_outputs):
		super(CNN_pure_TF2, self).__init__()
		self.configs = configs

		### provide argument input_shape for first Convolution layer, 
		### e.g. (128,128,128,1) 128*128*128 volume with single channel
		### (4,128,128,128,1) batch size is 4  
		### batch_size = configs['batch_size']

		# self.conv_1 = Conv3D(6, 6, strides=(2,2,2), padding='valid', activation='relu', input_shape=(configs['batch_size'],128,128,128, 1), name='conv_1')
		# self.maxpool_1 = MaxPool3D(pool_size=(2,2,2), strides=(1,1,1), name='maxpool_1')
		# self.conv_2 = Conv3D(1, 6, strides=(2,2,2), padding='valid', activation='relu', name='conv_2')
		# self.maxpool_2 = MaxPool3D(pool_size=(2,2,2), strides=(1,1,1), name='maxpool_2')
		# self.flatten = Flatten()
		# self.fc_1 = Dense(128, activation='relu', name='fc_1')
		# self.fc_2 = Dense(32, activation='relu', name='fc_2')
		# self.outputLayer = Dense(num_outputs, name='dense_out')


		# print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
		# print(self.configs['batch_size'])
		# print(type(self.configs['batch_size']))

		# exit()

		# self.configs['batch_size']

		self.conv_1 = Conv3D(6, 3, strides=(2,2,2), padding='valid', activation='relu', input_shape=(self.configs['batch_size'],128,128,128,1), name='conv_1')
		self.maxpool_1 = MaxPool3D(pool_size=(2,2,2), strides=(1,1,1), name='maxpool_1')
		# self.conv_2 = Conv3D(1, 6, strides=(2,2,2), padding='valid', activation='relu', name='conv_2')
		# self.maxpool_2 = MaxPool3D(pool_size=(2,2,2), strides=(1,1,1), name='maxpool_2')
		self.flatten = Flatten()
		self.fc_1 = Dense(128, activation='relu', name='fc_1')
		self.fc_2 = Dense(32, activation='relu', name='fc_2')
		self.outputLayer = Dense(num_outputs, name='dense_out')

	def call(self, x):
		x = self.conv_1(x)
		x = self.maxpool_1(x)

		# x = self.conv_2(x)
		# x = self.maxpool_2(2)

		x = self.flatten(x)
		x = self.fc_1(x)
		x = self.fc_2(x)
		x = self.outputLayer(x)
		return x

	### test full prediction with fully connected layers
	def test_manual_pred_full(self, x):
		x = self.conv_1(x)
		x = self.maxpool_1(x)
		x = self.flatten(x)
		x = self.fc_2(x)
		x = self.outputLayer(x)
		return x

	### test prediction with only CNN associated layers
	def test_manual_pred_CNN(self, x):
		x = self.conv_1(x)
		x = self.maxpool_1(x)
		x = self.flatten(x)
		return x



## Pure LSTM with NO hidden dense layers
class LSTM_pure_TF2(Model):
	def __init__(self, configs, num_outputs):
		super(LSTM_pure_TF2, self).__init__()
		self.configs = configs
		self.lstm_mean = LSTM(self.configs['LSTM_nodes'], return_sequences=False, name='lstm')
		self.outputLayer = Dense(num_outputs, name='dense_out')

	def call(self, x):
		x = self.lstm_mean(x)
		x = self.outputLayer(x)
		return x


# tf.keras.backend.set_floatx('float64') ## to avoid TF casting prediction to float32
''' Network definition '''
### LSTM
class LSTM_mean_TF2(Model):
	def __init__(self, configs, num_outputs):
		super(LSTM_mean_TF2, self).__init__()
		self.configs = copy.deepcopy(configs)
		self.num_nodes_list = list(self.configs['num_neurons_mean'])
		# self.inputLayer = Dense(num_inputs, activation='linear')
		self.lstm_mean = LSTM(self.configs['LSTM_nodes'], return_sequences=False, name='lstm')    #     stateful=True, recurrent_initializer='glorot_uniform'),
		# initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.1)
		self.fcs = []
		for i in range(len(self.num_nodes_list)):
			# self.fcs.append(Dense(self.num_nodes_list[i], activation='relu'))
			self.fcs.append(Dense(self.num_nodes_list[i], activation='relu'))
			# self.fcs.append(
			# 	Dense(self.num_nodes_list[i], activation='relu',
			# 		  kernel_initializer=initializer,
					  # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.02)
			# 		  )
			# 	)
		self.outputLayer = Dense(num_outputs, name='dense_out')

	def call(self, x):
		# x = self.inputLayer(x)
		x = self.lstm_mean(x)
		for i in range(len(self.num_nodes_list)):
			x = self.fcs[i](x)
		x = self.outputLayer(x)
		return x

### The Upper and lower bound prediction networks take the LSTM parameters (weights) from the 
### training restuls from the 'LSTM_mean_TF2' model and only train the output Dense layer(s) 

class LSTM_PI_TF2(Model):
	def __init__(self, configs, num_outputs, net=None):
		super(LSTM_PI_TF2, self).__init__()
		self.configs = configs
		self.configs = copy.deepcopy(configs)

		# random.seed(self.configs['seed'])
		# np.random.seed(self.configs['seed'])
		# tf.random.set_seed(self.configs['seed'])
		if net == 'up':
			self.num_nodes_list = list(self.configs['num_neurons_up'])
			bias = self.configs['bias_up']
		elif net == 'down':
			self.num_nodes_list = list(self.configs['num_neurons_down'])
			bias = self.configs['bias_down']	
		# self.inputLayer = Dense(num_inputs, activation='linear')
		# initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.1)
		self.lstm_PI = LSTM(self.configs['LSTM_nodes'], return_sequences=False)    #     stateful=True, recurrent_initializer='glorot_uniform'),
		self.fcs = []
		for i in range(len(self.num_nodes_list)):
			self.fcs.append(
				Dense(self.num_nodes_list[i], activation='relu',
					  # kernel_initializer=initializer,
					  kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0., l2=0.)
					  )
				)

		self.outputLayer = Dense(num_outputs)

		if bias is None:
			self.custom_bias = tf.Variable([0.0]) # dtype=tf.float64
		else:
			self.custom_bias = tf.Variable([bias]) # dtype=tf.float64

	def call(self, x):
		x = self.lstm_PI(x)
		for i in range(len(self.num_nodes_list)):
			x = self.fcs[i](x)
		x = self.outputLayer(x)
		x = tf.nn.bias_add(x, self.custom_bias)
		x = tf.math.sqrt(tf.math.square(x) + 1e-8)  
		return x

	def freeze_lstm(self):
		self.lstm_PI.trainable = False







### CNN train steps
class CL_CNN_train_steps:
	def __init__(self, net_cnn_mean, net_cnn_up=None, net_cnn_down=None, 
				optimizers_cnn=['Adam','Adam', 'Adam'], 
				lr_cnn=[0.01, 0.01, 0.01],                 
				exponential_decay=False,
				decay_steps=None,
				decay_rate=None,
				**kwargs):

		self.losses = CL_losses()
		# losses.MSE()

		self.exponential_decay = exponential_decay
		self.decay_steps = decay_steps
		self.decay_rate = decay_rate
		self.lr_cnn = lr_cnn
		self.criterion_mean = tf.keras.losses.MeanSquaredError()
		self.criterion_PI = tf.keras.losses.MeanSquaredError()
		self.criterion_mean_all = tf.keras.losses.MeanSquaredError()

		self.train_loss_mean = tf.keras.metrics.Mean(name='train_loss_mean')
		self.train_loss_up = tf.keras.metrics.Mean(name='train_loss_up')
		self.train_loss_down = tf.keras.metrics.Mean(name='train_loss_down')

		self.valid_loss_mean = tf.keras.metrics.Mean(name='valid_loss_mean')
		self.valid_loss_up = tf.keras.metrics.Mean(name='valid_loss_up')
		self.valid_loss_down = tf.keras.metrics.Mean(name='valid_loss_down')

		self.test_loss_mean = tf.keras.metrics.Mean(name='test_loss_mean')
		self.test_loss_up = tf.keras.metrics.Mean(name='test_loss_up')
		self.test_loss_down = tf.keras.metrics.Mean(name='test_loss_down')

		self.test_loss_mean_all = tf.keras.metrics.Mean(name='test_loss_mean_all')

		self.net_cnn_mean = net_cnn_mean
		if exponential_decay is False:
			if optimizers_cnn[0] == 'Adam':
				self.optimizer_cnn_mean = tf.keras.optimizers.Adam(learning_rate=self.lr_cnn[0])
			elif optimizers_cnn[0] == 'SGD':
				self.optimizer_cnnmean = tf.keras.optimizers.SGD(learning_rate=self.lr_cnn[0])
		else:
			self.global_step_cnn_0 = tf.Variable(0, trainable=False)
			decayed_l_rate_cnn_0 = tf.compat.v1.train.exponential_decay(self.lr_cnn[0], self.global_step_cnn_0, decay_steps=decay_steps, decay_rate=decay_rate, staircase=False)
			if optimizers_cnn[0] == 'Adam':
				self.optimizer_cnn_mean = tf.keras.optimizers.Adam(learning_rate=decayed_l_rate_cnn_0)
			elif optimizers_cnn[0] == 'SGD':
				self.optimizer_cnn_mean = tf.keras.optimizers.SGD(learning_rate=decayed_l_rate_cnn_0)

		if net_cnn_up is not None:
			self.net_cnn_up = net_cnn_up
			if exponential_decay is False:
				if optimizers_cnn[1] == 'Adam':
					self.optimizer_cnn_up = tf.keras.optimizers.Adam(learning_rate=self.lr_cnn[1])
				elif optimizers_cnn[1] == 'SGD':
					self.optimizer_cnn_up = tf.keras.optimizers.SGD(learning_rate=self.lr_cnn[1])
			else:
				self.global_step_cnn_1 = tf.Variable(0, trainable=False)
				decayed_l_rate_cnn_1 = tf.compat.v1.train.exponential_decay(self.lr_cnn[1], self.global_step_cnn_1, decay_steps=decay_steps, decay_rate=decay_rate, staircase=False)
				if optimizers_cnn[1] == 'Adam':
					self.optimizer_cnn_up = tf.keras.optimizers.Adam(learning_rate=decayed_l_rate_cnn_1)
				elif optimizers_cnn[1] == 'SGD':
					self.optimizer_cnn_up = tf.keras.optimizers.SGD(learning_rate=decayed_l_rate_cnn_1)

		if net_cnn_down is not None:
			self.net_cnn_down = net_cnn_down
			if exponential_decay is False:
				if optimizers_cnn[2] == 'Adam':
					self.optimizer_cnn_down = tf.keras.optimizers.Adam(learning_rate=self.lr_cnn[2])
				elif optimizers_cnn[2] == 'SGD':
					self.optimizer_cnn_down = tf.keras.optimizers.SGD(learning_rate=self.lr_cnn[2])
			else:
				self.global_step_cnn_2 = tf.Variable(0, trainable=False)
				decayed_l_rate_cnn_2 = tf.compat.v1.train.exponential_decay(self.lr_cnn[2], self.global_step_cnn_2, decay_steps=decay_steps, decay_rate=decay_rate, staircase=False)
				if optimizers_cnn[2] == 'Adam':
					self.optimizer_cnn_down = tf.keras.optimizers.Adam(learning_rate=decayed_l_rate_cnn_2)
				elif optimizers_cnn[2] == 'SGD':
					self.optimizer_cnn_down = tf.keras.optimizers.SGD(learning_rate=decayed_l_rate_cnn_2)


	def add_model_regularizer_loss(self, model):
		loss = 0
		for l in model.layers:
			# if hasattr(l, 'layers') and l.layers:  # the layer itself is a model
			#     loss += add_model_loss(l)
			if hasattr(l, 'kernel_regularizer') and l.kernel_regularizer:
				loss += l.kernel_regularizer(l.kernel)
			if hasattr(l, 'bias_regularizer') and l.bias_regularizer:
				loss += l.bias_regularizer(l.bias)
		return loss

	##############################
	''' Training for CNN MEAN '''
	##############################
	@tf.function
	def train_step_mean_CNN(self, xtrain, ytrain, xTest, yTest):
		with tf.GradientTape() as tape:
			train_predictions = self.net_cnn_mean(xtrain, training=True)
			train_loss = self.criterion_mean(ytrain, train_predictions)
			''' Add regularization losses '''
			train_loss += self.add_model_regularizer_loss(self.net_cnn_mean)
			test_predictions = self.net_cnn_mean(xTest, training=False)
			test_loss = self.criterion_mean(yTest, test_predictions)
		gradients = tape.gradient(train_loss, self.net_cnn_mean.trainable_variables)
		self.optimizer_cnn_mean.apply_gradients(zip(gradients, self.net_cnn_mean.trainable_variables))
		self.train_loss_mean(train_loss)
		self.test_loss_mean(test_loss)

		if self.exponential_decay:
			self.global_step_cnn_0.assign_add(1)

	@tf.function
	def batch_train_step_mean_CNN(self, x_batch_train, y_batch_train, weights=None):
		with tf.GradientTape() as tape:
			batch_train_predictions = self.net_cnn_mean(x_batch_train, training=True)
			if weights is not None:
				batch_train_loss = self.losses.wMSE(y_batch_train, batch_train_predictions, weights)
			else:
				batch_train_loss = self.criterion_mean(y_batch_train, batch_train_predictions)
				# print(type(y_batch_train))
				# print(y_batch_train)
				# print(type(batch_train_predictions))
				# print(batch_train_predictions)
				# tmp_r2 = r2_score(y_batch_train.numpy(), batch_train_predictions.numpy())
				# print('--- batch train loss: {}, R2: {}'.format(batch_train_loss, tmp_r2))
			''' Add regularization losses '''
			batch_train_loss += self.add_model_regularizer_loss(self.net_cnn_mean)
		gradients = tape.gradient(batch_train_loss, self.net_cnn_mean.trainable_variables)
		self.optimizer_cnn_mean.apply_gradients(zip(gradients, self.net_cnn_mean.trainable_variables))
		self.train_loss_mean(batch_train_loss)

	@tf.function
	def batch_valid_step_mean_CNN(self, x_batch_valid, y_batch_valid):
		with tf.GradientTape() as tape:
			batch_valid_predictions = self.net_cnn_mean(x_batch_valid, training=False)
			batch_valid_loss = self.criterion_mean(y_batch_valid, batch_valid_predictions)
		self.valid_loss_mean(batch_valid_loss)

	@tf.function
	def batch_test_step_mean_CNN(self, x_batch_test, y_batch_test):
		with tf.GradientTape() as tape:
			batch_test_predictions = self.net_cnn_mean(x_batch_test, training=False)
			batch_test_loss = self.criterion_mean(y_batch_test, batch_test_predictions)
		self.test_loss_mean(batch_test_loss)

	@tf.function
	def test_step_mean_CNN(self, x_test, y_test):
		with tf.GradientTape() as tape:
			test_predictions = self.net_cnn_mean(x_test, training=False)
			test_loss = self.criterion_mean_all(y_test, test_predictions)
		return test_predictions, test_loss


	#################################################
	''' Training for CNN UP --- train DENSE only '''
	#################################################
	@tf.function
	def train_step_up_CNN(self, xtrain, ytrain, xTest, yTest):
		with tf.GradientTape() as tape:
			train_predictions = self.net_cnn_up(xtrain, training=True)
			train_loss = self.criterion_PI(ytrain, train_predictions)
			''' Add regularization losses '''
			train_loss += self.add_model_regularizer_loss(self.net_cnn_up)
			test_predictions = self.net_cnn_up(xTest, training=False)
			test_loss = self.criterion_PI(yTest, test_predictions)
		gradients = tape.gradient(train_loss, self.net_cnn_up.trainable_variables)
		self.optimizer_cnn_up.apply_gradients(zip(gradients, self.net_cnn_up.trainable_variables))
		self.train_loss_up(train_loss)
		self.test_loss_up(test_loss)

		if self.exponential_decay:
			self.global_step_cnn_1.assign_add(1)

	@tf.function
	def batch_train_step_up_CNN(self, x_batch_train, y_batch_train, freeze_cnn=False):
		with tf.GradientTape() as tape:
			batch_train_predictions = self.net_cnn_up(x_batch_train, training=True)
			batch_train_loss = self.criterion_PI(y_batch_train, batch_train_predictions)
			''' Add regularization losses '''
			batch_train_loss += self.add_model_regularizer_loss(self.net_cnn_up)
		gradients = tape.gradient(batch_train_loss, self.net_cnn_up.trainable_variables)
		self.optimizer_cnn_up.apply_gradients(zip(gradients, self.net_cnn_up.trainable_variables))
		self.train_loss_up(batch_train_loss)

		if freeze_cnn:
			self.net_cnn_up.freeze_cnn()

	@tf.function
	def batch_valid_step_up_CNN(self, x_batch_valid, y_batch_valid):
		with tf.GradientTape() as tape:
			batch_valid_predictions = self.net_cnn_up(x_batch_valid, training=False)
			batch_valid_loss = self.criterion_PI(y_batch_valid, batch_valid_predictions)
		self.valid_loss_up(batch_valid_loss)

	@tf.function
	def batch_test_step_up_CNN(self, x_batch_test, y_batch_test):
		with tf.GradientTape() as tape:
			batch_test_predictions = self.net_cnn_up(x_batch_test, training=False)
			batch_test_loss = self.criterion_PI(y_batch_test, batch_test_predictions)
		self.test_loss_up(batch_test_loss)

	###################################################
	''' Training for CNN DOWN --- train DENSE only '''
	###################################################
	@tf.function
	def train_step_down_CNN(self, xtrain, ytrain, xTest, yTest):
		with tf.GradientTape() as tape:
			train_predictions = self.net_cnn_down(xtrain, training=True)
			train_loss = self.criterion_PI(ytrain, train_predictions)
			''' Add regularization losses '''
			train_loss += self.add_model_regularizer_loss(self.net_cnn_down)
			test_predictions = self.net_cnn_down(xTest, training=False)
			test_loss = self.criterion_PI(yTest, test_predictions)
		gradients = tape.gradient(train_loss, self.net_cnn_down.trainable_variables)
		self.optimizer_cnn_down.apply_gradients(zip(gradients, self.net_cnn_down.trainable_variables))
		self.train_loss_down(train_loss)
		self.test_loss_down(test_loss)

		if self.exponential_decay:
			self.global_step_cnn_2.assign_add(1)

	@tf.function
	def batch_train_step_down_CNN(self, x_batch_train, y_batch_train, freeze_cnn=False):
		with tf.GradientTape() as tape:
			batch_train_predictions = self.net_cnn_down(x_batch_train, training=True)
			batch_train_loss = self.criterion_PI(y_batch_train, batch_train_predictions)
			''' Add regularization losses '''
			batch_train_loss += self.add_model_regularizer_loss(self.net_cnn_down)
		gradients = tape.gradient(batch_train_loss, self.net_cnn_down.trainable_variables)
		self.optimizer_cnn_down.apply_gradients(zip(gradients, self.net_cnn_down.trainable_variables))
		self.train_loss_down(batch_train_loss)

		if freeze_cnn:
			self.net_cnn_down.freeze_cnn()

	@tf.function
	def batch_valid_step_down_CNN(self, x_batch_valid, y_batch_valid):
		with tf.GradientTape() as tape:
			batch_valid_predictions = self.net_cnn_down(x_batch_valid, training=False)
			batch_valid_loss = self.criterion_PI(y_batch_valid, batch_valid_predictions)
		self.valid_loss_down(batch_valid_loss)

	@tf.function
	def batch_test_step_down_CNN(self, x_batch_test, y_batch_test):
		with tf.GradientTape() as tape:
			batch_test_predictions = self.net_cnn_down(x_batch_test, training=False)
			batch_test_loss = self.criterion_PI(y_batch_test, batch_test_predictions)
		self.test_loss_down(batch_test_loss)
