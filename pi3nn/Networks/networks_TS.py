''' Network structure and training steps --- Time series data '''

import tensorflow as tf
import numpy as np
import random
# Force using CPU globally by hiding GPU(s)
# tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import Model, layers
import warnings
import copy
from sklearn.metrics import r2_score
from pi3nn.Utils.Utils import CL_losses


import os
num_threads = 1
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)



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


### LSTM train steps
class CL_LSTM_train_steps:
	def __init__(self, net_lstm_mean, net_lstm_up=None, net_lstm_down=None, 
				optimizers_lstm=['Adam','Adam', 'Adam'], 
				lr_lstm=[0.01, 0.01, 0.01],                 
				exponential_decay=False,
				decay_steps=None,
				decay_rate=None,
				**kwargs):

		self.losses = CL_losses()
		# losses.MSE()

		self.exponential_decay = exponential_decay
		self.decay_steps = decay_steps
		self.decay_rate = decay_rate
		self.lr_lstm = lr_lstm
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

		self.net_lstm_mean = net_lstm_mean
		if exponential_decay is False:
			if optimizers_lstm[0] == 'Adam':
				self.optimizer_lstm_mean = tf.keras.optimizers.Adam(learning_rate=self.lr_lstm[0])
			elif optimizers_lstm[0] == 'SGD':
				self.optimizer_lstm_mean = tf.keras.optimizers.SGD(learning_rate=self.lr_lstm[0])
		else:
			self.global_step_lstm_0 = tf.Variable(0, trainable=False)
			decayed_l_rate_lstm_0 = tf.compat.v1.train.exponential_decay(self.lr_lstm[0], self.global_step_lstm_0, decay_steps=decay_steps, decay_rate=decay_rate, staircase=False)
			if optimizers_lstm[0] == 'Adam':
				self.optimizer_lstm_mean = tf.keras.optimizers.Adam(learning_rate=decayed_l_rate_lstm_0)
			elif optimizers_lstm[0] == 'SGD':
				self.optimizer_lstm_mean = tf.keras.optimizers.SGD(learning_rate=decayed_l_rate_lstm_0)

		if net_lstm_up is not None:
			self.net_lstm_up = net_lstm_up
			if exponential_decay is False:
				if optimizers_lstm[1] == 'Adam':
					self.optimizer_lstm_up = tf.keras.optimizers.Adam(learning_rate=self.lr_lstm[1])
				elif optimizers_lstm[1] == 'SGD':
					self.optimizer_lstm_up = tf.keras.optimizers.SGD(learning_rate=self.lr_lstm[1])
			else:
				self.global_step_lstm_1 = tf.Variable(0, trainable=False)
				decayed_l_rate_lstm_1 = tf.compat.v1.train.exponential_decay(self.lr_lstm[1], self.global_step_lstm_1, decay_steps=decay_steps, decay_rate=decay_rate, staircase=False)
				if optimizers_lstm[1] == 'Adam':
					self.optimizer_lstm_up = tf.keras.optimizers.Adam(learning_rate=decayed_l_rate_lstm_1)
				elif optimizers_lstm[1] == 'SGD':
					self.optimizer_lstm_up = tf.keras.optimizers.SGD(learning_rate=decayed_l_rate_lstm_1)

		if net_lstm_down is not None:
			self.net_lstm_down = net_lstm_down
			if exponential_decay is False:
				if optimizers_lstm[2] == 'Adam':
					self.optimizer_lstm_down = tf.keras.optimizers.Adam(learning_rate=self.lr_lstm[2])
				elif optimizers_lstm[2] == 'SGD':
					self.optimizer_lstm_down = tf.keras.optimizers.SGD(learning_rate=self.lr_lstm[2])
			else:
				self.global_step_lstm_2 = tf.Variable(0, trainable=False)
				decayed_l_rate_lstm_2 = tf.compat.v1.train.exponential_decay(self.lr_lstm[2], self.global_step_lstm_2, decay_steps=decay_steps, decay_rate=decay_rate, staircase=False)
				if optimizers_lstm[2] == 'Adam':
					self.optimizer_lstm_down = tf.keras.optimizers.Adam(learning_rate=decayed_l_rate_lstm_2)
				elif optimizers_lstm[2] == 'SGD':
					self.optimizer_lstm_down = tf.keras.optimizers.SGD(learning_rate=decayed_l_rate_lstm_2)


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
	''' Training for LSTM MEAN '''
	##############################
	@tf.function
	def train_step_mean_LSTM(self, xtrain, ytrain, xTest, yTest):
		with tf.GradientTape() as tape:
			train_predictions = self.net_lstm_mean(xtrain, training=True)
			train_loss = self.criterion_mean(ytrain, train_predictions)
			''' Add regularization losses '''
			train_loss += self.add_model_regularizer_loss(self.net_lstm_mean)
			test_predictions = self.net_lstm_mean(xTest, training=False)
			test_loss = self.criterion_mean(yTest, test_predictions)
		gradients = tape.gradient(train_loss, self.net_lstm_mean.trainable_variables)
		self.optimizer_lstm_mean.apply_gradients(zip(gradients, self.net_lstm_mean.trainable_variables))
		self.train_loss_mean(train_loss)
		self.test_loss_mean(test_loss)

		if self.exponential_decay:
			self.global_step_lstm_0.assign_add(1)

	@tf.function
	def batch_train_step_mean_LSTM(self, x_batch_train, y_batch_train, weights=None):
		with tf.GradientTape() as tape:
			batch_train_predictions = self.net_lstm_mean(x_batch_train, training=True)

			if weights is not None:
				batch_train_loss = self.losses.wMSE(y_batch_train, batch_train_predictions, weights)
			else:
				batch_train_loss = self.criterion_mean(y_batch_train, batch_train_predictions)
			''' Add regularization losses '''
			batch_train_loss += self.add_model_regularizer_loss(self.net_lstm_mean)
		gradients = tape.gradient(batch_train_loss, self.net_lstm_mean.trainable_variables)
		self.optimizer_lstm_mean.apply_gradients(zip(gradients, self.net_lstm_mean.trainable_variables))
		self.train_loss_mean(batch_train_loss)

	@tf.function
	def batch_valid_step_mean_LSTM(self, x_batch_valid, y_batch_valid):
		with tf.GradientTape() as tape:
			batch_valid_predictions = self.net_lstm_mean(x_batch_valid, training=False)
			batch_valid_loss = self.criterion_mean(y_batch_valid, batch_valid_predictions)
		self.valid_loss_mean(batch_valid_loss)

	@tf.function
	def batch_test_step_mean_LSTM(self, x_batch_test, y_batch_test):
		with tf.GradientTape() as tape:
			batch_test_predictions = self.net_lstm_mean(x_batch_test, training=False)
			batch_test_loss = self.criterion_mean(y_batch_test, batch_test_predictions)
		self.test_loss_mean(batch_test_loss)

	@tf.function
	def test_step_mean_LSTM(self, x_test, y_test):
		with tf.GradientTape() as tape:
			test_predictions = self.net_lstm_mean(x_test, training=False)
			test_loss = self.criterion_mean_all(y_test, test_predictions)
		return test_predictions, test_loss


	#################################################
	''' Training for LSTM UP --- train DENSE only '''
	#################################################
	@tf.function
	def train_step_up_LSTM(self, xtrain, ytrain, xTest, yTest):
		with tf.GradientTape() as tape:
			train_predictions = self.net_lstm_up(xtrain, training=True)
			train_loss = self.criterion_PI(ytrain, train_predictions)
			''' Add regularization losses '''
			train_loss += self.add_model_regularizer_loss(self.net_lstm_up)
			test_predictions = self.net_lstm_up(xTest, training=False)
			test_loss = self.criterion_PI(yTest, test_predictions)
		gradients = tape.gradient(train_loss, self.net_lstm_up.trainable_variables)
		self.optimizer_lstm_up.apply_gradients(zip(gradients, self.net_lstm_up.trainable_variables))
		self.train_loss_up(train_loss)
		self.test_loss_up(test_loss)

		if self.exponential_decay:
			self.global_step_lstm_1.assign_add(1)

	@tf.function
	def batch_train_step_up_LSTM(self, x_batch_train, y_batch_train, freeze_lstm=False):
		with tf.GradientTape() as tape:
			batch_train_predictions = self.net_lstm_up(x_batch_train, training=True)
			batch_train_loss = self.criterion_PI(y_batch_train, batch_train_predictions)
			''' Add regularization losses '''
			batch_train_loss += self.add_model_regularizer_loss(self.net_lstm_up)
		gradients = tape.gradient(batch_train_loss, self.net_lstm_up.trainable_variables)
		self.optimizer_lstm_up.apply_gradients(zip(gradients, self.net_lstm_up.trainable_variables))
		self.train_loss_up(batch_train_loss)

		if freeze_lstm:
			self.net_lstm_up.freeze_lstm()

	@tf.function
	def batch_valid_step_up_LSTM(self, x_batch_valid, y_batch_valid):
		with tf.GradientTape() as tape:
			batch_valid_predictions = self.net_lstm_up(x_batch_valid, training=False)
			batch_valid_loss = self.criterion_PI(y_batch_valid, batch_valid_predictions)
		self.valid_loss_up(batch_valid_loss)

	@tf.function
	def batch_test_step_up_LSTM(self, x_batch_test, y_batch_test):
		with tf.GradientTape() as tape:
			batch_test_predictions = self.net_lstm_up(x_batch_test, training=False)
			batch_test_loss = self.criterion_PI(y_batch_test, batch_test_predictions)
		self.test_loss_up(batch_test_loss)

	###################################################
	''' Training for LSTM DOWN --- train DENSE only '''
	###################################################
	@tf.function
	def train_step_down_LSTM(self, xtrain, ytrain, xTest, yTest):
		with tf.GradientTape() as tape:
			train_predictions = self.net_lstm_down(xtrain, training=True)
			train_loss = self.criterion_PI(ytrain, train_predictions)
			''' Add regularization losses '''
			train_loss += self.add_model_regularizer_loss(self.net_lstm_down)
			test_predictions = self.net_lstm_down(xTest, training=False)
			test_loss = self.criterion_PI(yTest, test_predictions)
		gradients = tape.gradient(train_loss, self.net_lstm_down.trainable_variables)
		self.optimizer_lstm_down.apply_gradients(zip(gradients, self.net_lstm_down.trainable_variables))
		self.train_loss_down(train_loss)
		self.test_loss_down(test_loss)

		if self.exponential_decay:
			self.global_step_lstm_2.assign_add(1)

	@tf.function
	def batch_train_step_down_LSTM(self, x_batch_train, y_batch_train, freeze_lstm=False):
		with tf.GradientTape() as tape:
			batch_train_predictions = self.net_lstm_down(x_batch_train, training=True)
			batch_train_loss = self.criterion_PI(y_batch_train, batch_train_predictions)
			''' Add regularization losses '''
			batch_train_loss += self.add_model_regularizer_loss(self.net_lstm_down)
		gradients = tape.gradient(batch_train_loss, self.net_lstm_down.trainable_variables)
		self.optimizer_lstm_down.apply_gradients(zip(gradients, self.net_lstm_down.trainable_variables))
		self.train_loss_down(batch_train_loss)

		if freeze_lstm:
			self.net_lstm_down.freeze_lstm()

	@tf.function
	def batch_valid_step_down_LSTM(self, x_batch_valid, y_batch_valid):
		with tf.GradientTape() as tape:
			batch_valid_predictions = self.net_lstm_down(x_batch_valid, training=False)
			batch_valid_loss = self.criterion_PI(y_batch_valid, batch_valid_predictions)
		self.valid_loss_down(batch_valid_loss)

	@tf.function
	def batch_test_step_down_LSTM(self, x_batch_test, y_batch_test):
		with tf.GradientTape() as tape:
			batch_test_predictions = self.net_lstm_down(x_batch_test, training=False)
			batch_test_loss = self.criterion_PI(y_batch_test, batch_test_predictions)
		self.test_loss_down(batch_test_loss)
