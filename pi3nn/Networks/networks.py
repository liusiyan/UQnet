''' Network structure and training steps '''

import tensorflow as tf
import numpy as np
# Force using CPU globally by hiding GPU(s)
# tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, layers
import warnings

from sklearn.metrics import r2_score
from pi3nn.Utils.Utils import CL_losses

import os
num_threads = 1
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

''' Network definition '''
class UQ_Net_mean_TF2(Model):
	def __init__(self, configs, num_inputs, num_outputs):
		super(UQ_Net_mean_TF2, self).__init__()
		self.configs = configs
		self.num_nodes_list = list(self.configs['num_neurons_mean'])

		self.inputLayer = Dense(num_inputs, activation='linear')
		initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.1)

		self.fcs = []
		for i in range(len(self.num_nodes_list)):
			self.fcs.append(
				Dense(self.num_nodes_list[i], activation='relu',
					  kernel_initializer=initializer,
					  kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.02)
					  )
				)

		self.outputLayer = Dense(num_outputs)

	def call(self, x):
		x = self.inputLayer(x)
		for i in range(len(self.num_nodes_list)):
			x = self.fcs[i](x)
		x = self.outputLayer(x)
		return x

class UQ_Net_std_TF2(Model):
	def __init__(self,  configs, num_inputs, num_outputs, net=None, bias=None):
		super(UQ_Net_std_TF2, self).__init__()
		self.configs = configs
		if net == 'up':
			self.num_nodes_list = list(self.configs['num_neurons_up'])
		elif net == 'down':
			self.num_nodes_list = list(self.configs['num_neurons_down'])

		self.inputLayer = Dense(num_inputs, activation='linear')
		initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.1)

		self.fcs = []
		for i in range(len(self.num_nodes_list)):
			self.fcs.append(
				Dense(self.num_nodes_list[i], activation='relu',
					  kernel_initializer=initializer,
					  kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.02) # 0.02, 0.02
					  )
				)

		self.outputLayer = Dense(num_outputs)
		if bias is None:
			self.custom_bias = tf.Variable([0.0])
		else:
			self.custom_bias = tf.Variable([bias])

	def call(self, x):
		x = self.inputLayer(x)
		for i in range(len(self.num_nodes_list)):
			x = self.fcs[i](x)
		x = self.outputLayer(x)
		x = tf.nn.bias_add(x, self.custom_bias)
		x = tf.math.sqrt(tf.math.square(x) + self.configs['a_param'])  # 0.2 1e-8 1e-10 0.2
		return x


class CL_UQ_Net_train_steps:
	def __init__(self, net_mean, net_std_up, net_std_down,
				# xTrain, yTrain, xTest=None, yTest=None,
				optimizers=['Adam', 'Adam', 'Adam'],
				lr=[0.01, 0.01, 0.01],
				exponential_decay=False,
				decay_steps=None,
				decay_rate=None
		):

		self.exponential_decay = exponential_decay
		self.decay_steps = decay_steps
		self.decay_rate = decay_rate

		self.criterion_mean = tf.keras.losses.MeanSquaredError()
		self.criterion_std = tf.keras.losses.MeanSquaredError()
		self.criterion_mean_all = tf.keras.losses.MeanSquaredError()

		# accumulate the loss and compute the mean until .reset_states()
		self.train_loss_net_mean = tf.keras.metrics.Mean(name='train_loss_net_mean')
		self.train_loss_net_std_up = tf.keras.metrics.Mean(name='train_loss_net_std_up')
		self.train_loss_net_std_down = tf.keras.metrics.Mean(name='train_loss_net_std_down')
		self.valid_loss_net_mean = tf.keras.metrics.Mean(name='valid_loss_net_mean')
		self.valid_loss_net_std_up = tf.keras.metrics.Mean(name='valid_loss_net_std_up')
		self.valid_loss_net_std_down = tf.keras.metrics.Mean(name='valid_loss_net_std_down')
		self.test_loss_net_mean = tf.keras.metrics.Mean(name='test_loss_net_mean')
		self.test_loss_net_std_up = tf.keras.metrics.Mean(name='test_loss_net_std_up')
		self.test_loss_net_std_down = tf.keras.metrics.Mean(name='test_loss_net_std_down')

		self.net_mean = net_mean
		if exponential_decay is False:
			if optimizers[0] == 'Adam':
				self.optimizer_net_mean = tf.keras.optimizers.Adam(learning_rate=lr[0])
			elif optimizers[0] == 'SGD':
				self.optimizer_net_mean = tf.keras.optimizers.SGD(learning_rate=lr[0])
		else:
			self.global_step_0 = tf.Variable(0, trainable=False)
			decayed_l_rate_0 = tf.compat.v1.train.exponential_decay(lr[0], self.global_step_0, decay_steps=decay_steps, decay_rate=decay_rate, staircase=False)
			if optimizers[0] == 'Adam':
				self.optimizer_net_mean = tf.keras.optimizers.Adam(learning_rate=decayed_l_rate_0)
			elif optimizers[0] == 'SGD':
				self.optimizer_net_mean = tf.keras.optimizers.SGD(learning_rate=decayed_l_rate_0)


		self.net_std_up = net_std_up
		if exponential_decay is False:
			if optimizers[1] == 'Adam':
				self.optimizer_net_std_up = tf.keras.optimizers.Adam(learning_rate=lr[1])
			elif optimizers[1] == 'SGD':
				self.optimizer_net_std_up = tf.keras.optimizers.SGD(learning_rate=lr[1])
		else:
			self.global_step_1 = tf.Variable(0, trainable=False)
			decayed_l_rate_1 = tf.compat.v1.train.exponential_decay(lr[1], self.global_step_1, decay_steps=decay_steps, decay_rate=decay_rate, staircase=False)
			if optimizers[1] == 'Adam':
				self.optimizer_net_std_up = tf.keras.optimizers.Adam(learning_rate=decayed_l_rate_1)
			elif optimizers[1] == 'SGD':
				self.optimizer_net_std_up = tf.keras.optimizers.SGD(learning_rate=decayed_l_rate_1)

		self.net_std_down = net_std_down
		if exponential_decay is False:
			if optimizers[2] == 'Adam':
				self.optimizer_net_std_down = tf.keras.optimizers.Adam(learning_rate=lr[2])
			elif optimizers[2] == 'SGD':
				self.optimizer_net_std_down = tf.keras.optimizers.SGD(learning_rate=lr[2])
		else:
			self.global_step_2 = tf.Variable(0, trainable=False)
			decayed_l_rate_2 = tf.compat.v1.train.exponential_decay(lr[2], self.global_step_2, decay_steps=decay_steps, decay_rate=decay_rate, staircase=False)
			if optimizers[1] == 'Adam':
				self.optimizer_net_std_down = tf.keras.optimizers.Adam(learning_rate=decayed_l_rate_2)
			elif optimizers[1] == 'SGD':
				self.optimizer_net_std_down = tf.keras.optimizers.SGD(learning_rate=decayed_l_rate_2)


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

	''' Training/validation for mean values (For Non-batch training)'''
	@tf.function
	def train_step_mean(self, xTrain, yTrain, xValid, yValid, xTest=None, yTest=None, testDataEvaluationDuringTrain=False):
		with tf.GradientTape() as tape:
			train_predictions = self.net_mean(xTrain, training=True)
			train_loss = self.criterion_mean(yTrain, train_predictions)
			''' Add regularization losses '''
			train_loss += self.add_model_regularizer_loss(self.net_mean)

			valid_predictions = self.net_mean(xValid, training=False)
			valid_loss = self.criterion_mean(yValid, valid_predictions)

			if testDataEvaluationDuringTrain:
				test_predictions = self.net_mean(xTest, training=False)
				test_loss = self.criterion_mean(yTest, test_predictions)
			else:
				test_loss = 0
		gradients = tape.gradient(train_loss, self.net_mean.trainable_variables)
		self.optimizer_net_mean.apply_gradients(zip(gradients, self.net_mean.trainable_variables))

		self.train_loss_net_mean(train_loss) # accumulate the loss and compute the mean until .reset_states()
		self.valid_loss_net_mean(valid_loss)

		if testDataEvaluationDuringTrain:
			self.test_loss_net_mean(test_loss)

		if self.exponential_decay:
			self.global_step_0.assign_add(1)

	''' Training/validation/testing for mean values (batch version) '''
	@tf.function        
	def batch_train_step_mean(self, x_batch_train, y_batch_train):
		with tf.GradientTape() as tape:
			batch_train_predictions = self.net_mean(x_batch_train, training=True)
			batch_train_loss = self.criterion_mean(y_batch_train, batch_train_predictions)
			''' Add regularization losses '''
			batch_train_loss += self.add_model_regularizer_loss(self.net_mean)
		gradients = tape.gradient(batch_train_loss, self.net_mean.trainable_variables)
		self.optimizer_net_mean.apply_gradients(zip(gradients, self.net_mean.trainable_variables))
		self.train_loss_net_mean(batch_train_loss) # store and compute the mean of batch training losses, reset in next epoch
		# print(self.optimizer_net_mean._decayed_lr(tf.float32).numpy())  # print learning rate

	@tf.function
	def batch_valid_step_mean(self, x_batch_valid, y_batch_valid):
		with tf.GradientTape() as tape:
			batch_valid_predictions = self.net_mean(x_batch_valid, training=False)
			batch_valid_loss = self.criterion_mean(y_batch_valid, batch_valid_predictions)
		self.valid_loss_net_mean(batch_valid_loss)

	@tf.function
	def batch_test_step_mean(self, x_batch_test, y_batch_test):
		with tf.GradientTape() as tape:
			batch_test_predictions = self.net_mean(x_batch_test, training=False)
			batch_test_loss = self.criterion_mean(y_batch_test, batch_test_predictions)
		self.test_loss_net_mean(batch_test_loss)

	@tf.function
	def test_step_mean(self, x_test, y_test):
		with tf.GradientTape() as tape:
			test_predictions = self.net_mean(x_test, training=False)
			test_loss = self.criterion_mean_all(y_test, test_predictions)
		return test_predictions, test_loss

	''' Training/validation for upper boundary (For Non-batch training) '''
	@tf.function
	def train_step_up(self, xTrain, yTrain, xValid, yValid, xTest=None, yTest=None, testDataEvaluationDuringTrain=False):
		with tf.GradientTape() as tape:
			train_predictions = self.net_std_up(xTrain, training=True)
			train_loss = self.criterion_std(yTrain, train_predictions)
			''' Add regularization losses '''
			train_loss += self.add_model_regularizer_loss(self.net_std_up)

			valid_predictions = self.net_std_up(xValid, training=False)
			valid_loss = self.criterion_std(yValid, valid_predictions)

			if testDataEvaluationDuringTrain:
				test_predictions = self.net_std_up(xTest, training=False)
				test_loss = self.criterion_std(yTest, test_predictions)
			else:
				test_loss = 0
		gradients = tape.gradient(train_loss, self.net_std_up.trainable_variables)
		self.optimizer_net_std_up.apply_gradients(zip(gradients, self.net_std_up.trainable_variables))

		self.train_loss_net_std_up(train_loss)
		self.valid_loss_net_std_up(valid_loss)

		if testDataEvaluationDuringTrain:
			self.test_loss_net_std_up(test_loss)

		if self.exponential_decay:
			self.global_step_1.assign_add(1)


	''' Training/validation/testing for UP values (batch version) '''
	@tf.function        
	def batch_train_step_up(self, x_batch_train, y_batch_train):
		with tf.GradientTape() as tape:
			batch_train_predictions = self.net_std_up(x_batch_train, training=True)
			batch_train_loss = self.criterion_std(y_batch_train, batch_train_predictions)
			''' Add regularization losses '''
			batch_train_loss += self.add_model_regularizer_loss(self.net_std_up)
		gradients = tape.gradient(batch_train_loss, self.net_std_up.trainable_variables)
		self.optimizer_net_std_up.apply_gradients(zip(gradients, self.net_std_up.trainable_variables))
		self.train_loss_net_std_up(batch_train_loss) # store and compute the mean of batch training losses, reset in next epoch

	@tf.function
	def batch_valid_step_up(self, x_batch_valid, y_batch_valid):
		with tf.GradientTape() as tape:
			batch_valid_predictions = self.net_std_up(x_batch_valid, training=False)
			batch_valid_loss = self.criterion_std(y_batch_valid, batch_valid_predictions)
		self.valid_loss_net_std_up(batch_valid_loss)

	@tf.function
	def batch_test_step_up(self, x_batch_test, y_batch_test):
		with tf.GradientTape() as tape:
			batch_test_predictions = self.net_std_up(x_batch_test, training=False)
			batch_test_loss = self.criterion_std(y_batch_test, batch_test_predictions)
		self.test_loss_net_std_up(batch_test_loss)


	''' Training/validation for lower boundary (For Non-batch training)'''
	@tf.function
	def train_step_down(self, xTrain, yTrain, xValid, yValid, xTest=None, yTest=None, testDataEvaluationDuringTrain=False):
		with tf.GradientTape() as tape:
			train_predictions = self.net_std_down(xTrain, training=True)
			train_loss = self.criterion_std(yTrain, train_predictions)
			''' Add regularization losses '''
			train_loss += self.add_model_regularizer_loss(self.net_std_down)

			valid_predictions = self.net_std_down(xValid, training=False)
			valid_loss = self.criterion_std(yValid, valid_predictions)

			if testDataEvaluationDuringTrain:
				test_predictions = self.net_std_down(xTest, training=False)
				test_loss = self.criterion_std(yTest, test_predictions)
			else:
				test_loss = 0
		gradients = tape.gradient(train_loss, self.net_std_down.trainable_variables)
		self.optimizer_net_std_down.apply_gradients(zip(gradients, self.net_std_down.trainable_variables))

		self.train_loss_net_std_down(train_loss)
		self.valid_loss_net_std_down(valid_loss)

		if testDataEvaluationDuringTrain:
			self.test_loss_net_std_down(test_loss)

		if self.exponential_decay:
			self.global_step_2.assign_add(1)


	''' Training/validation/testing for DOWN values (batch version) '''
	@tf.function        
	def batch_train_step_down(self, x_batch_train, y_batch_train):
		with tf.GradientTape() as tape:
			batch_train_predictions = self.net_std_down(x_batch_train, training=True)
			batch_train_loss = self.criterion_std(y_batch_train, batch_train_predictions)
			''' Add regularization losses '''
			batch_train_loss += self.add_model_regularizer_loss(self.net_std_down)
		gradients = tape.gradient(batch_train_loss, self.net_std_down.trainable_variables)
		self.optimizer_net_std_down.apply_gradients(zip(gradients, self.net_std_down.trainable_variables))
		self.train_loss_net_std_down(batch_train_loss) # store and compute the mean of batch training losses, reset in next epoch

	@tf.function
	def batch_valid_step_down(self, x_batch_valid, y_batch_valid):
		with tf.GradientTape() as tape:
			batch_valid_predictions = self.net_std_down(x_batch_valid, training=False)
			batch_valid_loss = self.criterion_std(y_batch_valid, batch_valid_predictions)
		self.valid_loss_net_std_down(batch_valid_loss)

	@tf.function
	def batch_test_step_down(self, x_batch_test, y_batch_test):
		with tf.GradientTape() as tape:
			batch_test_predictions = self.net_std_down(x_batch_test, training=False)
			batch_test_loss = self.criterion_std(y_batch_test, batch_test_predictions)
		self.test_loss_net_std_down(batch_test_loss)



