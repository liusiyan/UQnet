
''' Hyper-parameters optimizations '''


### hyperopt package used for fast hypar-parameters tuning 

import random
import numpy as np
import pandas as pd
import tensorflow as tf

##### Hyperopt hyperparameter tuning test
from hyperopt import fmin, hp, Trials, STATUS_OK, tpe, rand
from pi3nn.Networks.networks import UQ_Net_mean_TF2, UQ_Net_std_TF2
from pi3nn.Trainers.trainers import CL_trainer
from pi3nn.Utils.Utils import CL_Utils
utils = CL_Utils()

class CL_params_optimizer:
	def __init__(self, configs, xTrain, yTrain, xValid, yValid, xTest, yTest, upper_limit_iters=1000, num_trials=10, rstate=None):
		self.configs = configs
		self.target_PICP = self.configs['quantile']
		self.configs['restore_best_weights'] = True
		self.configs['batch_training'] = False ## currently unavailable, addtional work needs to be done
		self.configs['saveWeights'] = False
		self.configs['loadWeights_test'] = False
		self.num_trials = num_trials

		self.xTrain = xTrain
		self.yTrain = yTrain
		self.xValid = xValid
		self.yValid = yValid
		self.xTest = xTest
		self.yTest = yTest

		self.layer_options = [1, 2, 3] ### 1, 2, and 3 layers of network will be tested
		self.min_nodes = 1
		self.max_nodes = 128

		print('--- Auto hyper-parameters tuning...')



	def structure_params(self, num_layers, label):
		params = {}
		for n in range(num_layers):
			params['n_nodes_layer_{}_{}'.format(label, n)] = hp.randint('n_nodes_{}_{}_{}'.format(label, num_layers, n), self.min_nodes, self.max_nodes)
		return params
 

	def auto_params_optimization(self, upper_limit_iters=1000, num_trials=20, rnd_state=None):

		space = {
		'seed': hp.randint('seed', 0, 100),
		'Max_iter': hp.randint('Max_iter', 1, upper_limit_iters), 
		'lr_mean': hp.uniform('lr_mean', 0.0001, 0.01), 
		'lr_up': hp.uniform('lr_up', 0.0001, 0.01),
		'lr_down': hp.uniform('lr_down', 0.0001, 0.01),
		'optimizer': hp.choice('optimizer', ['Adam', 'SGD']),
		'exponential_decay': hp.choice('exponential_decay', [True, False]),
		'decay_steps': hp.randint('decay_steps', 1, upper_limit_iters),
		'decay_rate': hp.uniform('decay_rate', 0.5, 0.999),
		# 'batch_training': hp.choice('batch_training', [True, False]),
		# 'batch_shuffle': hp.choice('batch_shuffle', [True, False]),
		# 'batch_size': hp.choice('batch_size', [8, 16, 32, 64, 128, 256]),
		'early_stop': hp.choice('early_stop', [True, False]),
		'early_stop_start_iter': hp.randint('early_stop_start_iter', 1, upper_limit_iters),
		'wait_patience': hp.randint('wait_patience', 1, upper_limit_iters),
		# 'Nnode': hp.choice('Nnode', [8, 16, 32, 64, 128]),  ## apply to all three networks, single hidden layer for now
		}

		### add layers/nodes info to the 'space'
		space['layers_mean'] = hp.choice('layers_mean', [self.structure_params(n, 'mean') for n in self.layer_options])
		space['layers_up'] = hp.choice('layers_up', [self.structure_params(n, 'up') for n in self.layer_options])
		space['layers_down'] = hp.choice('layers_down', [self.structure_params(n, 'down') for n in self.layer_options])

		if self.configs['batch_training']:
			space['batch_shuffle'] = hp.choice('batch_shuffle', [True, False])
			space['batch_size'] = hp.choice('batch_size', [8, 16, 32, 64, 128, 256])


		trials = Trials()
		if rnd_state is None:
			rstate = np.random.RandomState()
		if rnd_state is not None:
			rstate = np.random.RandomState(rnd_state)
		optim_params = fmin(self.params_optim_target_function, space, algo=tpe.suggest, max_evals=num_trials, trials=trials, rstate=rstate)

		# print('--- Optimized parameters: ')
		# for key, value in optim_params.items():
		# 	if key == 'Nnode':
		# 		print('- {}: {}'.format(key, [8, 16, 32, 64, 128][value]))
		# 	elif key == 'early_stop':
		# 		print('- {}: {}'.format(key, [True, False][value]))
		# 	elif key == 'exponential_decay':
		# 		print('- {}: {}'.format(key, [True, False][value]))
		# 	elif key == 'optimizer':
		# 		print('- {}: {}'.format(key, ['Adam', 'SGD'][value]))
		# 	else:
		# 		print('- {}: {}'.format(key, value))

		## assign the optimized params
		self.configs['seed'] = optim_params['seed']
		self.configs['Max_iter'] = optim_params['Max_iter']

		layers_mean = optim_params['layers_mean'] + 1
		layers_up = optim_params['layers_up'] + 1
		layers_down = optim_params['layers_down'] + 1

		optim_mean_layers = []
		for i in range(layers_mean):
			optim_mean_layers.append(optim_params['n_nodes_mean_{}_{}'.format(layers_mean, i)])

		optim_up_layers = []
		for i in range(layers_up):
			optim_up_layers.append(optim_params['n_nodes_up_{}_{}'.format(layers_up, i)])

		optim_down_layers = []
		for i in range(layers_down):
			optim_down_layers.append(optim_params['n_nodes_down_{}_{}'.format(layers_down, i)])

		self.configs['num_neurons_mean'] = optim_mean_layers
		self.configs['num_neurons_up'] = optim_up_layers
		self.configs['num_neurons_down'] = optim_down_layers

		self.configs['lr'][0] = optim_params['lr_mean']
		self.configs['lr'][1] = optim_params['lr_up']
		self.configs['lr'][2] = optim_params['lr_down']
		self.configs['optimizers'][0] = ['Adam', 'SGD'][optim_params['optimizer']]
		self.configs['optimizers'][1] = ['Adam', 'SGD'][optim_params['optimizer']]
		self.configs['optimizers'][2] = ['Adam', 'SGD'][optim_params['optimizer']]
		self.configs['exponential_decay'] = [True, False][optim_params['exponential_decay']]
		self.configs['decay_steps'] = optim_params['decay_steps']
		self.configs['decay_rate'] = optim_params['decay_rate']
		self.configs['early_stop'] = [True, False][optim_params['early_stop']]
		self.configs['early_stop_start_iter'] = optim_params['early_stop_start_iter']
		self.configs['wait_patience'] = optim_params['wait_patience']

		if self.configs['batch_training']:
			self.configs['batch_shuffle'] = optim_params['batch_shuffle']
			self.configs['batch_size'] = optim_params['batch_size']

		# self.configs['num_neurons_mean_net'] = [8, 16, 32, 64, 128][optim_params['Nnode']]     
		# self.configs['num_neurons_up_down_net'] = [8, 16, 32, 64, 128][optim_params['Nnode']]
		return self.configs


	def params_optim_target_function(self, params):

		self.configs['seed'] = params['seed']
		self.configs['Max_iter'] = params['Max_iter']

		# params['layers_mean'] --> {'n_nodes_layer_mean_0': 116, 'n_nodes_layer_mean_1': 41} 

		self.configs['num_neurons_mean'] = [params['layers_mean'][nodes] for nodes in params['layers_mean']]
		self.configs['num_neurons_up'] = [params['layers_up'][nodes] for nodes in params['layers_up']]
		self.configs['num_neurons_down'] = [params['layers_down'][nodes] for nodes in params['layers_down']]

		# print('--- MEAN:')
		# print(params['layers_mean'])
		# for key, nodes in enumerate(params['layers_mean']):
		# 	print('--layer: {} nodes: {}'.format(key, params['layers_mean'][nodes]))

		# print([params['layers_mean'][nodes] for nodes in params['layers_mean']])


		# print('--- UP:')
		# print(params['layers_up'])
		# for key, nodes in enumerate(params['layers_up']):
		# 	print('--layer: {} nodes: {}'.format(key, params['layers_up'][nodes]))

		# print([params['layers_up'][nodes] for nodes in params['layers_up']])

		# print('--- DOWN:')
		# print(params['layers_down'])
		# for key, nodes in enumerate(params['layers_down']):
		# 	print('--layer: {} nodes: {}'.format(key, params['layers_down'][nodes]))

		# # print([params['layers_down'][nodes] for nodes in params['layers_down']])
		# # # exit()

		self.configs['lr'] = []
		self.configs['lr'].append(params['lr_mean'])
		self.configs['lr'].append(params['lr_up'])
		self.configs['lr'].append(params['lr_down'])
		self.configs['optimizers'] = []
		self.configs['optimizers'].append(params['optimizer'])
		self.configs['optimizers'].append(params['optimizer'])
		self.configs['optimizers'].append(params['optimizer'])
		self.configs['exponential_decay'] = params['exponential_decay']
		self.configs['decay_steps'] = params['decay_steps']
		self.configs['decay_rate'] = params['decay_rate']
		self.configs['early_stop'] = params['early_stop']
		self.configs['early_stop_start_iter'] = params['early_stop_start_iter']
		self.configs['wait_patience'] = params['wait_patience']

		if self.configs['batch_training']:
			self.configs['batch_size'] = params['batch_size']
			self.configs['batch_shuffle'] = params['batch_shuffle']
			# self.configs['batch_shuffle_buffer'] = params['batch_shuffle_buffer']

		random.seed(self.configs['seed'])
		np.random.seed(self.configs['seed'])
		tf.random.set_seed(self.configs['seed'])

		num_inputs = utils.getNumInputsOutputs(self.xTrain)
		num_outputs = utils.getNumInputsOutputs(self.yTrain)

		''' Create network instances'''
		net_mean = UQ_Net_mean_TF2(self.configs, num_inputs, num_outputs)
		net_up = UQ_Net_std_TF2(self.configs, num_inputs, num_outputs, net='up')
		net_down = UQ_Net_std_TF2(self.configs, num_inputs, num_outputs, net='down')

		# ''' Initialize trainer and conduct training/optimizations '''
		trainer = CL_trainer(self.configs, net_mean, net_up, net_down, self.xTrain, self.yTrain, xValid=self.xValid, yValid=self.yValid, xTest=self.xTest, yTest=self.yTest, testDataEvaluationDuringTrain=False)
		trainer.train()        # training for 3 networks
		trainer.boundaryOptimization()  # boundary optimization
		trainer.testDataPrediction()    # evaluation of the trained nets on testing data
		PICP_train, PICP_valid, PICP_test, MPIW_train, MPIW_valid, MPIW_test = trainer.capsCalculation()       # metrics calculation

		customized_loss = np.abs(PICP_valid - self.target_PICP)
		# customized_loss = np.abs(PICP_valid - self.target_PICP) * 0.1* np.abs(MPIW_valid)

		return {'loss': customized_loss, 'status': STATUS_OK}



