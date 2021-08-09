
''' Off-load the main training loops '''

import numpy as np
import pandas as pd
import tensorflow as tf
import os

from src.Networks.network_V2 import UQ_Net_mean_TF2, UQ_Net_std_TF2
from src.Networks.network_V2 import CL_UQ_Net_train_steps
from src.Visualizations.visualization import CL_plotter
from src.Optimizations.boundary_optimizer import CL_boundary_optimizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class CL_trainer:

	def __init__(self, configs, net_mean, net_std_up, net_std_down,
				 xTrain, yTrain, xTest, yTest):
		''' Take all 3 network instance and the trainSteps (CL_UQ_Net_train_steps) instance '''
		self.configs = configs
		self.net_mean = net_mean
		self.net_std_up = net_std_up
		self.net_std_down = net_std_down
		self.xTrain = xTrain
		self.yTrain = yTrain
		self.xTest = xTest
		self.yTest = yTest

		self.trainSteps = CL_UQ_Net_train_steps(self.net_mean, self.net_std_up, self.net_std_down,
								   optimizers=self.configs['optimizers'], ## 'Adam', 'SGD'
								   lr=self.configs['lr'],         ## order: mean, up, down
								   exponential_decay=self.configs['exponential_decay'],
								   decay_steps=self.configs['decay_steps'],
								   decay_rate=self.configs['decay_rate'])

		# self.early_stop_start_iter = configs['early_stop_start_iter']
		# self.verbose = 1

		self.plotter = CL_plotter()

		self.train_loss_mean_list = []
		self.test_loss_mean_list = []
		self.iter_mean_list = []

		self.train_loss_up_list = []
		self.test_loss_up_list = []
		self.iter_up_list = []

		self.train_loss_down_list = []
		self.test_loss_down_list = []
		self.iter_down_list = []

		self.saveFigPrefix = self.configs['data_name']   # prefix for the saved plots


	def train(self):

		''' prepare results txt file '''
		results_path = './Results_PI3NN/'+ self.configs['data_name'] + '_PI3NN_results.txt'
		with open(results_path, 'a') as fwrite:
			fwrite.write('EXP '+'split_seed '+'random_seed '+'PICP_test '+'MPIW_test '+'RMSE '+'R2'+'\n')


		''' Main training iterations '''
		''' Training for the mean '''
		print('--- Start training for MEAN ---')
		stop_training = False
		early_stop_wait = 0
		stopped_iter = 0
		min_delta = 0

		stopped_baseline = None
		if stopped_baseline is not None:
			best_loss = stopped_baseline
		else:
			best_loss = np.Inf
		best_weights = None

		for i in range(self.configs['Max_iter']):
			self.trainSteps.train_loss_net_mean.reset_states()
			self.trainSteps.test_loss_net_mean.reset_states()
			# print(i)
			self.trainSteps.train_step_UQ_Net_mean_TF2(self.xTrain, self.yTrain, self.xTest, self.yTest)
			current_train_loss = self.trainSteps.train_loss_net_mean.result()
			current_test_loss = self.trainSteps.test_loss_net_mean.result()
			if i % 100 == 0:
				print('Iter: {}, train_mean loss: {}, test_mean loss: {}'.format(i, current_train_loss, current_test_loss))
			self.train_loss_mean_list.append(current_train_loss.numpy())
			self.test_loss_mean_list.append(current_test_loss.numpy())
			if self.configs['early_stop'] and i >= self.configs['early_stop_start_iter']:
				if np.less(current_train_loss - min_delta, best_loss):
					best_loss = current_train_loss
					early_stop_wait = 0
					if self.configs['restore_best_weights']:
						best_weights = self.trainSteps.net_mean.get_weights()
				else:
					early_stop_wait += 1
					# print('--- Iter: {}, early_stop_wait: {}'.format(i+1, early_stop_wait))
					if early_stop_wait >= self.configs['wait_patience']:
						stopped_iter = i
						stop_training = True
						if self.configs['restore_best_weights']:
							if self.configs['verbose'] > 0:
								print('--- Restoring mean model weights from the end of the best iteration')
							self.trainSteps.net_mean.set_weights(best_weights)
						if self.configs['saveWeights']:
							print('--- Saving best model weights to h5 file: {}_best_mean_iter_{}.h5'.format(self.configs['data_name'], str(i+1)))
							self.trainSteps.net_mean.save_weights(os.getcwd()+
								'/Results_PI3NN/checkpoints_mean/'+self.configs['data_name']+'_best_mean_iter_' + str(i + 1) + '.h5')
			self.iter_mean_list.append(i)
			if stop_training:
				print('--- Early stopping criteria met.  Iteration: {}, train_loss:{}, test_loss:{}'.format(i+1, current_train_loss, current_test_loss ))
				break

		if self.configs['plot_loss_history']:
			self.plotter.plotTrainValidationLoss(self.train_loss_mean_list, self.test_loss_mean_list,
											trainPlotLabel='training loss', validPlotLabel='test loss',
											xlabel='iterations', ylabel='Loss', title='('+self.saveFigPrefix+')Train/test loss for mean values',
											gridOn=True, legendOn=True,
											saveFigPath=self.configs['plot_loss_history_path']+self.saveFigPrefix+'_MEAN_loss_seed_'+str(self.configs['split_seed'])+'_'+str(self.configs['seed'])+'.png')
											# xlim=[50, len(train_loss_mean_list)])
		if self.configs['save_loss_history']:
			df_loss_MEAN = pd.DataFrame({
				'iter': self.iter_mean_list,
				'train_loss': self.train_loss_mean_list,
				'test_loss': self.test_loss_mean_list
			})
			df_loss_MEAN.to_csv(
				self.configs['save_loss_history_path'] + self.configs['data_name'] + '_MEAN_loss_seed_' + str(self.configs['split_seed']) + '_' + str(self.configs['seed']) + '.csv')


		''' Generate up and down training/validation data '''
		diff_train = (self.yTrain.reshape(self.yTrain.shape[0], -1) - self.trainSteps.net_mean(self.xTrain))
		yTrain_up_data = tf.expand_dims(diff_train[diff_train > 0], axis=1)
		xTrain_up_data = self.xTrain[(diff_train > 0).numpy().flatten(), :]
		yTrain_down_data = -1.0 * tf.expand_dims(diff_train[diff_train < 0], axis=1)
		xTrain_down_data = self.xTrain[(diff_train < 0).numpy().flatten(), :]

		diff_test = (self.yTest.reshape(self.yTest.shape[0], -1) - self.trainSteps.net_mean(self.xTest))
		yTest_up_data = tf.expand_dims(diff_test[diff_test > 0], axis=1)
		xTest_up_data = self.xTest[(diff_test > 0).numpy().flatten(), :]
		yTest_down_data = -1.0 * tf.expand_dims(diff_test[diff_test < 0], axis=1)
		xTest_down_data = self.xTest[(diff_test < 0).numpy().flatten(), :]

		self.xTrain_up = xTrain_up_data
		self.yTrain_up = yTrain_up_data.numpy()
		self.xTrain_down = xTrain_down_data
		self.yTrain_down = yTrain_down_data.numpy()

		self.xTest_up = xTest_up_data
		self.yTest_up = yTest_up_data.numpy()
		self.xTest_down = xTest_down_data
		self.yTest_down = yTest_down_data.numpy()


		''' Training for the UP '''
		print('--- Start training for UP ---')

		stop_training = False
		early_stop_wait = 0
		stopped_iter = 0
		min_delta = 0

		stopped_baseline = None
		if stopped_baseline is not None:
			best_loss = stopped_baseline
		else:
			best_loss = np.Inf
		best_weights = None

		for i in range(self.configs['Max_iter']):
			self.trainSteps.train_loss_net_std_up.reset_states()
			self.trainSteps.test_loss_net_std_up.reset_states()
			self.trainSteps.train_step_UQ_Net_std_UP_TF2(self.xTrain_up, self.yTrain_up, self.xTest_up, self.yTest_up)
			current_train_loss = self.trainSteps.train_loss_net_std_up.result()
			current_test_loss = self.trainSteps.test_loss_net_std_up.result()
			if i % 100 == 0:
				print('Iter: {}, train_up loss: {}, test_up loss: {}'.format(i, current_train_loss, current_test_loss))
			self.train_loss_up_list.append(current_train_loss.numpy())
			self.test_loss_up_list.append(current_test_loss.numpy())
			if self.configs['early_stop'] and i >= self.configs['early_stop_start_iter']:
				if np.less(current_train_loss - min_delta, best_loss):
					best_loss = current_train_loss
					early_stop_wait = 0
					if self.configs['restore_best_weights']:
						best_weights = self.trainSteps.net_std_up.get_weights()
				else:
					early_stop_wait += 1
					# print('--- Iter: {}, early_stop_wait: {}'.format(i+1, early_stop_wait))
					if early_stop_wait >= self.configs['wait_patience']:
						stopped_iter = i
						stop_training = True
						if self.configs['restore_best_weights']:
							if self.configs['verbose'] > 0:
								print('--- Restoring std_up model weights from the end of the best iteration')
							self.trainSteps.net_std_up.set_weights(best_weights)
						if self.configs['saveWeights']:
							print('--- Saving best model weights to h5 file: {}_best_std_up_iter_{}.h5'.format(self.configs['data_name'], str(i+1)))
							self.trainSteps.net_std_up.save_weights(os.getcwd()+
								'/Results_PI3NN/checkpoints_up/'+self.configs['data_name']+'_best_std_up_iter_' + str(i + 1) + '.h5')

			self.iter_up_list.append(i)
			if stop_training:
				print('--- Early stopping criteria met.  Iteration: {}, train_loss:{}, test_loss:{}'.format(i+1, current_train_loss, current_test_loss ))
				break

			### Test model saving
			# if configs['saveWeights']:
			#     trainSteps.net_std_up.save_weights('./checkpoints_up/up_checkpoint_iter_'+str(i+1)+'.h5')

		if self.configs['plot_loss_history']:
			self.plotter.plotTrainValidationLoss(self.train_loss_up_list, self.test_loss_up_list,
											trainPlotLabel='training loss', validPlotLabel='test loss',
											xlabel='iterations', ylabel='Loss', title='('+self.saveFigPrefix+')Train/test loss for UP values',
											gridOn=True, legendOn=True,
											saveFigPath=self.configs['plot_loss_history_path']+self.saveFigPrefix+'_UP_loss_seed_'+str(self.configs['split_seed'])+'_'+str(self.configs['seed'])+'.png')
											# xlim=[50, len(train_loss_up_list)])
		if self.configs['save_loss_history']:
			df_loss_UP = pd.DataFrame({
				'iter': self.iter_up_list,
				'train_loss': self.train_loss_up_list,
				'test_loss': self.test_loss_up_list
			})
			df_loss_UP.to_csv(
				self.configs['save_loss_history_path'] + self.configs['data_name'] + '_UP_loss_seed_' + str(self.configs['split_seed']) + '_' + str(self.configs['seed']) + '.csv')


		''' Training for the DOWN '''
		print('--- Start training for DOWN ---')

		stop_training = False
		early_stop_wait = 0
		stopped_iter = 0
		min_delta = 0

		stopped_baseline = None
		if stopped_baseline is not None:
			best_loss = stopped_baseline
		else:
			best_loss = np.Inf
		best_weights = None

		for i in range(self.configs['Max_iter']):
			self.trainSteps.train_loss_net_std_down.reset_states()
			self.trainSteps.test_loss_net_std_down.reset_states()
			self.trainSteps.train_step_UQ_Net_std_DOWN_TF2(self.xTrain_down, self.yTrain_down, self.xTest_down, self.yTest_down)
			current_train_loss = self.trainSteps.train_loss_net_std_down.result()
			current_test_loss = self.trainSteps.test_loss_net_std_down.result()
			if i % 100 == 0:
				print('Iter: {}, train_down loss: {}, test_down loss: {}'.format(i, current_train_loss, current_test_loss))
			self.train_loss_down_list.append(current_train_loss.numpy())
			self.test_loss_down_list.append(current_test_loss.numpy())

			if self.configs['early_stop'] and i >= self.configs['early_stop_start_iter']:
				if np.less(current_train_loss - min_delta, best_loss):
					best_loss = current_train_loss
					early_stop_wait = 0
					if self.configs['restore_best_weights']:
						best_weights = self.trainSteps.net_std_down.get_weights()
				else:
					early_stop_wait += 1
					# print('--- Iter: {}, early_stop_wait: {}'.format(i+1, early_stop_wait))
					if early_stop_wait >= self.configs['wait_patience']:
						stopped_iter = i
						stop_training = True
						if self.configs['restore_best_weights']:
							if self.configs['verbose'] > 0:
								print('--- Restoring std_down model weights from the end of the best iteration')
							self.trainSteps.net_std_down.set_weights(best_weights)
						if self.configs['saveWeights']:
							print('--- Saving best model weights to h5 file: {}_best_std_down_iter_{}.h5'.format(self.configs['data_name'], str(i+1)))
							self.trainSteps.net_std_up.save_weights(os.getcwd()+
								'/Results_PI3NN/checkpoints_down/'+self.configs['data_name']+'_best_std_down_iter_' + str(i + 1) + '.h5')
			self.iter_down_list.append(i)
			if stop_training:
				print('--- Early stopping criteria met.  Iteration: {}, train_loss:{}, test_loss:{}'.format(i+1, current_train_loss, current_test_loss ))
				break

			### Test model saving
			# if configs['saveWeights']:
			#     trainSteps.net_std_down.save_weights('./checkpoints_down/down_checkpoint_iter_'+str(i+1)+'.h5')
		if self.configs['plot_loss_history']:
			self.plotter.plotTrainValidationLoss(self.train_loss_down_list, self.test_loss_down_list,
											trainPlotLabel='training loss', validPlotLabel='test loss',
											xlabel='iterations', ylabel='Loss', title='('+self.saveFigPrefix+')Train/test loss for DOWN values',
											gridOn=True, legendOn=True,
											saveFigPath=self.configs['plot_loss_history_path']+self.saveFigPrefix+'_DOWN_loss_seed_'+str(self.configs['split_seed'])+'_'+str(self.configs['seed'])+'.png')
											# xlim=[50, len(train_loss_down_list)])
		if self.configs['save_loss_history']:
			df_loss_DOWN = pd.DataFrame({
				'iter': self.iter_down_list,
				'train_loss': self.train_loss_down_list,
				'test_loss': self.test_loss_down_list
			})
			df_loss_DOWN.to_csv(
				self.configs['save_loss_history_path']+self.configs['data_name']+'_DOWN_loss_seed_'+str(self.configs['split_seed'])+'_'+str(self.configs['seed'])+'.csv')


	def boundaryOptimization(self):
		Ntrain = self.xTrain.shape[0]
		num_outlier = int(Ntrain * (1 - self.configs['quantile']) / 2)
		print('-- Number of outlier based on the defined quantile: {}'.format(num_outlier))

		output = self.trainSteps.net_mean(self.xTrain, training=False)
		output_up = self.trainSteps.net_std_up(self.xTrain, training=False)
		output_down = self.trainSteps.net_std_down(self.xTrain, training=False)

		boundaryOptimizer = CL_boundary_optimizer(self.yTrain, output, output_up, output_down, num_outlier,
												  c_up0_ini=0.0,
												  c_up1_ini=100000.0,
												  c_down0_ini=0.0,
												  c_down1_ini=100000.0,
												  max_iter=1000)
		self.c_up = boundaryOptimizer.optimize_up(verbose=0)
		self.c_down = boundaryOptimizer.optimize_down(verbose=0)

		print('c_up: {}'.format(self.c_up))
		print('c_down: {}'.format(self.c_down))


	def testDataPrediction(self):
		self.test_output = self.trainSteps.net_mean(self.xTest, training=False)
		self.test_output_up = self.trainSteps.net_std_up(self.xTest, training=False)
		self.test_output_down = self.trainSteps.net_std_down(self.xTest, training=False)

		self.train_output = self.trainSteps.net_mean(self.xTrain, training=False)
		self.train_output_up = self.trainSteps.net_std_up(self.xTrain, training=False)
		self.train_output_down = self.trainSteps.net_std_down(self.xTrain, training=False)

		
	def capsCalculation(self):
		print('---------------- calculate caps ----------------')
		y_U_cap_train = (self.train_output + self.c_up * self.train_output_up).numpy().flatten() > self.yTrain
		y_L_cap_train = (self.train_output - self.c_down * self.train_output_down).numpy().flatten() < self.yTrain
		y_all_cap_train = y_U_cap_train * y_L_cap_train  # logic_or
		PICP_train = np.sum(y_all_cap_train) / y_L_cap_train.shape[0]  # 0-1
		MPIW_train = np.mean((self.train_output + self.c_up * self.train_output_up).numpy().flatten() - (
				self.train_output - self.c_down * self.train_output_down).numpy().flatten())
		print('Num of train in y_U_cap_train: {}'.format(np.count_nonzero(y_U_cap_train)))
		print('Num of train in y_L_cap_train: {}'.format(np.count_nonzero(y_L_cap_train)))
		print('Num of train in y_all_cap_train: {}'.format(np.count_nonzero(y_all_cap_train)))
		print('np.sum results(train): {}'.format(np.sum(y_all_cap_train)))
		print('PICP_train: {}'.format(PICP_train))
		print('MPIW_train: {}'.format(MPIW_train))

		print('---------------- ------ ----------------')

		y_U_cap_test = (self.test_output + self.c_up * self.test_output_up).numpy().flatten() > self.yTest
		y_L_cap_test = (self.test_output - self.c_down * self.test_output_down).numpy().flatten() < self.yTest
		y_all_cap_test = y_U_cap_test * y_L_cap_test  # logic_or
		self.PICP_test = np.sum(y_all_cap_test) / y_L_cap_test.shape[0]  # 0-1
		self.MPIW_test = np.mean((self.test_output + self.c_up * self.test_output_up).numpy().flatten() - (
				self.test_output - self.c_down * self.test_output_down).numpy().flatten())
		# print('y_U_cap: {}'.format(y_U_cap))
		# print('y_L_cap: {}'.format(y_L_cap))
		print('Num of true in y_U_cap: {}'.format(np.count_nonzero(y_U_cap_test)))
		print('Num of true in y_L_cap: {}'.format(np.count_nonzero(y_L_cap_test)))
		print('Num of true in y_all_cap: {}'.format(np.count_nonzero(y_all_cap_test)))
		print('np.sum results: {}'.format(np.sum(y_all_cap_test)))
		print('PICP_test: {}'.format(self.PICP_test))
		print('MPIW_test: {}'.format(self.MPIW_test))

		print('*********** Test *****************')
		print('*********** Test *****************')
		# print(y_all_cap)
		print(np.sum(y_all_cap_test))

		self.MSE_test = np.mean(np.square(self.test_output.numpy().flatten() - self.yTest))
		self.RMSE_test = np.sqrt(self.MSE_test)
		self.R2_test = r2_score(self.yTest, self.test_output.numpy().flatten())

		print('Test MSE: {}'.format(self.MSE_test))
		print('Test RMSE: {}'.format(self.RMSE_test))
		print('Test R2: {}'.format(self.R2_test))


	def saveResultsToTxt(self):
		''' Save results to txt file '''
		results_path = './Results_PI3NN/'+ self.configs['data_name'] + '_PI3NN_results.txt'
		with open(results_path, 'a') as fwrite:
			fwrite.write(str(self.configs['experiment_id'])+' '+str(self.configs['split_seed'])+' '+str(self.configs['seed'])+' '+str(round(self.PICP_test,3))+' '+str(round(self.MPIW_test, 3))+ ' '
			+str(round(self.RMSE_test,3))+' '+str(round(self.R2_test, 3))+'\n' )

