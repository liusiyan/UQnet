# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from scipy import stats
import os
import importlib
import DeepNetPI_V2
import DataGen_V2
import utils
from sklearn.metrics import r2_score
import os
import random

import itertools

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # avoids a warning
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

importlib.reload(DeepNetPI_V2)
importlib.reload(DataGen_V2)
importlib.reload(utils)

from DataGen_V2 import DataGenerator
from DeepNetPI_V2 import TfNetwork
from utils import *
import numpy as np
import datetime

import tensorflow.compat.v1 as tf
# Force using CPU globally by hiding GPU(s)
tf.config.set_visible_devices([], 'GPU')
tf.disable_v2_behavior()
tf.config.threading.set_intra_op_parallelism_threads(4)

from function_generator_OOD import CL_function_generator  # the OOD data synthesizer from user defined function

''' 
This is the code for QD method used for our comparison which was adjusted from the original QD code (also included in the upper level folder)
or you can find it at https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals

All of our experiments on QD method with synthetic datasets were conducted on a single Ubuntu workstation, and we use Intel I9-10980xe CPU generated 
all the results instead of using a GPU because the training are relatively fast on CPU.

The results will be generated in the ./Results/ including the summary of the training results (.txt files), plotted loss curves (./Results/loss_curves/)
and loss history for each case (.csv format in ./Results/loss_history/)

We also prepared pre-generated results for your reference (in ./Pre_generated_results/)

Have fun!
'''
def standardizer(input_np):
	input_mean = input_np.mean(axis=0, keepdims=1)
	input_std = input_np.std(axis=0, keepdims=1)
	input_std[input_std < 1e-10] = 1.0
	standardized_input = (input_np - input_mean) / input_std
	return standardized_input, input_mean, input_std

''' Training data generation '''
function_generator = CL_function_generator(generator_rnd_seed=1) # this random seed is used for adding various random noise

data_name = 'FUN'   ### Customized function instead of UCI data sets
type_in = '~' + data_name 	# data type to use - drunk_bow_tie x_cubed_gap ~boston ~concrete
original_data_path = '../../UCI_datasets/'          ## original UCI data sets
splitted_data_path = '../../UCI_TrainTest_Split/'   ## pre-split data
results_path = './Results/'+data_name + '_QD_results.txt'

random_seed_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# seed_combination_list = list(itertools.product(split_seed_list, random_seed_list))
# print('-- The splitting and random seed combination list: {}'.format(seed_combination_list))

save_loss_history = True
save_loss_history_path = './Results/loss_history/'
plot_loss_history = True
plot_loss_history_path = './Results/loss_curves/'

with open(results_path, 'a') as fwrite:
	fwrite.write('EXP '+'random_seed '+'PICP_test '+'MPIW_test '+'RMSE '+'R2 '+'qd_loss '+'NLL '+'alpha '+'loss '+'data '+'h_size '+'bstraps '+'ensemb '+
				 'soft '+'lambda_in'+'\n')

for iii in range(len(random_seed_list)):
	if iii == 0:  # protein >=3
		# split_seed = seed_combination_list[iii][0]
		seed = random_seed_list[iii]
		print('--- Running EXP {}/{}'.format(iii+1, len(random_seed_list)))
		print('--- Dataset: {}'.format(data_name))
		print('--- Random seed: {}'.format(seed))

		random.seed(seed)
		np.random.seed(seed)
		# tf.random.set_seed(seed)
		tf.random.set_random_seed(seed)

		start_time = datetime.datetime.now()

		# soften_list = [10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180., 190., 200.]
		# lambda_in_list = [5., 10., 15., 20., 25.]

		loss_type = 'qd_soft' 		# loss type to train on - qd_soft gauss_like(=mve) mse (mse=simple point prediction)
		n_samples = 10000		# if generating data, how many points to generate
		# if data_name == 'naval' or data_name == 'MSD':
		# 	h_size = [100]  # from the original value proposed by the author
		# else:
		# 	h_size = [50]  # from the original value proposed by the author
		h_size = [100]
		# h_size = [50]	    # number of hidden units in network: [50]=layer_1 of 50, [8,4]=layer_1 of 8, layer_2 of 4  [100] was used for protein and Song Year
		alpha = 0.05		# data points captured = (1 - alpha)

		scale_c = 1.0 # default

		n_epoch = 1000  # from the original value proposed by the author
		l_rate = 0.02  # from the original value proposed by the author
		decay_rate = 0.9  # from the original value proposed by the author
		sigma_in = 0.1  # from the original value proposed by the author
		lambda_in = 15.0  # from the original value proposed by the author

		optim = 'adam' 		# opitimiser - SGD adam
		soften = 160. 		# hyper param for QD_soft
		# lambda_in = 4.0 	# hyper param for   ## 4.0 for naval, 40.0 for protein, 30.0 for wine, 6.0 for yacht, 15 for rest of them

		is_run_test = False	# if averaging over lots of runs - turns off some prints and graphs
		n_ensemble = 1	# number of individual NNs in ensemble
		n_bootstraps = 1 		# how many boostrap resamples to perform
		n_runs = 20 if is_run_test else 1
		is_batch = True 		# train in batches?
		n_batch = 100   		# batch size
		lube_perc = 90 	# 90	# if model uncertainty method = perc - 50 to 100
		perc_or_norm ='norm' # model uncertainty method - perc norm (paper uses norm)
		is_early_stop = False # stop training early (didn't use in paper)
		is_bootstrap = False if n_bootstraps == 1 else True
		train_prop = 0.9 		# % of data to use as training, 0.8 for hyperparam selection

		out_biases=[3.,-3.] # chose biases for output layer (for gauss_like is overwritten to 0,1)
		activation='relu' 	# NN activation fns - tanh relu

		# plotting options
		is_use_val=True
		save_graphs=True
		# show_graphs=True if is_run_test else True
		show_graphs = True if is_run_test else False
		# show_train=True if is_run_test else False
		show_train=True
		is_y_rescale=False
		is_y_sort=False
		is_print_info=True
		var_plot=0 # lets us plot against different variables, use 0 for univariate
		is_err_bars=True
		is_norm_plot=False
		is_boundary=False # boundary stuff ONLY works for univariate - turn off for larger
		is_bound_val=False # plot validation points for boundary
		is_bound_train=True # plot training points for boundary
		is_bound_indiv=True # plot individual boundary estimates
		is_bound_ideal=True # plot ideal boundary
		is_title=True # show title w metrics on graph
		bound_limit=6. # how far to plot boundary

		# resampling
		bootstrap_method='replace_resample' # whether to boostrap or jacknife - prop_of_data replace_resample
		prop_select=0.8 # if jacknife (=prop_of_data), how much data to use each time

		# other
		in_ddof=1 if n_runs > 1 else 0 # this is for results over runs only

		# pre calcs
		if alpha == 0.05:
			n_std_devs = 1.96
		elif alpha == 0.10:
			n_std_devs = 1.645
		elif alpha == 0.01:
			n_std_devs = 2.575
		else:
			raise Exception('ERROR unusual alpha')

		results_runs = []
		results_runs_test = []
		run=0
		for run in range(0, n_runs):

			x_dim = 10
			coeff_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
			power_list = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

			Npar = 10
			Ntrain = 5000
			Ntest = 10000
			Nout = 1

			xTrain = np.random.normal(loc=0.0, scale=1.0, size=(Ntrain, Npar)) * 1.0  # mean zero and variance 1
			xTest = np.random.normal(loc=0.0, scale=1.0, size=(Ntest, Npar)) * 5.0  # mean zero and variance 1

			# xTrain = tf.random.normal([Ntrain, Npar], mean=0.0, stddev=1.0) * 1.0   # mean zero and variance 1
			# xTest = tf.random.normal([Ntest, Npar], mean=0.0, stddev=1.0) * 5.0   # mean zero and variance 1

			yTrain = function_generator.generate_y(xTrain, x_dim, coeff_list, power_list, denominator=10,
												   noise=False, noise_amplitude=0.0, noise_shift=0)
			yTest = function_generator.generate_y(xTest, x_dim, coeff_list, power_list, denominator=10,
												  noise=False, noise_amplitude=0.0, noise_shift=0)


			# np.savetxt('QD_yTest.txt', xTest, delimiter=',')

			# normalization
			''' Standardize inputs '''
			# standardized_input, input_mean, input_std = dataLoader.standardizer(input)
			xTrain, xTrain_mean, xTrain_std = standardizer(xTrain)
			xTest = (xTest - xTrain_mean) / xTrain_std

			yTrain, yTrain_mean, yTrain_std = standardizer(yTrain)
			yTest = (yTest - yTrain_mean) / yTrain_std

			X_train = xTrain
			y_train = yTrain
			X_val = xTest
			y_val = yTest

		######################################

			print('\n--- view data ---')
			# Gen.ViewData(n_rows=5, hist=False, plot=False)

			X_boundary = []
			y_boundary = []
			y_pred_all = []
			y_pred_all_train = []
			y_pred_all_test = []
			X_train_orig, y_train_orig = X_train, y_train
			for b in range(0,n_bootstraps):

				# bootstrap sample
				if is_bootstrap:

					np.random.seed(b)
					if bootstrap_method=='replace_resample':
						# resample w replacement method
						id = np.random.choice(X_train_orig.shape[0], X_train_orig.shape[0], replace=True)
						X_train = X_train_orig[id]
						y_train = y_train_orig[id]

					elif bootstrap_method=='prop_of_data':
						# select x% of data each time NO resampling
						perm = np.random.permutation(X_train_orig.shape[0])
						X_train = X_train_orig[perm[:int(perm.shape[0]*prop_select)]]
						y_train = y_train_orig[perm[:int(perm.shape[0]*prop_select)]]

				i=0
				fail_times = 0
				while i < n_ensemble:
					is_failed_run = False

					tf.reset_default_graph()
					sess = tf.Session()

					# info
					if is_print_info:
						print('\nrun number', run+1, ' of ', n_runs,
							' -- bootstrap number', b+1, ' of ', n_bootstraps,
							' -- ensemble number', i+1, ' of ', n_ensemble)

					# load network
					NN = TfNetwork(x_size=X_train.shape[1], y_size=2, h_size=h_size,
						type_in="pred_intervals", alpha=alpha, loss_type=loss_type,
						soften=soften, lambda_in=lambda_in, sigma_in=sigma_in, activation=activation, bias_rand=False, out_biases=out_biases,
						rnd_seed=seed)

					# train
					NN.train(sess, X_train, y_train, X_val, y_val,
						n_epoch=n_epoch, l_rate=l_rate, decay_rate=decay_rate,
						resume_train=False, print_params=False, is_early_stop=is_early_stop,
						is_use_val=is_use_val, optim=optim, is_batch=is_batch, n_batch=n_batch,
						is_run_test=is_run_test,is_print_info=is_print_info, rnd_seed=seed)

					# visualise training
					if show_train:
						# NN.vis_train(save_graphs, is_use_val)
						NN.vis_train_v2(data_name=data_name, rnd_seed=seed, ensemble=i,
									 save_graphs=save_graphs, is_use_val=is_use_val, bool_save_loss=True,
										save_loss_path=save_loss_history_path,
										plot_loss_path=plot_loss_history_path)

					''' make predictions (for all training data) '''
					y_loss_train, y_pred_train, y_metric_train, y_U_cap_train, y_U_prop_train, \
						y_L_cap_train, y_L_prop_train, y_all_cap_train, y_all_prop_train \
						= NN.predict(sess, X=X_train,y=y_train,in_sess=True)


					# make predictions (for testing data)
					y_loss, y_pred, y_metric, y_U_cap, y_U_prop, \
						y_L_cap, y_L_prop, y_all_cap, y_all_prop \
						= NN.predict(sess, X=X_val,y=y_val,in_sess=True)

					# check whether the run failed or not
					if np.abs(y_loss) > 20000. and fail_times < 5: # jump out of some endless failures
					# if False:
						is_failed_run = True
						fail_times+=1
						print('\n\n### one messed up! repeating ensemble ### failed {}/5 times!'.format(fail_times))
						with open(results_path, 'a') as fwrite:
							fwrite.write(str(iii + 1) + ' ' + ' ' + str(seed) + ' failed '+str(fail_times)+'times!'+'\n')
						continue # without saving!
					else:
						i+=1 # continue to next

					# save prediction
					y_pred_all.append(y_pred)
					y_pred_all_train.append(y_pred_train)
					# y_pred_all_test.append(y_pred_test)

					# predicting for boundary, need to do this for each model
					if is_boundary:
						X_boundary.append(np.linspace(start=-bound_limit,stop=bound_limit, num=500)[:,np.newaxis])
						t, y_boundary_temp, t, t, t, t, t, t, t = NN.predict(sess, X=X_boundary[i-1],
							y=np.zeros_like(X_boundary[i-1]),in_sess=True)
						y_boundary.append(y_boundary_temp)

					sess.close()

			# we may have predicted with gauss_like or qd_soft, here we need to get estimates for
			# upper/lower pi's AND gaussian params no matter which method we used (so can compare)
			y_pred_all = np.array(y_pred_all)
			y_pred_all_train = np.array(y_pred_all_train)
			# y_pred_all_test = np.array(y_pred_all_test)

			if loss_type == 'qd_soft':
				y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, \
					y_pred_L = pi_to_gauss(y_pred_all, lube_perc, perc_or_norm, n_std_devs)

			''' for training prediction '''
			if loss_type == 'qd_soft':
				y_pred_gauss_mid_train, y_pred_gauss_dev_train, y_pred_U_train, \
					y_pred_L_train = pi_to_gauss(y_pred_all_train, lube_perc, perc_or_norm, n_std_devs)


				# y_pred_gauss_mid_test, y_pred_gauss_dev_test, y_pred_U_test, \
				# 	y_pred_L_test = pi_to_gauss(y_pred_all_test, lube_perc, perc_or_norm, n_std_devs)


			elif loss_type == 'gauss_like': # work out bounds given mu sigma
				y_pred_gauss_mid_all = y_pred_all[:,:,0]
				# occasionally may get -ves for std dev so need to do max
				y_pred_gauss_dev_all = np.sqrt(np.maximum(np.log(1.+np.exp(y_pred_all[:,:,1])),10e-6))
				y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, \
					y_pred_L = gauss_to_pi(y_pred_gauss_mid_all, y_pred_gauss_dev_all, n_std_devs)

			elif loss_type == 'mse': # as for gauss_like but we don't know std dev so guess
				y_pred_gauss_mid_all = y_pred_all[:,:,0]
				y_pred_gauss_dev_all = np.zeros_like(y_pred_gauss_mid_all)+0.01
				y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, \
					y_pred_L = gauss_to_pi(y_pred_gauss_mid_all, y_pred_gauss_dev_all, n_std_devs)



			''' Calculate the confidence scores (y-axis) range from 0-1'''

			#### for train
			y_U_cap_train = y_pred_U_train > y_train.reshape(-1)
			y_L_cap_train = y_pred_L_train < y_train.reshape(-1)
			MPIW_array_train = y_pred_U_train - y_pred_L_train
			MPIW_train = np.mean(MPIW_array_train)
			# if the y_U_cap - y_L_cap for testing data point

			#### for test (evaluate each y_U_cap - y_L_cap in the pre-calculated MPIW_train single value
			# for the confidence score)
			y_U_cap = y_pred_U > y_val.reshape(-1)
			y_L_cap = y_pred_L < y_val.reshape(-1)
			MPIW_array_test = y_pred_U - y_pred_L

			confidence_arr_test = [min(MPIW_train/test_width, 1.0) for test_width in MPIW_array_test]
			confidence_arr_train = [min(MPIW_train/train_width, 1.0) for train_width in MPIW_array_train]
			# print(confidence_arr)
			''' Calculate the L2 distance to the mean of training data (x-axis), range from 0-30'''

			# dist = torch.sqrt(torch.sum(xvalid ** 2.0, 1, keepdim=True))
			# dist1 = torch.sqrt(torch.sum(xtrain ** 2.0, 1, keepdim=True))

			dist_arr_train = np.sqrt(np.sum(X_train ** 2.0, axis=1))
			dist_arr_test = np.sqrt(np.sum(X_val ** 2.0, axis=1))

			print('dist_arr_train shape: {}'.format(dist_arr_train.shape))
			print('confidence arr train len: {}'.format(len(confidence_arr_train)))

			print('dist_arr_test shape: {}'.format(dist_arr_test.shape))
			print('confidence arr test len: {}'.format(len(confidence_arr_test)))

			''' Save to file and plot the results '''
			confidence_arr_train = np.array(confidence_arr_train)
			confidence_arr_test = np.array(confidence_arr_test)

			QD_OOD_train_np = np.hstack((dist_arr_train.reshape(-1, 1), confidence_arr_train.reshape(-1, 1)))
			QD_OOD_test_np = np.hstack((dist_arr_test.reshape(-1, 1), confidence_arr_test.reshape(-1, 1)))

			np.savetxt('QD_OOD_train_np.txt', QD_OOD_train_np, delimiter=',')
			np.savetxt('QD_OOD_test_np.txt', QD_OOD_test_np, delimiter=',')


			plt.plot(dist_arr_train, confidence_arr_train, 'r.', label='Training data (in distribution)')
			plt.plot(dist_arr_test, confidence_arr_test, 'b.', label='testing data (out of distribution')
			plt.xlabel('L2 distance to the mean of training data $\{x_i\}_{i=1}^N$')
			plt.legend(loc='lower left')
			plt.title('QD')
			plt.ylim(0, 1.2)
			plt.savefig('QD_OOD_plot.png')
			# plt.show()

			# work out metrics (for validation/testing data)
			y_U_cap = y_pred_U > y_val.reshape(-1)
			y_L_cap = y_pred_L < y_val.reshape(-1)
			y_all_cap = y_U_cap*y_L_cap
			PICP = np.sum(y_all_cap)/y_L_cap.shape[0]

			MPIW_array_test = y_pred_U - y_pred_L

			MPIW = np.mean(MPIW_array_test)
			print(MPIW_array_test)
			print('MPIW array shape: {}'.format(MPIW_array_test.shape))

			# MPIW = np.mean(y_pred_U - y_pred_L)
			y_pred_mid = np.mean((y_pred_U, y_pred_L), axis=0)
			MSE = np.mean(np.square(scale_c*(y_pred_mid - y_val[:,0])))
			RMSE = np.sqrt(MSE)
			CWC = np_QD_loss(y_val, y_pred_L, y_pred_U, alpha, soften, lambda_in)
			neg_log_like = gauss_neg_log_like(y_val, y_pred_gauss_mid, y_pred_gauss_dev, scale_c)
			residuals = residuals = y_pred_mid - y_val[:,0]
			shapiro_W, shapiro_p = stats.shapiro(residuals[:])
			R2 = r2_score(y_val[:,0],y_pred_mid)
			results_runs.append((PICP, MPIW, CWC, RMSE, neg_log_like, shapiro_W, shapiro_p, R2))

			### write resutls to file
			''' Save results to dat file '''
			with open(results_path, 'a') as fwrite:
				fwrite.write(str(iii+1)+' '+' '+str(seed)+' '+str(round(PICP,3))+' '+str(round(MPIW, 3))+ ' '
				+str(round(RMSE,3))+' '+str(round(R2, 3))+' '+str(round(CWC, 3)) + ' '+str(round(neg_log_like,3))+' '+str(alpha)+' '+NN.loss_type+' '+type_in+' '+str(NN.h_size) + ' '\
				+str(n_bootstraps)+' '+str(n_ensemble)+' '+' '+str(NN.soften)+' '+str(NN.lambda_in)
							 +'\n' )


			# concatenate for graphs
			title = 'PICP=' + str(round(PICP,3))\
						+ ', MPIW=' + str(round(MPIW,3)) \
						+ ', qd_loss=' + str(round(CWC,3)) \
						+ ', NLL=' + str(round(neg_log_like,3)) \
						+ ', alpha=' + str(alpha) \
						+ ', loss=' + NN.loss_type \
						+ ', data=' + type_in + ',' \
						+ '\nh_size=' + str(NN.h_size) \
						+ ', bstraps=' + str(n_bootstraps) \
						+ ', ensemb=' + str(n_ensemble) \
						+ ', RMSE=' + str(round(RMSE,3)) \
						+ ', soft=' + str(NN.soften) \
						+ ', lambda=' + str(NN.lambda_in)

			# visualise
			if show_graphs:
				# error bars
				if is_err_bars:
					plot_err_bars(X_val, y_val, y_pred_U, y_pred_L,
						is_y_sort, is_y_rescale, scale_c, save_graphs,
						title, var_plot, is_title)

			# 	# visualise boundary
			# 	if is_boundary:
			# 		y_bound_all=np.array(y_boundary)
			# 		plot_boundary(y_bound_all, X_boundary, y_val, X_val,
			# 			y_train, X_train, loss_type,
			# 			Gen.y_ideal_U, Gen.y_ideal_L, Gen.X_ideal, Gen.y_ideal_mean, is_bound_ideal,
			# 			is_y_rescale, Gen.scale_c, save_graphs,
			# 			in_ddof, perc_or_norm, lube_perc, n_std_devs,
			# 			is_bound_val, is_bound_train, is_bound_indiv,
			# 			title, var_plot, is_title)

			# 	# normal dist stuff
			# 	if is_norm_plot:
			# 		title = 'shapiro_W=' + str(round(shapiro_W,3)) + \
			# 			', data=' + type_in +', loss=' + NN.loss_type + \
			# 			', n_val=' + str(y_val.shape[0])
			# 		fig, (ax1,ax2) = plt.subplots(2)
			# 		ax1.set_xlabel('y_pred - y_val') # histogram
			# 		ax1.hist(residuals,bins=30)
			# 		ax1.set_title(title, fontsize=10)
			# 		stats.probplot(residuals[:], plot=ax2) # QQ plot
			# 		ax2.set_title('')
			# 		fig.show()

		# sumarise results, print for paste to excel
		print('\n\nn_samples, h_size, n_epoch, l_rate, decay_rate, soften, lambda_in, sigma_in')
		print(n_samples, h_size, n_epoch, l_rate, decay_rate, soften, lambda_in, sigma_in)
		print('\n\ndata=',type_in, 'loss_type=',loss_type)
		results_runs = np.array(results_runs)
		results_runs_test = np.array(results_runs_test)
		metric_names= ['PICP_test', 'MPIW_test', 'CWC_test', 'RMSE_test', 'NLL_test', 'shap_W_test', 'shap_p_test','R2_test']
		print('runs\tboots\tensemb')
		print(n_runs, '\t', n_bootstraps, '\t', n_ensemble)

		print('======= Validation data results ======')
		print('\tavg\tstd_err\tstd_dev')
		for i in range(0,len(metric_names)):
			avg = np.mean(results_runs[:,i])
			std_dev = np.std(results_runs[:,i], ddof=in_ddof)
			std_err = std_dev/np.sqrt(n_runs)
			print(metric_names[i], '\t', round(avg,3),
				'\t', round(std_err,3),
				'\t', round(std_dev,3))


		# timing info
		end_time = datetime.datetime.now()
		total_time = end_time - start_time
		print('seconds taken:', round(total_time.total_seconds(),1),
			'\nstart_time:', start_time.strftime('%H:%M:%S'),
			'end_time:', end_time.strftime('%H:%M:%S'))



