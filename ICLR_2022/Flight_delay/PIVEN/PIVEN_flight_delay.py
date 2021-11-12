# -*- coding: utf-8 -*-
"""
Run method, and save results.
Run as:
    python main.py --dataset <ds> --method <met>
    where dataset name should be in UCI_Datasets folder
    and method is piven, qd, deep-ens, mid or only-rmse.
"""
import argparse
import json
import datetime
import tensorflow as tf
# import tensorflow.compat.v1 as tf
import scipy.stats as stats
import itertools
import os
import random
import numpy as np


import data_loader
from DataGen import DataGenerator
from DeepNetPI import TfNetwork
from utils import *
from sklearn.model_selection import train_test_split


start_time = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='flight_delay', metavar='',
                    help='dataset name, flight_delay')
parser.add_argument('--method', type=str, help='piven, qd, mid, only-rmse, deep-ens', required=True)
args = parser.parse_args()
method = args.method

#############  added code start  ############# 
print(args)
print(args.dataset)
original_data_path = '../flight_delay_data/' ## flight delay data
results_path = './Results/piven/'+args.dataset + '_PIVEN_UCI.txt'


seed = 12345
neurons = [100]
lambda_in = 15.0
sigma_in = 0.2

random.seed(seed)
np.random.seed(seed)
tf.compat.v1.random.set_random_seed(seed)
# tf.random.set_random_seed(seed)
h_size = neurons


# n_runs = params['n_runs']  # number of runs
n_runs = 1
n_epoch = 500  # number epochs to train for
# h_size = params['h_size']  # number of hidden units in network: [50]=layer_1 of 50, [8,4]=layer_1 of 8, layer_2 of 4
l_rate = 0.01  # learning rate of optimizer
decay_rate = 0.99  # learning rate decay
soften = 160.0 # soften param in the loss
patience = -1  # patience
n_ensemble = 1  # number of individual NNs in ensemble  # 5
alpha = 0.05  # data points captured = (1 - alpha)
train_prop = 0.9  # % of data to use as training
in_ddof = 1 if n_runs > 1 else 0  # this is for results over runs only
is_early_stop = patience != -1

if args.dataset == 'YearPredictionMSD':
    n_batch = 1000  # batch size
    out_biases = [5., -5.]
else:
    n_batch = 100  # batch size
    out_biases = [2., -2.]  # chose biases for output layer (for deep_ens is overwritten to 0,1)


results_runs = []
run = 0
fail_times = 0
for run in range(0, n_runs):


    ''' ######### flight delay data loading ###############'''
    xTrain, yTrain, test_data_list  = data_loader.load_flight_delays(original_data_path)

    # y_train = np.reshape(y_train, (-1, 1))
    # y_test = np.reshape(y_test, (-1, 1))

    '''choose the train/test dataset '''
    x_train = xTrain
    y_train = yTrain
    y_train = y_train.reshape(-1, 1)
    # y_scale = yTrain_scale
    test_idx = 0  # [0, 1, 2, 3] for test 1,2,3,4
    X_test = test_data_list[test_idx][0]
    y_test = test_data_list[test_idx][1]
    y_test = y_test.reshape(-1, 1)


    X_val = X_test
    y_val = y_test.reshape(-1, 1)


    y_pred_all = []
    y_pred_all_train = []

    i = 0
    while i < n_ensemble:
        is_failed_run = False

        tf.reset_default_graph()
        sess = tf.Session()

        print(f'\nrun number {run+1} of {n_runs} -- ensemble number {i+1} of {n_ensemble}')

        # create network
        NN = TfNetwork(x_size=x_train.shape[1],
                       y_size=2,
                       h_size=h_size,
                       alpha=alpha,
                       soften=soften,
                       lambda_in=lambda_in,
                       sigma_in=sigma_in,
                       out_biases=out_biases,
                       method=method,
                       patience=patience,
                       dataset=args.dataset,
                       rnd_seed=seed)

        # train
        NN.train(sess, x_train, y_train, X_val, y_val,
                 n_epoch=n_epoch,
                 l_rate=l_rate,
                 decay_rate=decay_rate,
                 is_early_stop=is_early_stop,
                 n_batch=n_batch)

        # predict
        y_loss, y_pred = NN.predict(sess, X_test=X_test, y_test=y_test)

        # prediction for training data
        y_loss_train, y_pred_train = NN.predict(sess, X_test=x_train, y_test=y_train)

        # check whether the run failed or not
        if np.abs(y_loss) > 20. and fail_times < 1: # jump out of some endless failures
            # if False:
            is_failed_run = True
            fail_times+=1
            print('\n\n### one messed up! repeating ensemble ### failed {}/5 times!'.format(fail_times))
            continue  # without saving result
        else:
            i += 1  # continue to next

        # save prediction
        y_pred_all.append(y_pred)
        y_pred_all_train.append(y_pred_train)
        sess.close()

    y_pred_all = np.array(y_pred_all)
    y_pred_all_train = np.array(y_pred_all_train)

    if method == 'deep-ens':
        y_pred_gauss_mid_all = y_pred_all[:, :, 0]
        # occasionally may get -ves for std dev so need to do max
        y_pred_gauss_dev_all = np.sqrt(np.maximum(np.log(1. + np.exp(y_pred_all[:, :, 1])), 10e-6))
        y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, \
        y_pred_L = gauss_to_pi(y_pred_gauss_mid_all, y_pred_gauss_dev_all)

    else:
        ### for test data
        y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, y_pred_L, y_pred_v = pi_to_gauss(y_pred_all, method=method)
        ### for training data
        y_pred_gauss_mid_train, y_pred_gauss_dev_train, y_pred_U_train, y_pred_L_train, y_pred_v_train = pi_to_gauss(y_pred_all_train, method=method)
    

    ### calculate the confidence scores 
    # for train
    y_U_cap_train = y_pred_U_train > y_train.reshape(-1)
    y_L_cap_train = y_pred_L_train < y_train.reshape(-1)
    MPIW_array_train = y_pred_U_train - y_pred_L_train      
    MPIW_train = np.mean(MPIW_array_train)

    MPIW_array_test = y_pred_U - y_pred_L

    confidence_arr_test = [min(MPIW_train/test_width, 1.0) for test_width in MPIW_array_test]
    confidence_arr_train = [min(MPIW_train/train_width, 1.0) for train_width in MPIW_array_train]

    print('----------- OOD analysis --- confidence scores ----------------')
    print('--- Train conf_scores MEAN: {}, STD: {}'.format(np.mean(confidence_arr_train), np.std(confidence_arr_train)))
    print('--- Test: {} rank: {} conf_scores MEAN: {}, STD: {}'.format(test_idx+1, test_idx+1, np.mean(confidence_arr_test), np.std(confidence_arr_test)))

    dist_arr_train = np.sqrt(np.sum(x_train ** 2.0, axis=1))
    dist_arr_test = np.sqrt(np.sum(X_val ** 2.0, axis=1))

    confidence_arr_train = np.array(confidence_arr_train)
    confidence_arr_test = np.array(confidence_arr_test)

    PIVEN_OOD_train_np = np.hstack((dist_arr_train.reshape(-1, 1), confidence_arr_train.reshape(-1, 1)))
    PIVEN_OOD_test_np = np.hstack((dist_arr_test.reshape(-1, 1), confidence_arr_test.reshape(-1, 1)))

    np.savetxt('PIVEN_OOD_flight_delay_'+ str(test_idx+1) +'_train_np.txt', PIVEN_OOD_train_np, delimiter=',')
    np.savetxt('PIVEN_OOD_flight_delay_'+ str(test_idx+1) +'_test_np.txt', PIVEN_OOD_test_np, delimiter=',')

    # # work out metrics
    # y_U_cap = y_pred_U > y_test.reshape(-1)
    # y_L_cap = y_pred_L < y_test.reshape(-1)

    # y_all_cap = y_U_cap * y_L_cap
    # PICP = np.sum(y_all_cap) / y_L_cap.shape[0]
    # MPIW = np.mean(y_pred_U - y_pred_L)
    # y_pred_mid = np.mean((y_pred_U, y_pred_L), axis=0)
    # # MSE = np.mean(np.square(Gen.scale_c * (y_pred_mid - y_test[:, 0])))
    # # RMSE = np.sqrt(MSE)

    # if method == 'qd' or method == 'deep-ens':
    #     RMSE_ELI = 0.0  # RMSE_PIVEN
    # else:
    #     if method == 'piven':
    #         y_piven = y_pred_v * y_pred_U + (1 - y_pred_v) * y_pred_L

    #     elif method == 'mid':
    #         y_piven = 0.5 * y_pred_U + 0.5 * y_pred_L

    #     elif method == 'only-rmse':
    #         y_piven = y_pred_v

    #     MSE_ELI = np.mean(np.square(Gen.scale_c * (y_piven - y_test[:, 0])))
    #     RMSE_ELI = np.sqrt(MSE_ELI) # RMSE_PIVEN

    # CWC = np_QD_loss(y_test, y_pred_L, y_pred_U, alpha, lambda_in)  # from qd paper.
    # neg_log_like = gauss_neg_log_like(y_test, y_pred_gauss_mid, y_pred_gauss_dev, Gen.scale_c)
    # residuals = y_pred_mid - y_test[:, 0]
    # shapiro_W, shapiro_p = stats.shapiro(residuals[:])
    # results_runs.append((PICP, MPIW, CWC, RMSE, RMSE_ELI, neg_log_like, shapiro_W, shapiro_p))

# # summarize results
# results_path = f"./Results/{method}/"
# results_path += f"{params['dataset']}-{start_time.strftime('%d-%m-%H-%M')}-{method}.csv"

# results = np.array(results_runs)
# results_to_csv(results_path, results, params, n_runs, n_ensemble, in_ddof)

# timing info
end_time = datetime.datetime.now()
total_time = end_time - start_time
print('\n\nminutes taken:', round(total_time.total_seconds() / 60, 3),
      '\nstart_time:', start_time.strftime('%H:%M:%S'),
      'end_time:', end_time.strftime('%H:%M:%S'))
