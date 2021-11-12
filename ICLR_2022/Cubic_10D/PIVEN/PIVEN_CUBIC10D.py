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
import matplotlib.pyplot as plt
from scipy import stats



from DataGen import DataGenerator
from DeepNetPI import TfNetwork
from utils import *
from sklearn.model_selection import train_test_split


start_time = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='boston', metavar='',
                    help='dataset name, from UCI_Datasets folder')
parser.add_argument('--method', type=str, help='piven, qd, mid, only-rmse, deep-ens', required=True)
args = parser.parse_args()

#############  added code start  ############# 
print(args)
print(args.dataset)
original_data_path = '../UCI_datasets/'          ## original UCI data sets
splitted_data_path = '../UCI_TrainTest_Split/'   ## pre-split data
results_path = './Results/piven/'+args.dataset + '_PIVEN_UCI.txt'


split_seed_list = [1, 2, 3]
random_seed_list = [10, 20, 30]
num_neurons_list = [[50], [75], [100]]
lambda_in_list = [5.0, 15.0, 20.0]  # lambda_in param in the loss
sigma_in_list = [0.05, 0.2, 0.4]    #initialize std dev of NN weights
params_combination_list = list(itertools.product(split_seed_list, random_seed_list, num_neurons_list, lambda_in_list, sigma_in_list))

#############  added code end  ############# 

method = args.method
params_file = 'params.json' if method != 'deep-ens' else 'params_deep_ens.json'

# get params of the given dataset
# with open(params_file) as params_json:
#   all_params = json.load(params_json)
#   try:
#       params = next(el for el in all_params if el['dataset'] == args.dataset)
#   except StopIteration:
#       raise ValueError(f"Invalid dataset name: {args.dataset}")


#### sensitivity for parameter 'lambda_in' and 'sigma_in'
# lambda_in_list = params['lambda_in']  # lambda_in param in the loss
# sigma_in_list = params['sigma_in']  #  initialize std dev of NN weights

with open(results_path, 'a') as fwrite:
    fwrite.write('random_seed '+'neurons '+'soft '+'lambda_in '+'sigma_in '+'PICP_test '+'MPIW_test '+'RMSE '+'alpha '+'h_size '+'epochs '+'lr '+'decay_rate '+'ensemb '+'n_runs'
                 +'\n')

seed = 1234567
neurons = 100   
lambda_in = 15.0
sigma_in = 0.2

random.seed(seed)
np.random.seed(seed)
tf.random.set_random_seed(seed)
h_size = [neurons]


# n_runs = params['n_runs']  # number of runs
n_runs = 1
n_epoch = 1000
# h_size = params['h_size']  # number of hidden units in network: [50]=layer_1 of 50, [8,4]=layer_1 of 8, layer_2 of 4
l_rate = 0.01
decay_rate = 0.9
soften = 160.0
patience = -1
n_ensemble = 1  # number of individual NNs in ensemble  # 5
alpha = 0.10  # data points captured = (1 - alpha)
train_prop = 0.9  # % of data to use as training
in_ddof = 1 if n_runs > 1 else 0  # this is for results over runs only
is_early_stop = patience != -1

if args.dataset == 'YearPredictionMSD':
    n_batch = 1000  # batch size
    out_biases = [5., -5.]
else:
    n_batch = 100  # batch size
    out_biases = [1., -1.]  # chose biases for output layer (for deep_ens is overwritten to 0,1)


results_runs = []
run = 0
fail_times = 0
for run in range(0, n_runs):

    ''' ######### Original data loading ###############'''
    # Gen = DataGenerator(dataset_name=args.dataset)
    # X_train, y_train, X_test, y_test = Gen.create_data(seed_in=run, train_prop=train_prop)
    # if is_early_stop:
    #     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=run)
    # else:
    #     X_train, X_val, y_train, y_val = X_train, X_test, y_train, y_test


    '''######### Load Cubic 10D data ##########'''
    Gen = DataGenerator(dataset_name=args.dataset)
    X_train, y_train, X_test, y_test = Gen.create_cubic_10D_data()

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # import matplotlib.pyplot as plt
    # plt.plot(y_test)
    # plt.show()


    ''' ######### UCL pre-splitted data loading###############'''
    # Gen = DataGenerator(dataset_name=args.dataset)
    # X_train, y_train, X_test, y_test = Gen.LoadData_Splitted_UCI(args.dataset, original_data_path, splitted_data_path, split_seed)
    # y_train = np.reshape(y_train, (-1, 1))
    # y_test = np.reshape(y_test, (-1, 1))

    # X_train = xTrain
    # y_train = yTrain
    X_val = X_test
    y_val = y_test


    y_pred_all = []
    y_pred_all_train = []

    i = 0
    while i < n_ensemble:
        is_failed_run = False

        tf.reset_default_graph()
        sess = tf.Session()

        print(f'\nrun number {run+1} of {n_runs} -- ensemble number {i+1} of {n_ensemble}')

        # create network
        NN = TfNetwork(x_size=X_train.shape[1],
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
        NN.train(sess, X_train, y_train, X_val, y_val,
                 n_epoch=n_epoch,
                 l_rate=l_rate,
                 decay_rate=decay_rate,
                 is_early_stop=is_early_stop,
                 n_batch=n_batch)

        # predict --- test
        y_loss, y_pred = NN.predict(sess, X_test=X_test, y_test=y_test)
        # predict --- train
        y_loss_train, y_pred_train = NN.predict(sess, X_test=X_train, y_test=y_train)

        # check whether the run failed or not
        if np.abs(y_loss) > 20. and fail_times < 1: # jump out of some endless failures
            # if False:
            is_failed_run = True
            fail_times+=1
            print('\n\n### one messed up! repeating ensemble ### failed {}/5 times!'.format(fail_times))
            with open(results_path, 'a') as fwrite:
                        fwrite.write(str(seed) + ' failed '+str(fail_times)+'times!'+'\n')
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
        ### for test
        y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, y_pred_L, y_pred_v = pi_to_gauss(y_pred_all, method=method)
        ### for train
        y_pred_gauss_mid_train, y_pred_gauss_dev_train, y_pred_U_train, y_pred_L_train, y_pred_v_train = pi_to_gauss(y_pred_all_train, method=method)

    # work out metrics
    y_U_cap = y_pred_U > y_test.reshape(-1)
    y_L_cap = y_pred_L < y_test.reshape(-1)
    y_all_cap = y_U_cap * y_L_cap
    PICP = np.sum(y_all_cap) / y_L_cap.shape[0]
    MPIW = np.mean(y_pred_U - y_pred_L)
    y_pred_mid = np.mean((y_pred_U, y_pred_L), axis=0)

    MSE = np.mean(np.square(Gen.scale_c * (y_pred_mid - y_test[:, 0])))
    RMSE = np.sqrt(MSE)



    #### PIW analysis
    train_PIW_arr = y_pred_U_train - y_pred_L_train
    test_PIW_arr = y_pred_U - y_pred_L

    np.savetxt('PIVEN_MPIW_train_bias.dat', train_PIW_arr)
    np.savetxt('PIVEN_MPIW_test_bias.dat', test_PIW_arr)

    kde_train = stats.gaussian_kde(train_PIW_arr)
    kde_test = stats.gaussian_kde(test_PIW_arr)

    x1 = np.linspace(train_PIW_arr.min(), train_PIW_arr.max(), 100)
    p1 = kde_train(x1)

    x2 = np.linspace(test_PIW_arr.min(), test_PIW_arr.max(), 100)
    p2 = kde_test(x2)

    plt.plot(x1,p1, label='train')
    plt.plot(x2,p2, label='test')
    plt.legend()
    plt.savefig('PIVEN_cubic10D_bias.png')
    plt.show()

    ###------- Option I: calculate confidence interval
    conf_score = kde_train(test_PIW_arr)/p1.max()
    print('--Conf_score mean: {}, STD: {}'.format(np.mean(conf_score), np.std(conf_score)))


    if method == 'qd' or method == 'deep-ens':
        RMSE_ELI = 0.0  # RMSE_PIVEN
    else:
        if method == 'piven':
            y_piven = y_pred_v * y_pred_U + (1 - y_pred_v) * y_pred_L

        elif method == 'mid':
            y_piven = 0.5 * y_pred_U + 0.5 * y_pred_L

        elif method == 'only-rmse':
            y_piven = y_pred_v

        MSE_ELI = np.mean(np.square(Gen.scale_c * (y_piven - y_test[:, 0])))
        RMSE_ELI = np.sqrt(MSE_ELI) # RMSE_PIVEN

    CWC = np_QD_loss(y_test, y_pred_L, y_pred_U, alpha, lambda_in)  # from qd paper.
    neg_log_like = gauss_neg_log_like(y_test, y_pred_gauss_mid, y_pred_gauss_dev, Gen.scale_c)
    residuals = y_pred_mid - y_test[:, 0]
    shapiro_W, shapiro_p = stats.shapiro(residuals[:])
    results_runs.append((PICP, MPIW, CWC, RMSE, RMSE_ELI, neg_log_like, shapiro_W, shapiro_p))

# summarize results
# results_path = f"./Results/{method}/"
# results_path += f"{params['dataset']}-{start_time.strftime('%d-%m-%H-%M')}-{method}.csv"

# results = np.array(results_runs)
# results_to_csv(results_path, results, params, n_runs, n_ensemble, in_ddof)

# timing info
# end_time = datetime.datetime.now()
# total_time = end_time - start_time
# print('\n\nminutes taken:', round(total_time.total_seconds() / 60, 3),
#       '\nstart_time:', start_time.strftime('%H:%M:%S'),
#       'end_time:', end_time.strftime('%H:%M:%S'))

# with open(results_path, 'a') as results_file:
#     results_file.write(f'minutes taken,{round(total_time.total_seconds() / 60, 3)},,\n')
#     results_file.close()



### write resutls to file
with open(results_path, 'a') as fwrite:
    fwrite.write(str(seed)+' '+str(neurons)+' '+str(soften)+' '+str(lambda_in)+' '+str(sigma_in)+' '\
    +str(round(PICP,3))+' '+str(round(MPIW, 3))+ ' '+str(round(RMSE,3))+' '+str(alpha)+' '+str(h_size) + ' '\
    +str(n_epoch)+' '+str(l_rate)+' '+str(decay_rate)+' '+str(n_ensemble)+' '+str(n_runs)\
                 +'\n' )
