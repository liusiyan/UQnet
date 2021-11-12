import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from scipy import stats
from sklearn.metrics import r2_score
import math

# import edl
import evidential_deep_learning as edl
import data_loader
import trainers
import models
from models.toy.h_params import h_params
import itertools
tf.config.threading.set_intra_op_parallelism_threads(1)
import random

from function_generator_OOD import CL_function_generator  # the OOD data synthesizer from user defined function

''' 
This is the code for evidential regression (DER) method used for our comparison which was adjusted from the original DER code 
(also included in the upper level folder) or you can find it at https://github.com/aamini/evidential-deep-learning

All of our experiments on DER method with synthetic datasets were conducted on a single Ubuntu workstation, and we use Intel I9-10980xe CPU generated 
all the results instead of using a GPU because the training are relatively fast on CPU.

The results will be generated in the ./Results/ including the summary of the training results (.txt files), plotted loss curves (./Results/loss_curves/)
and loss history for each case (.csv format in ./Results/loss_history/)

We also prepared pre-generated results for your reference (in ./Pre_generated_results/)

Have fun!
'''


def standardize(data):
    mu = data.mean(axis=0, keepdims=1)
    scale = data.std(axis=0, keepdims=1)
    scale[scale < 1e-10] = 1.0

    data = (data - mu) / scale
    return data, mu, scale

# dataset_list = ['boston', 'concrete', 'energy', 'kin8nm', 'naval', 'power', 'protein', 'wine', 'yacht', 'MSD']
''' Training data generation '''
function_generator = CL_function_generator(generator_rnd_seed=1) # this random seed is used for adding various random noise

data_name = 'FUN' # customized function instead of UCI data sets
results_path = './Results/'+data_name + '_DER_results.txt'

save_loss_history = True
save_loss_history_path = './Results/loss_history/'
plot_loss_history = True
plot_loss_history_path = './Results/loss_curves/'

parser = argparse.ArgumentParser()
parser.add_argument("--num-trials", default=1, type=int,
                    help="Number of trials to repreat training for \
                    statistically significant results.")
parser.add_argument("--num-epochs", default=40, type=int)
parser.add_argument('--datasets', nargs='+', default=['FUN'], choices=['FUN'])

# split_seed_list  = [1, 2, 3, 4, 5]
random_seed_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# seed_combination_list = list(itertools.product(split_seed_list, random_seed_list))
# print('-- The splitting and random seed combination list: {}'.format(seed_combination_list))

dataset = data_name
learning_rate = 1e-4 # h_params[dataset]["learning_rate"]
batch_size = 128      # h_params[dataset]["batch_size"]

with open(results_path, 'a') as fwrite:
    fwrite.write('EXP '+'random_seed '+'Mean_PICP_test '+'Mean_MPIW_test '+'Mean_RMSE '+'Mean_NLL '+'Mean_R2 '+'num_trials '+'data '+'lr '+'batch_size '+'\n')

### used for train specific case with singe or a series of index (0-24 for random seed combinations)
iii_low = 0
iii_high = 0
for iii in range(len(random_seed_list)):
    if iii >= iii_low and iii <=iii_high:

        # split_seed = seed_combination_list[iii][0]
        seed = random_seed_list[iii]
        print('--- Running EXP {}/{}'.format(iii + 1, len(random_seed_list)))
        print('--- Dataset: {}'.format(data_name))
        print('--- Random seed: {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        args = parser.parse_args()
        args.datasets[0] = data_name
        training_schemes = [trainers.Evidential]
        datasets = args.datasets
        # print('--- Printing datasets:')
        # print(datasets)
        num_trials = args.num_trials
        print('num_trials:{}'.format(num_trials))
        # num_trials = 3
        num_epochs = args.num_epochs
        dev = "/cpu:0" # for small datasets/models cpu is faster than gpu
        """" ================================================"""

        RMSE = np.zeros((len(datasets), len(training_schemes), num_trials))
        NLL = np.zeros((len(datasets), len(training_schemes), num_trials))

        PICP_arr = np.zeros(num_trials)
        MPIW_arr = np.zeros(num_trials)
        R2_arr = np.zeros(num_trials)

        for di, dataset in enumerate(datasets):
            # print(di)
            # print(dataset)
            for ti, trainer_obj in enumerate(training_schemes):
                for n in range(num_trials):
                    print('*********************************************')
                    print('--- data: {}, trial: {}, iii range:{}, {} '.format(data_name, n+1, iii_low, iii_high))
                    print('*********************************************')

                    # (x_train, y_train), (x_test, y_test), y_scale = data_loader.load_dataset(dataset, return_as_tensor=False)
                    ''' Load the customized data with fixed splitting train/test '''
                    # x_train, y_train, x_test, y_test, y_scale = data_loader.LoadData_Splitted_UCI(data_name, original_data_path, splitted_data_path, split_seed)
                    # batch_size = h_params[dataset]["batch_size"]

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

                    # normalization
                    x_train = xTrain
                    y_train = yTrain
                    x_test = xTest
                    y_test = yTest

                    x_train, x_train_mu, x_train_scale = standardize(x_train)
                    x_test = (x_test - x_train_mu) / x_train_scale

                    y_train, y_train_mu, y_train_scale = standardize(y_train)
                    y_test = (y_test - y_train_mu) / y_train_scale

                    y_scale = y_train_scale

                    num_iterations = num_epochs * x_train.shape[0]//batch_size

                    done = False
                    while not done:
                        with tf.device(dev):
                            model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
                            model, opts = model_generator.create(input_shape=x_train.shape[1:], tf_seed=seed)

                            trainer = trainer_obj(model, opts, dataset, learning_rate=learning_rate)
                            model, rmse, nll, loss = trainer.train(x_train, y_train, x_test, y_test, y_scale, batch_size=batch_size, iters=num_iterations,
                                                             verbose=True, data_name=data_name, rnd_seed=seed, trial_num=n,
                                                                   bool_plot_loss=True, bool_save_loss=True,
                                                                   save_loss_path=save_loss_history_path,
                                                                   plot_loss_path=plot_loss_history_path)

                            ''' Evaluate the PICP and MPIW for each trial '''
                            ### taken from the 'plot_ng' function from the original evidential regression code
                            x_test_input_tf = tf.convert_to_tensor(x_test, tf.float32)
                            outputs = model(x_test_input_tf)

                            mu, v, alpha, beta = tf.split(outputs, 4, axis=1)
                            epistemic_var = np.sqrt(beta / (v * (alpha - 1)))
                            epistemic_var = np.minimum(epistemic_var, 1e3)

                            y_pred_U = mu.numpy() + epistemic_var * 1.96
                            y_pred_L = mu.numpy() - epistemic_var * 1.96
                            # print('y_pred_U: {}'.format(y_pred_U))
                            # print('y_pred_L: {}'.format(y_pred_L))

                            ''' Do same thing for training data in order to do OOD analysis '''
                            x_train_input_tf = tf.convert_to_tensor(x_train, tf.float32)
                            outputs_train = model(x_train_input_tf)

                            mu_train, v_train, alpha_train, beta_train = tf.split(outputs_train, 4, axis=1)
                            epistemic_var_train = np.sqrt(beta_train / (v_train * (alpha_train - 1)))
                            epistemic_var_train = np.minimum(epistemic_var_train, 1e3)

                            y_pred_U_train = mu_train.numpy() + epistemic_var_train * 1.96
                            y_pred_L_train = mu_train.numpy() - epistemic_var_train * 1.96

                            if np.isnan(y_pred_U).any() or np.isnan(y_pred_L).any():
                                PICP = math.nan
                                MPIW = math.nan
                                R2 = math.nan
                                rmse = math.nan
                                nll = math.nan
                                print('--- the y_pred_U/L contains NaN(s) in current trial')
                            else:

                                ''' Calculate the confidence scores (y-axis) range from 0-1'''
                                #### for train
                                y_U_cap_train = y_pred_U_train > y_train
                                y_L_cap_train = y_pred_L_train < y_train
                                MPIW_array_train = y_pred_U_train - y_pred_L_train
                                MPIW_train = np.mean(MPIW_array_train)

                                #### for test (evaluate each y_U_cap - y_L_cap in the pre-calculated MPIW_train single value
                                # for the confidence score)
                                y_U_cap = y_pred_U > y_test
                                y_L_cap = y_pred_L < y_test

                                y_all_cap = y_U_cap * y_L_cap
                                PICP = np.sum(y_all_cap) / y_L_cap.shape[0]

                                MPIW_array = y_pred_U - y_pred_L
                                MPIW = np.mean(MPIW_array)
                                # MPIW = np.mean(y_pred_U - y_pred_L)

                                confidence_arr_test = [min(MPIW_train / test_width, 1.0) for test_width in MPIW_array]
                                confidence_arr_train = [min(MPIW_train / train_width, 1.0) for train_width in MPIW_array_train]

                                ''' Calculate the L2 distance to the mean of training data (x-axis), range from 0-30'''
                                dist_arr_train = np.sqrt(np.sum(x_train ** 2.0, axis=1))
                                dist_arr_test = np.sqrt(np.sum(x_test ** 2.0, axis=1))

                                print('dist_arr_train shape: {}'.format(dist_arr_train.shape))
                                print('confidence arr train len: {}'.format(len(confidence_arr_train)))

                                print('dist_arr_test shape: {}'.format(dist_arr_test.shape))
                                print('confidence arr test len: {}'.format(len(confidence_arr_test)))

                                ''' Save to file and plot the results '''
                                confidence_arr_train = np.array(confidence_arr_train)
                                confidence_arr_test = np.array(confidence_arr_test)

                                DER_OOD_train_np = np.hstack(
                                    (dist_arr_train.reshape(-1, 1), confidence_arr_train.reshape(-1, 1)))
                                DER_OOD_test_np = np.hstack(
                                    (dist_arr_test.reshape(-1, 1), confidence_arr_test.reshape(-1, 1)))

                                np.savetxt('DER_OOD_train_np.txt', DER_OOD_train_np, delimiter=',')
                                np.savetxt('DER_OOD_test_np.txt', DER_OOD_test_np, delimiter=',')

                                plt.plot(dist_arr_train, confidence_arr_train, 'r.', label='Training data (in distribution)')
                                plt.plot(dist_arr_test, confidence_arr_test, 'b.',label='testing data (out of distribution')
                                plt.xlabel('L2 distance to the mean of training data $\{x_i\}_{i=1}^N$')
                                plt.ylabel('The Confidence Score')
                                plt.legend(loc='lower left')
                                plt.title('DER')
                                plt.ylim(0, 1.2)
                                plt.savefig('DER_OOD_plot.png')
                                # plt.show()

                                R2 = r2_score(y_test, mu.numpy())
                                print('PICP: {}, MPIW: {}, R2: {}'.format(PICP, MPIW, R2))

                            del model
                            tf.keras.backend.clear_session()
                            done = False if np.isinf(nll) or np.isnan(nll) else True

                            ### new added done criteria
                            if np.isnan(loss):
                                done = True
                    print("saving {} {}".format(rmse, nll))
                    RMSE[di, ti, n] = rmse
                    NLL[di, ti, n] = nll
                    PICP_arr[n] = PICP
                    MPIW_arr[n] = MPIW
                    R2_arr[n] = R2


        print('PICP_arr: {}'.format(PICP_arr))
        print('MPIW_arr: {}'.format(MPIW_arr))
        print('R2_arr: {}'.format(R2_arr))

        PICP_mean = np.nanmean(PICP_arr)
        MPIW_mean = np.nanmean(MPIW_arr)
        RMSE_mean = np.nanmean(RMSE)
        NLL_mean = np.nanmean(NLL)
        R2_mean = np.nanmean(R2_arr)

        print('--- Mean PICP: {}'.format(PICP_mean))
        print('--- Mean MPIW: {}'.format(MPIW_mean))
        print('--- Mean RMSE: {}'.format(RMSE_mean))
        print('--- Mean NLL: {}'.format(NLL_mean))
        print('--- Mean R2: {}'.format(R2_mean))

        RESULTS = np.hstack((RMSE, NLL))
        print('RESULTS: {}'.format(RESULTS))

        mu = RESULTS.mean(axis=-1)
        error = np.std(RESULTS, axis=-1)

        print("==========================")
        print("[{}]: {} pm {}".format(dataset, mu, error))
        print("==========================")

        print("TRAINERS: {}\nDATASETS: {}".format([trainer.__name__ for trainer in training_schemes], datasets))
        print("MEAN: \n{}".format(mu))
        print("ERROR: \n{}".format(error))

        ''' Save results to dat file '''
        with open(results_path, 'a') as fwrite:
            fwrite.write(str(iii + 1) + ' ' + ' ' + str(seed) + ' ' + str(round(PICP_mean, 3)) + ' ' + str(round(MPIW_mean, 3)) + ' '+ \
                         str(round(RMSE_mean, 3)) + ' ' + str(round(NLL_mean, 3))+ ' ' +str(round(R2_mean, 3)) + ' ' + str(num_trials) + ' ' + data_name + ' ' + \
                         str(learning_rate) + ' ' + str(batch_size) +'\n')

    # import pdb; pdb.set_trace()
