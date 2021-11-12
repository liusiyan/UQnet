''' The Tensorflow 2.0 version of the interval regression for UCI 10 datasets
Dataset
(1) Boston housing            --- 13 inputs, 1 output, 506 data points
(2) Concrete                  --- 8 inputs, 1 output, 1030 data points
(3) Energy-efficiency         --- 8 inputs, 2 outputs, 768 data points
(4) kin8nm                    --- 8 inputs, 1 output, 8192 data points
(5) naval   16                  --- 16 inputs, 2 outputs, 11934 data points
(6) powerplot                 --- 4 inputs, 1 output, 9568 data points
(7) protein                   --- 9 inputs, 1 output, 45730 data points
(8) wine                      --- 11 inputs, 1 output, 1599 data points
(9) yacht                     --- 6 inputs, 1 output, 308 data points
(10) MSD                      --- 90 inputs, 1 output, 515,345 data points
'''


import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import datetime
from tqdm import tqdm
import time

import itertools
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Force using CPU globally by hiding GPU(s)
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from benchmark_data_loader_V2 import CL_dataLoader
from network_V2 import UQ_Net_mean_TF2, UQ_Net_std_TF2
from network_V2 import CL_UQ_Net_train_steps
from boundary_optimizer import CL_boundary_optimizer
from visualization import CL_plotter


'''
This is the code for our prediction interval method

All of our experiments on PI3NN method with UCI datasets were conducted on a single Ubuntu workstation, only one case ("MSD") was trained on a single NVIDIA RTX3090
due to its relatively larger data size. Rest of the cases were trained on a single Intel I9-10980xe CPU. You can enable the GPU training by commenting the 
above line : tf.config.set_visible_devices([], 'GPU')

To reproduce the results, simply assign the data set name to the "data_name" variable at the beginning of this code before running the main file.
Accepted data names are: 'boston', 'concrete', 'energy', 'kin8nm', 'naval', 'power', 'protein', 'wine', 'yacht', 'MSD'
The results will be generated in the ./Results_PI3NN/ including the summary of the training results (.txt files), plotted loss curves (./Results_PI3NN/loss_curves/)
and loss history for each case (.csv format in ./Results_PI3NN/loss_history/).

Note that if the GPU is used in the training, slightly different results maybe obtained (for example the MSD case in our experiment), exact same results can be
reproduced by forcing the training on CPU.

We also prepared pre-generated results for your reference (in ./Pre_generated_results/)

Have fun!

'''

''' Data selection '''
# dataset_list = ['boston', 'concrete', 'energy', 'kin8nm', 'naval', 'power', 'protein', 'wine', 'yacht', 'MSD']
# data_name = dataset_list[0]
data_name = 'yacht'
original_data_path = '../../UCI_datasets/'          ## original UCI data sets
splitted_data_path = '../../UCI_TrainTest_Split/'   ## pre-split data
results_path = './Results_PI3NN/'+data_name + '_PI3NN_results.txt'

split_seed_list  = [1, 2, 3, 4, 5]
random_seed_list = [10, 20, 30, 40, 50]
seed_combination_list = list(itertools.product(split_seed_list, random_seed_list))
print('-- The splitting and random seed combination list: {}'.format(seed_combination_list))

# seed_combination_list = [(1, 30)]
saveWeights = True
loadWeights_test = False
early_stop = True
save_loss_history = True
save_loss_history_path = './Results_PI3NN/loss_history/'
plot_loss_history = True
plot_loss_history_path = './Results_PI3NN/loss_curves/'

''' Data loader '''
dataLoader = CL_dataLoader(original_data_path)

with open(results_path, 'a') as fwrite:
    fwrite.write('EXP '+'split_seed '+'random_seed '+'PICP_test '+'MPIW_test '+'RMSE '+'R2'+'\n')

for iii in range(len(seed_combination_list)):
    if iii >= 0:
        split_seed = seed_combination_list[iii][0]
        seed = seed_combination_list[iii][1]
        print('--- Running EXP {}/{}'.format(iii+1, len(seed_combination_list)))
        print('--- Dataset: {}'.format(data_name))
        print('--- Splitting seed and random seed: {}, {}'.format(split_seed, seed))

        xyTrain_load, xyTest_load = dataLoader.LoadData_Splitted_UCI(data_name, splitted_data_path, split_seed)

        saveFigPrefix = data_name

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        num_neurons_mean_net = 100
        num_neurons_up_down_net = 100

        Max_iter = 500000  # max iteration of all training
        decay_rate = 0.8

        quantile = 0.95  # target percentile for optimization step

        if data_name == 'energy' or data_name == 'naval':
            xTrain = xyTrain_load[:, :-2] ## all columns except last two columns as inputs
            yTrain = xyTrain_load[:, -1] ## last column as output
            xTest = xyTest_load[:, :-2]
            yTest = xyTest_load[:, -1]
        else:
            xTrain = xyTrain_load[:, :-1]
            yTrain = xyTrain_load[:, -1]
            xTest = xyTest_load[:, :-1]
            yTest = xyTest_load[:, -1]


        ''' Standardize inputs '''
        # standardized_input, input_mean, input_std = dataLoader.standardizer(input)
        xTrain, xTrain_mean, xTrain_std = dataLoader.standardizer(xTrain)
        xTest = (xTest - xTrain_mean) / xTrain_std

        yTrain, yTrain_mean, yTrain_std = dataLoader.standardizer(yTrain)
        yTest = (yTest - yTrain_mean) / yTrain_std

        num_inputs = dataLoader.getNumInputsOutputs(xTrain)
        num_outputs = dataLoader.getNumInputsOutputs(yTrain)

        ''' Create network instances'''
        net_mean = UQ_Net_mean_TF2(num_inputs, num_outputs, num_neurons=num_neurons_mean_net)
        net_std_up = UQ_Net_std_TF2(num_inputs, num_outputs,  num_neurons=num_neurons_up_down_net)
        net_std_down = UQ_Net_std_TF2(num_inputs, num_outputs, num_neurons=num_neurons_up_down_net)

        ''' Initialize trainSteps instance'''
        # trainSteps = CL_UQ_Net_train_steps(net_mean,   optimizer_net_mean,
        #                                    net_std_up, optimizer_net_std_up,
        #                                    net_std_down, optimizer_net_std_down)
        trainSteps = CL_UQ_Net_train_steps(net_mean, net_std_up, net_std_down,
                                           optimizers=['Adam', 'Adam', 'Adam'], ## 'Adam', 'SGD'
                                           lr=[0.01, 0.01, 0.01],         ## order: mean, up, down
                                           exponential_decay=True,
                                           decay_steps=3000,
                                           decay_rate=decay_rate)
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

        ''' Main training iterations '''
        plotter = CL_plotter()
        train_loss_mean_list = []
        test_loss_mean_list = []
        iter_mean_list = []
        # if save_loss_history:
        #     save_loss_file_name_MEAN = data_name + '_loss_MEAN.txt'
        #     with open(save_loss_history_path + save_loss_file_name_MEAN, 'a') as fwrite_loss_MEAN:
        #         fwrite_loss_MEAN.write('EXP ' + 'split_seed ' + 'random_seed ' + 'model_type '+'iter '+'train_loss ' + 'test_loss ' + '\n')
        #
        #     save_loss_file_name_UP = data_name + '_loss_UP.txt'
        #     with open(save_loss_history_path + save_loss_file_name_UP, 'a') as fwrite_loss_UP:
        #         fwrite_loss_UP.write('EXP ' + 'split_seed ' + 'random_seed ' + 'model_type '+'iter '+'train_loss ' + 'test_loss ' + '\n')
        #
        #     save_loss_file_name_DOWN = data_name + '_loss_DOWN.txt'
        #     with open(save_loss_history_path + save_loss_file_name_DOWN, 'a') as fwrite_loss_DOWN:
        #         fwrite_loss_DOWN.write('EXP ' + 'split_seed ' + 'random_seed ' + 'model_type '+'iter '+'train_loss ' + 'test_loss ' + '\n')
        #

        ### early stopping (taken and adjusted from Tensorflow/keras early stopping function)
        if data_name == 'MSD': # 100 for MSD and 300 for rest of all
            wait_patience = 100
        else:
            wait_patience = 300
        early_stop_start_iter = 6000
        # stopped_baseline = None
        restore_best_weights = True
        verbose = 1

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

        for i in range(Max_iter):
            trainSteps.train_loss_net_mean.reset_states()
            trainSteps.test_loss_net_mean.reset_states()
            trainSteps.train_step_UQ_Net_mean_TF2(xTrain, yTrain, xTest, yTest)
            current_train_loss = trainSteps.train_loss_net_mean.result()
            current_test_loss = trainSteps.test_loss_net_mean.result()
            if i % 100 == 0:
                print('Iter: {}, train_mean loss: {}, test_mean loss: {}'.format(i, current_train_loss, current_test_loss))
            train_loss_mean_list.append(current_train_loss.numpy())
            test_loss_mean_list.append(current_test_loss.numpy())
            if early_stop and i >= early_stop_start_iter:
                if np.less(current_train_loss - min_delta, best_loss):
                    best_loss = current_train_loss
                    early_stop_wait = 0
                    if restore_best_weights:
                        best_weights = trainSteps.net_mean.get_weights()
                else:
                    early_stop_wait += 1
                    # print('--- Iter: {}, early_stop_wait: {}'.format(i+1, early_stop_wait))
                    if early_stop_wait >= wait_patience:
                        stopped_iter = i
                        stop_training = True
                        if restore_best_weights:
                            if verbose > 0:
                                print('--- Restoring mean model weights from the end of the best iteration')
                            trainSteps.net_mean.set_weights(best_weights)
                        if saveWeights:
                            print('--- Saving best model weights to h5 file: {}_best_mean_iter_{}.h5'.format(data_name, str(i+1)))
                            trainSteps.net_mean.save_weights(
                                './checkpoints_mean/'+data_name+'_best_mean_iter_' + str(i + 1) + '.h5')
            iter_mean_list.append(i)
            if stop_training:
                print('--- Early stopping criteria met.  Iteration: {}, train_loss:{}, test_loss:{}'.format(i+1, current_train_loss, current_test_loss ))
                break

            # ### Test model saving
            # if saveWeights:
            #     trainSteps.net_mean.save_weights('./checkpoints_mean/mean_checkpoint_iter_'+str(i+1)+'.h5')
        if plot_loss_history:
            plotter.plotTrainValidationLoss(train_loss_mean_list, test_loss_mean_list,
                                            trainPlotLabel='training loss', validPlotLabel='test loss',
                                            xlabel='iterations', ylabel='Loss', title='('+saveFigPrefix+')Train/test loss for mean values',
                                            gridOn=True, legendOn=True,
                                            saveFigPath=plot_loss_history_path+saveFigPrefix+'_MEAN_loss_seed_'+str(split_seed)+'_'+str(seed)+'.png')
                                            # xlim=[50, len(train_loss_mean_list)])
        if save_loss_history:
            df_loss_MEAN = pd.DataFrame({
                'iter': iter_mean_list,
                'train_loss': train_loss_mean_list,
                'test_loss': test_loss_mean_list
            })
            df_loss_MEAN.to_csv(
                save_loss_history_path + data_name + '_MEAN_loss_seed_' + str(split_seed) + '_' + str(seed) + '.csv')




        ''' Generate up and down training/validation data '''
        diff_train = (yTrain.reshape(yTrain.shape[0], -1) - trainSteps.net_mean(xTrain))
        yTrain_up_data = tf.expand_dims(diff_train[diff_train > 0], axis=1)
        xTrain_up_data = xTrain[(diff_train > 0).numpy().flatten(), :]
        yTrain_down_data = -1.0 * tf.expand_dims(diff_train[diff_train < 0], axis=1)
        xTrain_down_data = xTrain[(diff_train < 0).numpy().flatten(), :]

        diff_test = (yTest.reshape(yTest.shape[0], -1) - trainSteps.net_mean(xTest))
        yTest_up_data = tf.expand_dims(diff_test[diff_test > 0], axis=1)
        xTest_up_data = xTest[(diff_test > 0).numpy().flatten(), :]
        yTest_down_data = -1.0 * tf.expand_dims(diff_test[diff_test < 0], axis=1)
        xTest_down_data = xTest[(diff_test < 0).numpy().flatten(), :]

        # print('xTrain_up_data is {}, with shape: {}'.format(type(xTrain_up_data), xTrain_up_data.shape))
        # print('xTrain_down_data is {}, with shape: {}'.format(type(xTrain_down_data), xTrain_down_data.shape))
        # print('yTrain_up_data is {}, with shape: {}'.format(type(yTrain_up_data), yTrain_up_data.shape))
        # print('yTrain_down_data is {}, with shape: {}'.format(type(yTrain_down_data), yTrain_down_data.shape))

        # print('xTest_up_data is {}, with shape: {}'.format(type(xTest_up_data), xTest_up_data.shape))
        # print('xTest_down_data is {}, with shape: {}'.format(type(xTest_down_data), xTest_down_data.shape))
        # print('yTest_up_data is {}, with shape: {}'.format(type(yTest_up_data), yTest_up_data.shape))
        # print('yTest_down_data is {}, with shape: {}'.format(type(yTest_down_data), yTest_down_data.shape))

        xTrain_up = xTrain_up_data
        yTrain_up = yTrain_up_data.numpy()
        xTrain_down = xTrain_down_data
        yTrain_down = yTrain_down_data.numpy()

        xTest_up = xTest_up_data
        yTest_up = yTest_up_data.numpy()
        xTest_down = xTest_down_data
        yTest_down = yTest_down_data.numpy()

        print('--- Start the UP values training...')
        train_loss_up_list = []
        test_loss_up_list = []
        iter_up_list = []

        train_eval_count = 0

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

        for i in range(Max_iter):
            trainSteps.train_loss_net_std_up.reset_states()
            trainSteps.test_loss_net_std_up.reset_states()
            trainSteps.train_step_UQ_Net_std_UP_TF2(xTrain_up, yTrain_up, xTest_up, yTest_up)
            current_train_loss = trainSteps.train_loss_net_std_up.result()
            current_test_loss = trainSteps.test_loss_net_std_up.result()
            if i % 100 == 0:
                print('Iter: {}, train_up loss: {}, test_up loss: {}'.format(i, current_train_loss, current_test_loss))
            train_loss_up_list.append(current_train_loss.numpy())
            test_loss_up_list.append(current_test_loss.numpy())
            if early_stop and i >= early_stop_start_iter:
                if np.less(current_train_loss - min_delta, best_loss):
                    best_loss = current_train_loss
                    early_stop_wait = 0
                    if restore_best_weights:
                        best_weights = trainSteps.net_std_up.get_weights()
                else:
                    early_stop_wait += 1
                    # print('--- Iter: {}, early_stop_wait: {}'.format(i+1, early_stop_wait))
                    if early_stop_wait >= wait_patience:
                        stopped_iter = i
                        stop_training = True
                        if restore_best_weights:
                            if verbose > 0:
                                print('--- Restoring std_up model weights from the end of the best iteration')
                            trainSteps.net_std_up.set_weights(best_weights)
                        if saveWeights:
                            print('--- Saving best model weights to h5 file: {}_best_std_up_iter_{}.h5'.format(data_name, str(i+1)))
                            trainSteps.net_std_up.save_weights(
                                './checkpoints_up/'+data_name+'_best_std_up_iter_' + str(i + 1) + '.h5')

            iter_up_list.append(i)
            if stop_training:
                print('--- Early stopping criteria met.  Iteration: {}, train_loss:{}, test_loss:{}'.format(i+1, current_train_loss, current_test_loss ))
                break

            ### Test model saving
            # if saveWeights:
            #     trainSteps.net_std_up.save_weights('./checkpoints_up/up_checkpoint_iter_'+str(i+1)+'.h5')

        if plot_loss_history:
            plotter.plotTrainValidationLoss(train_loss_up_list, test_loss_up_list,
                                            trainPlotLabel='training loss', validPlotLabel='test loss',
                                            xlabel='iterations', ylabel='Loss', title='('+saveFigPrefix+')Train/test loss for UP values',
                                            gridOn=True, legendOn=True,
                                            saveFigPath=plot_loss_history_path+saveFigPrefix+'_UP_loss_seed_'+str(split_seed)+'_'+str(seed)+'.png')
                                            # xlim=[50, len(train_loss_up_list)])
        if save_loss_history:
            df_loss_UP = pd.DataFrame({
                'iter': iter_up_list,
                'train_loss': train_loss_up_list,
                'test_loss': test_loss_up_list
            })
            df_loss_UP.to_csv(
                save_loss_history_path + data_name + '_UP_loss_seed_' + str(split_seed) + '_' + str(seed) + '.csv')

        print('--- Start the DOWN values training...')
        train_loss_down_list = []
        test_loss_down_list = []
        iter_down_list = []

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

        for i in range(Max_iter):
            trainSteps.train_loss_net_std_down.reset_states()
            trainSteps.test_loss_net_std_down.reset_states()
            trainSteps.train_step_UQ_Net_std_DOWN_TF2(xTrain_down, yTrain_down, xTest_down, yTest_down)
            current_train_loss = trainSteps.train_loss_net_std_down.result()
            current_test_loss = trainSteps.test_loss_net_std_down.result()
            if i % 100 == 0:
                print('Iter: {}, train_down loss: {}, test_down loss: {}'.format(i, current_train_loss, current_test_loss))
            train_loss_down_list.append(current_train_loss.numpy())
            test_loss_down_list.append(current_test_loss.numpy())

            if early_stop and i >= early_stop_start_iter:
                if np.less(current_train_loss - min_delta, best_loss):
                    best_loss = current_train_loss
                    early_stop_wait = 0
                    if restore_best_weights:
                        best_weights = trainSteps.net_std_down.get_weights()
                else:
                    early_stop_wait += 1
                    # print('--- Iter: {}, early_stop_wait: {}'.format(i+1, early_stop_wait))
                    if early_stop_wait >= wait_patience:
                        stopped_iter = i
                        stop_training = True
                        if restore_best_weights:
                            if verbose > 0:
                                print('--- Restoring std_down model weights from the end of the best iteration')
                            trainSteps.net_std_down.set_weights(best_weights)
                        if saveWeights:
                            print('--- Saving best model weights to h5 file: {}_best_std_down_iter_{}.h5'.format(data_name, str(i+1)))
                            trainSteps.net_std_up.save_weights(
                                './checkpoints_down/'+data_name+'_best_std_down_iter_' + str(i + 1) + '.h5')
            iter_down_list.append(i)
            if stop_training:
                print('--- Early stopping criteria met.  Iteration: {}, train_loss:{}, test_loss:{}'.format(i+1, current_train_loss, current_test_loss ))
                break

            ### Test model saving
            # if saveWeights:
            #     trainSteps.net_std_down.save_weights('./checkpoints_down/down_checkpoint_iter_'+str(i+1)+'.h5')
        if plot_loss_history:
            plotter.plotTrainValidationLoss(train_loss_down_list, test_loss_down_list,
                                            trainPlotLabel='training loss', validPlotLabel='test loss',
                                            xlabel='iterations', ylabel='Loss', title='('+saveFigPrefix+')Train/test loss for DOWN values',
                                            gridOn=True, legendOn=True,
                                            saveFigPath=plot_loss_history_path+saveFigPrefix+'_DOWN_loss_seed_'+str(split_seed)+'_'+str(seed)+'.png')
                                            # xlim=[50, len(train_loss_down_list)])
        if save_loss_history:
            df_loss_DOWN = pd.DataFrame({
                'iter': iter_down_list,
                'train_loss': train_loss_down_list,
                'test_loss': test_loss_down_list
            })
            df_loss_DOWN.to_csv(
                save_loss_history_path+data_name+'_DOWN_loss_seed_'+str(split_seed)+'_'+str(seed)+'.csv')


        ''' Upper and lower bounds optimization '''
        # ------------------------------------------------------
        # Target percentile
        # quantile = 0.95
        # Determine how to move the upper and lower bounds

        Ntrain = xTrain.shape[0]
        num_outlier = int(Ntrain * (1 - quantile) / 2)
        print('-- Number of outlier based on the defined quantile: {}'.format(num_outlier))

        output = trainSteps.net_mean(xTrain, training=False)
        output_up = trainSteps.net_std_up(xTrain, training=False)
        output_down = trainSteps.net_std_down(xTrain, training=False)
        print(type(output))

        boundaryOptimizer = CL_boundary_optimizer(yTrain, output, output_up, output_down, num_outlier,
                                                  c_up0_ini=0.0,
                                                  c_up1_ini=100000.0,
                                                  c_down0_ini=0.0,
                                                  c_down1_ini=100000.0,
                                                  max_iter=1000)
        c_up = boundaryOptimizer.optimize_up(verbose=0)
        c_down = boundaryOptimizer.optimize_down(verbose=0)


        ''' Final prediction and visualization '''
        print('c_up: {}'.format(c_up))
        print('c_down: {}'.format(c_down))

        ### use testing data for final evaluation
        test_output = trainSteps.net_mean(xTest, training=False)
        test_output_up = trainSteps.net_std_up(xTest, training=False)
        test_output_down = trainSteps.net_std_down(xTest, training=False)

        train_output = trainSteps.net_mean(xTrain, training=False)
        train_output_up = trainSteps.net_std_up(xTrain, training=False)
        train_output_down = trainSteps.net_std_down(xTrain, training=False)

        ''' Added new metrics '''
        print('*********** New metrics *****************')

        print(test_output.shape)
        print(test_output_up.shape)
        print(test_output_down.shape)
        # print(test_output_down)

        print('---------------- calculate caps ----------------')
        y_U_cap_train = (train_output + c_up * train_output_up).numpy().flatten() > yTrain
        y_L_cap_train = (train_output - c_down * train_output_down).numpy().flatten() < yTrain
        y_all_cap_train = y_U_cap_train * y_L_cap_train  # logic_or
        PICP_train = np.sum(y_all_cap_train) / y_L_cap_train.shape[0]  # 0-1
        MPIW_train = np.mean((train_output + c_up * train_output_up).numpy().flatten() - (
                train_output - c_down * train_output_down).numpy().flatten())
        print('Num of train in y_U_cap_train: {}'.format(np.count_nonzero(y_U_cap_train)))
        print('Num of train in y_L_cap_train: {}'.format(np.count_nonzero(y_L_cap_train)))
        print('Num of train in y_all_cap_train: {}'.format(np.count_nonzero(y_all_cap_train)))
        print('np.sum results(train): {}'.format(np.sum(y_all_cap_train)))
        print('PICP_train: {}'.format(PICP_train))
        print('MPIW_train: {}'.format(MPIW_train))

        print('---------------- ------ ----------------')

        y_U_cap_test = (test_output + c_up * test_output_up).numpy().flatten() > yTest
        y_L_cap_test = (test_output - c_down * test_output_down).numpy().flatten() < yTest
        y_all_cap_test = y_U_cap_test * y_L_cap_test  # logic_or
        PICP = np.sum(y_all_cap_test) / y_L_cap_test.shape[0]  # 0-1
        MPIW = np.mean((test_output + c_up * test_output_up).numpy().flatten() - (
                test_output - c_down * test_output_down).numpy().flatten())
        # print('y_U_cap: {}'.format(y_U_cap))
        # print('y_L_cap: {}'.format(y_L_cap))
        print('Num of true in y_U_cap: {}'.format(np.count_nonzero(y_U_cap_test)))
        print('Num of true in y_L_cap: {}'.format(np.count_nonzero(y_L_cap_test)))
        print('Num of true in y_all_cap: {}'.format(np.count_nonzero(y_all_cap_test)))
        print('np.sum results: {}'.format(np.sum(y_all_cap_test)))
        print('PICP: {}'.format(PICP))
        print('MPIW: {}'.format(MPIW))

        print('*********** Test *****************')
        print('*********** Test *****************')
        # print(y_all_cap)
        print(np.sum(y_all_cap_test))

        MSE_test = np.mean(np.square(test_output.numpy().flatten() - yTest))
        RMSE_test = np.sqrt(MSE_test)
        R2_test = r2_score(yTest, test_output.numpy().flatten())

        print('Test MSE: {}'.format(MSE_test))
        print('Test RMSE: {}'.format(RMSE_test))
        print('Test R2: {}'.format(R2_test))

        ''' Save results to txt file '''
        with open(results_path, 'a') as fwrite:
            fwrite.write(str(iii+1)+' '+str(split_seed)+' '+str(seed)+' '+str(round(PICP,3))+' '+str(round(MPIW, 3))+ ' '
            +str(round(RMSE_test,3))+' '+str(round(R2_test, 3))+'\n' )


# print('----------------------------------------------------------')
# print('---------------------Weights loaded !!!-------------------')
# print('----------------------------------------------------------')

# evl_path = './Results/'+data_name + '_EVL_PI_results_bias_3.txt' 
# with open(evl_path, 'a') as fwrite:
#     fwrite.write('iter '+'split_seed '+'random_seed '+'PICP_test '+'MPIW_test '+'RMSE '+'R2'+'\n')

#
# if loadWeights_test:
#     # trainSteps.net_mean.save_weights('./checkpoints_mean/mean_checkpoint_iter_'+str(i+1)+'.h5')
#     # trainSteps.net_std_up.save_weights('./checkpoints_up/up_checkpoint_iter_'+str(i+1)+'.h5')
#     # trainSteps.net_std_down.save_weights('./checkpoints_down/down_checkpoint_iter_'+str(i+1)+'.h5')
#
#     for j in tqdm(range(Max_iter)):
#         trainSteps.net_mean.load_weights('./checkpoints_mean/mean_checkpoint_iter_'+str(j+1)+'.h5')
#         trainSteps.net_std_up.load_weights('./checkpoints_up/up_checkpoint_iter_'+str(j+1)+'.h5')
#         trainSteps.net_std_down.load_weights('./checkpoints_down/down_checkpoint_iter_'+str(j+1)+'.h5')
#
#
#         ## optimize up and down from training data
#         output = trainSteps.net_mean(xTrain, training=False)
#         output_up = trainSteps.net_std_up(xTrain, training=False)
#         output_down = trainSteps.net_std_down(xTrain, training=False)
#
#         boundaryOptimizer = CL_boundary_optimizer(yTrain, output, output_up, output_down, num_outlier,
#                                                   c_up0_ini=0.0,
#                                                   c_up1_ini=100000.0,
#                                                   c_down0_ini=0.0,
#                                                   c_down1_ini=100000.0,
#                                                   max_iter=1000)
#         c_up = boundaryOptimizer.optimize_up(verbose=0)
#         c_down = boundaryOptimizer.optimize_down(verbose=0)
#
#         ## apply the optimized upper/lower bounds to testing data
#         test_output = trainSteps.net_mean(xTest, training=False)
#         test_output_up = trainSteps.net_std_up(xTest, training=False)
#         test_output_down = trainSteps.net_std_down(xTest, training=False)
#
#         y_U_cap_test = (test_output + c_up * test_output_up).numpy().flatten() > yTest
#         y_L_cap_test = (test_output - c_down * test_output_down).numpy().flatten() < yTest
#         y_all_cap_test = y_U_cap_test * y_L_cap_test  # logic_or
#         PICP = np.sum(y_all_cap_test) / y_L_cap_test.shape[0]  # 0-1
#         MPIW = np.mean((test_output + c_up * test_output_up).numpy().flatten() - (
#                 test_output - c_down * test_output_down).numpy().flatten())
#         # print('y_U_cap: {}'.format(y_U_cap))
#         # print('y_L_cap: {}'.format(y_L_cap))
#         print('Num of true in y_U_cap: {}'.format(np.count_nonzero(y_U_cap_test)))
#         print('Num of true in y_L_cap: {}'.format(np.count_nonzero(y_L_cap_test)))
#         print('Num of true in y_all_cap: {}'.format(np.count_nonzero(y_all_cap_test)))
#         print('np.sum results: {}'.format(np.sum(y_all_cap_test)))
#         print('PICP: {}'.format(PICP))
#         print('MPIW: {}'.format(MPIW))
#
#         print('*********** Test *****************')
#         print('*********** Test *****************')
#         # print(y_all_cap)
#         print(np.sum(y_all_cap_test))
#
#         MSE_test = np.mean(np.square(test_output.numpy().flatten() - yTest))
#         RMSE_test = np.sqrt(MSE_test)
#         R2_test = r2_score(yTest, test_output.numpy().flatten())
#
#         print('Test MSE: {}'.format(MSE_test))
#         print('Test RMSE: {}'.format(RMSE_test))
#         print('Test R2: {}'.format(R2_test))
#
#         with open(evl_path, 'a') as fwrite:
#             fwrite.write(str(j+1)+' '+str(split_seed)+' '+str(seed)+' '+str(round(PICP,3))+' '+str(round(MPIW, 3))+ ' '
#         +str(round(RMSE_test,3))+' '+str(round(R2_test, 3))+'\n' )
#
#
#
#



