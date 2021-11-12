
''' Off-load the main training loops '''

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time

from src.Networks.network_V2 import UQ_Net_mean_TF2, UQ_Net_std_TF2
from src.Networks.network_V2 import CL_UQ_Net_train_steps
from src.Visualizations.visualization import CL_plotter
from src.Optimizations.boundary_optimizer import CL_boundary_optimizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats


class CL_trainer:

    def __init__(self, configs, net_mean, net_std_up, net_std_down,
                 xTrain, yTrain, xTest=None, yTest=None, flightDelayTestDataList=None):
        ''' Take all 3 network instance and the trainSteps (CL_UQ_Net_train_steps) instance '''
        self.configs = configs
        self.net_mean = net_mean
        self.net_std_up = net_std_up
        self.net_std_down = net_std_down
        self.xTrain = xTrain
        self.yTrain = yTrain
        if xTest is not None:
            self.xTest = xTest
        if yTest is not None:
            self.yTest = yTest

        if flightDelayTestDataList is not None:
            self.xTest = flightDelayTestDataList[0][0]
            self.yTest = flightDelayTestDataList[0][1]


        self.trainSteps = CL_UQ_Net_train_steps(self.net_mean, self.net_std_up, self.net_std_down,
                                   self.xTrain, self.yTrain, self.xTest, self.yTest,
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



    # def train(self, xTest, yTest, flightDelayTestDataList=None, testDataEvaluation=True):
    def train(self, flightDelayTestDataList=None, testDataEvaluation=True):
        ## only print out the intermediate test evaluation for first testing data for simplicity
        # self.xTest = xTest
        # self.yTest = yTest

        if self.configs['data_name'] == 'flight_delay_test_five':
            if flightDelayTestDataList is not None:
                self.xTest = flightDelayTestDataList[0][0]
                self.yTest = flightDelayTestDataList[0][1]

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

        if self.configs['batch_training'] == True:
            #### test tf.data.Dataset for mini-batch training
            train_dataset = tf.data.Dataset.from_tensor_slices((self.xTrain, self.yTrain))
            if self.configs['batch_shuffle'] == True:
                train_dataset = train_dataset.shuffle(buffer_size=self.configs['batch_shuffle_buffer']).batch(self.configs['batch_size'])

        # test_dataset = tf.data.Dataset.from_tensor_slices((self.xTest, self.yTest))
        # test_dataset = test_dataset.batch(1024)

        self.trainSteps.train_loss_net_mean.reset_states()
        self.trainSteps.test_loss_net_mean.reset_states()

        t11 = 0
        if self.configs['batch_training'] == False:
            for i in range(self.configs['Max_iter']):
                self.trainSteps.train_loss_net_mean.reset_states()
                self.trainSteps.test_loss_net_mean.reset_states()
                # print(i)
                # t1 = time.time()
                # self.trainSteps.train_step_UQ_Net_mean_TF2(self.xTrain, self.yTrain, self.xTest, self.yTest, testDataEvaluation=testDataEvaluation)
                self.trainSteps.train_step_UQ_Net_mean_TF2(testDataEvaluation=testDataEvaluation)
                # print('--- Iter: {}, Time used: {:.4f}s'.format(i, time.time() - t1))
                current_train_loss = self.trainSteps.train_loss_net_mean.result()
                current_test_loss = self.trainSteps.test_loss_net_mean.result()
                if i % 100 == 0:  # if i % 100 == 0:
                    print('Iter: {}, train_mean loss: {}, test_mean loss: {}'.format(i, current_train_loss, current_test_loss))
                    # print('-- Time used: {:.4f} s'.format(time.time() - t11))
                    # t11 = time.time()

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

        if self.configs['batch_training'] == True:
            for i in range(self.configs['Max_iter']):
                # print(i)
                #### test tf.data.Dataset for mini-batch training
                for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                    self.trainSteps.train_loss_net_mean.reset_states()
                    self.trainSteps.test_loss_net_mean.reset_states()
                    # self.trainSteps.train_step_UQ_Net_mean_TF2(self.xTrain, self.yTrain, self.xTest, self.yTest)
                    self.trainSteps.train_step_UQ_Net_mean_TF2(x_batch_train, y_batch_train, self.xTest, self.yTest, testDataEvaluation=testDataEvaluation)
                    current_train_loss = self.trainSteps.train_loss_net_mean.result()
                    current_test_loss = self.trainSteps.test_loss_net_mean.result()
                    if step % 100 == 0:  # if i % 100 == 0:
                        print('Iter: {}, step: {}, train_mean loss: {}, test_mean loss: {}'.format(i, step, current_train_loss, current_test_loss))

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

        if self.configs['batch_training'] == False:
            for i in range(self.configs['Max_iter']):
                self.trainSteps.train_loss_net_std_up.reset_states()
                self.trainSteps.test_loss_net_std_up.reset_states()
                # t2 = time.time()
                self.trainSteps.train_step_UQ_Net_std_UP_TF2(testDataEvaluation=testDataEvaluation)
                # print('--- Iter: {}, Time used: {:.4f}s'.format(i, time.time() - t2))
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

        if self.configs['batch_training'] == True:
            for i in range(self.configs['Max_iter']):
                #### test tf.data.Dataset for mini-batch training
                for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                    self.trainSteps.train_loss_net_std_up.reset_states()
                    self.trainSteps.test_loss_net_std_up.reset_states()
                    self.trainSteps.train_step_UQ_Net_std_UP_TF2(x_batch_train, y_batch_train, self.xTest_up, self.yTest_up, testDataEvaluation=testDataEvaluation)
                    current_train_loss = self.trainSteps.train_loss_net_std_up.result()
                    current_test_loss = self.trainSteps.test_loss_net_std_up.result()
                    if i % 100 == 0:
                        print('Iter: {}, step: {}, train_up loss: {}, test_up loss: {}'.format(i, step, current_train_loss, current_test_loss))
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


        if self.configs['batch_training'] == False:
            for i in range(self.configs['Max_iter']):
                self.trainSteps.train_loss_net_std_down.reset_states()
                self.trainSteps.test_loss_net_std_down.reset_states()
                self.trainSteps.train_step_UQ_Net_std_DOWN_TF2(testDataEvaluation=testDataEvaluation)
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

        if self.configs['batch_training'] == True:
            for i in range(self.configs['Max_iter']):
                #### test tf.data.Dataset for mini-batch training
                for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                    self.trainSteps.train_loss_net_std_down.reset_states()
                    self.trainSteps.test_loss_net_std_down.reset_states()
                    self.trainSteps.train_step_UQ_Net_std_DOWN_TF2(x_batch_train, y_batch_train, self.xTest_down, self.yTest_down, testDataEvaluation=testDataEvaluation)
                    current_train_loss = self.trainSteps.train_loss_net_std_down.result()
                    current_test_loss = self.trainSteps.test_loss_net_std_down.result()
                    if i % 100 == 0:
                        print('Iter: {}, step:{}, train_down loss: {}, test_down loss: {}'.format(i, step, current_train_loss, current_test_loss))
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




    def testDataPrediction(self, flightDelayTestDataList=None):
        if self.configs['data_name'] == 'flight_delay_test_five':
            self.train_output = self.trainSteps.net_mean(self.xTrain, training=False)
            self.train_output_up = self.trainSteps.net_std_up(self.xTrain, training=False)
            self.train_output_down = self.trainSteps.net_std_down(self.xTrain, training=False)

            self.flightDelayTestOutputList = []
            for i in range(len(flightDelayTestDataList)):
                test_output = self.trainSteps.net_mean(flightDelayTestDataList[i][0], training=False)
                test_output_up = self.trainSteps.net_std_up(flightDelayTestDataList[i][0], training=False)
                test_output_down = self.trainSteps.net_std_down(flightDelayTestDataList[i][0], training=False)
                self.flightDelayTestOutputList.append((test_output, test_output_up, test_output_down))

        else:
            self.train_output = self.trainSteps.net_mean(self.xTrain, training=False)
            self.train_output_up = self.trainSteps.net_std_up(self.xTrain, training=False)
            self.train_output_down = self.trainSteps.net_std_down(self.xTrain, training=False)

            self.test_output = self.trainSteps.net_mean(self.xTest, training=False)
            self.test_output_up = self.trainSteps.net_std_up(self.xTest, training=False)
            self.test_output_down = self.trainSteps.net_std_down(self.xTest, training=False)            

    def capsCalculation(self, flightDelayTestDataList=None):
        print('---------------- calculate caps ----------------')
        y_U_cap_train = (self.train_output + self.c_up * self.train_output_up).numpy().flatten() > self.yTrain
        y_L_cap_train = (self.train_output - self.c_down * self.train_output_down).numpy().flatten() < self.yTrain
        y_all_cap_train = y_U_cap_train * y_L_cap_train  # logic_or
        PICP_train = np.sum(y_all_cap_train) / y_L_cap_train.shape[0]  # 0-1

        train_PIW_arr = (self.train_output + self.c_up * self.train_output_up).numpy().flatten() - (
                self.train_output - self.c_down * self.train_output_down).numpy().flatten()
        MPIW_train = np.mean(train_PIW_arr)
        print('Num of train in y_U_cap_train: {}'.format(np.count_nonzero(y_U_cap_train)))
        print('Num of train in y_L_cap_train: {}'.format(np.count_nonzero(y_L_cap_train)))
        print('Num of train in y_all_cap_train: {}'.format(np.count_nonzero(y_all_cap_train)))
        print('np.sum results(train): {}'.format(np.sum(y_all_cap_train)))
        print('PICP_train: {}'.format(PICP_train))
        print('MPIW_train: {}'.format(MPIW_train))

        print('---------------- ------ ----------------')

        if self.configs['data_name'] == 'flight_delay_test_five':
            self.PICP_test_list = []
            self.MPIW_test_list = []
            self.MSE_test_list = []
            self.RMSE_test_list = []
            self.R2_test_list = []
            test_PIW_arr_list = []
            print('  Test_case       PICP          MPIW           MSE            RMSE           R2   ')
            for i in range(len(self.flightDelayTestOutputList)):
                y_U_cap_test = (self.flightDelayTestOutputList[i][0] + self.c_up * self.flightDelayTestOutputList[i][1]).numpy().flatten() > flightDelayTestDataList[i][1]
                y_L_cap_test = (self.flightDelayTestOutputList[i][0] - self.c_down * self.flightDelayTestOutputList[i][2]).numpy().flatten() < flightDelayTestDataList[i][1]
                y_all_cap_test = y_U_cap_test * y_L_cap_test  # logic_or
                PICP_test = np.sum(y_all_cap_test) / y_L_cap_test.shape[0]  # 0-1

                test_PIW_arr = (self.flightDelayTestOutputList[i][0] + self.c_up * self.flightDelayTestOutputList[i][1]).numpy().flatten() - (
                        self.flightDelayTestOutputList[i][0] - self.c_down * self.flightDelayTestOutputList[i][2]).numpy().flatten()
                test_PIW_arr_list.append(test_PIW_arr)

                MPIW_test = np.mean(test_PIW_arr)
                MSE_test = np.mean(np.square(self.flightDelayTestOutputList[i][0].numpy().flatten() - flightDelayTestDataList[i][1]))
                RMSE_test = np.sqrt(MSE_test)
                R2_test = r2_score(flightDelayTestDataList[i][1], self.flightDelayTestOutputList[i][0].numpy().flatten())

                self.PICP_test_list.append(PICP_test)
                self.MPIW_test_list.append(MPIW_test)
                self.MSE_test_list.append(MSE_test)
                self.RMSE_test_list.append(RMSE_test)
                self.R2_test_list.append(R2_test)

                print('{:10d}    {:10.4f}    {:10.4f}    {:10.4f}     {:10.4f}     {:10.4f}   '.format(i+1, PICP_test, MPIW_test, MSE_test, RMSE_test, R2_test))

            # return self.PICP_test_list, self.MPIW_test_list

        else:
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


        # #### PIW analysis
        # np.savetxt('MPIW_train_bias.dat', train_PIW_arr)
        # np.savetxt('MPIW_test_bias_r_4.dat', test_PIW_arr_list[0])
        # np.savetxt('MPIW_test_bias_r_3.dat', test_PIW_arr_list[1])
        # np.savetxt('MPIW_test_bias_r_2.dat', test_PIW_arr_list[2])
        # np.savetxt('MPIW_test_bias_r_1.dat', test_PIW_arr_list[3])


        # kde_train = stats.gaussian_kde(train_PIW_arr)

        # kde_test_list = [stats.gaussian_kde(arr) for arr in test_PIW_arr_list]

        # x1 = np.linspace(train_PIW_arr.min(), train_PIW_arr.max(), 100)
        # p1 = kde_train(x1)

        # x2 = np.linspace(test_PIW_arr_list[0].min(), test_PIW_arr_list[0].max(), 100)
        # p2 = kde_test_list[0](x2)

        # x3 = np.linspace(test_PIW_arr_list[1].min(), test_PIW_arr_list[1].max(), 100)
        # p3 = kde_test_list[1](x3)

        # x4 = np.linspace(test_PIW_arr_list[2].min(), test_PIW_arr_list[2].max(), 100)
        # p4 = kde_test_list[2](x4)

        # x5 = np.linspace(test_PIW_arr_list[3].min(), test_PIW_arr_list[3].max(), 100)
        # p5 = kde_test_list[3](x5)

        # plt.plot(x1,p1, label='train')
        # plt.plot(x2,p2, label='test_rank_4')
        # plt.plot(x3,p3, label='test_rank_3')
        # plt.plot(x4,p4, label='test_rank_2')
        # plt.plot(x5,p5, label='test_rank_1')
        # plt.xlim([0,2500])
        # plt.legend()
        # plt.savefig('PI3NN_flight_delay_bias.png')
        # plt.show()
        # plt.clf()

        # ###------- Option I: calculate confidence interval
        # conf_scores = [kde_train(arr)/p1.max() for arr in test_PIW_arr_list]
        # print('--Conf_score for rank 4, mean: {}, STD: {}'.format(np.mean(conf_scores[0]), np.std(conf_scores[0])))
        # print('--Conf_score for rank 3, mean: {}, STD: {}'.format(np.mean(conf_scores[1]), np.std(conf_scores[1])))
        # print('--Conf_score for rank 2, mean: {}, STD: {}'.format(np.mean(conf_scores[2]), np.std(conf_scores[2])))
        # print('--Conf_score for rank 1, mean: {}, STD: {}'.format(np.mean(conf_scores[3]), np.std(conf_scores[3])))

        # plt.plot([np.mean(score) for score in conf_scores])
        # plt.yscale('log')
        # plt.title('Conf_scores_mean from R4 to R1')
        # plt.savefig('PI3NN_flight_delay_conf_scores_mean.png')
        # plt.show()
        # plt.clf()

        # plt.plot([np.std(score) for score in conf_scores])
        # plt.yscale('log')
        # plt.title('Conf_scores_STD from R4 to R1')
        # plt.savefig('PI3NN_flight_delay_conf_scores_STD.png')
        # plt.show()
        # plt.clf()

    def save_PI(self, bias=0.0):

        y_U_PI_array_train = (self.train_output + self.c_up * self.train_output_up).numpy().flatten()
        y_L_PI_array_train = (self.train_output - self.c_down * self.train_output_down).numpy().flatten()

        y_U_PI_array_test = (self.test_output + self.c_up * self.test_output_up).numpy().flatten()
        y_L_PI_array_test = (self.test_output - self.c_down * self.test_output_down).numpy().flatten()


        # print(y_U_PI_array_train.shape)
        # print(y_L_PI_array_train.shape)
        # print(y_U_PI_array_test.shape)
        # print(y_L_PI_array_test.shape)

        path = './Results_PI3NN/npy/'
        train_bounds = np.vstack((y_U_PI_array_train, y_L_PI_array_train))
        test_bounds = np.vstack((y_U_PI_array_test, y_L_PI_array_test))
        np.save(path+'train_bounds'+'_bias_'+str(bias)+'.npy', train_bounds)
        np.save(path+'test_bounds'+'_bias_'+str(bias)+'.npy', test_bounds)
        np.save(path+'yTrain'+'_bias_'+str(bias)+'.npy', self.yTrain)
        np.save(path+'yTest'+'_bias_'+str(bias)+'.npy', self.yTest)
        print('--- results npy saved')



    def load_and_plot_PI(self, bias=0.0):

        path = './Results_PI3NN/npy/'
        path_fig = './Results_PI3NN/plots/'

        train_bounds = np.load(path+'train_bounds'+'_bias_'+str(bias)+'.npy')
        test_bounds = np.load(path+'test_bounds'+'_bias_'+str(bias)+'.npy')
        yTrain = np.load(path+'yTrain'+'_bias_'+str(bias)+'.npy')
        yTest = np.load(path+'yTest'+'_bias_'+str(bias)+'.npy')

        fig, ax = plt.subplots(1)
        x_train_arr = np.arange(len(train_bounds[0]))
        x_test_arr = np.arange(len(train_bounds[0]), len(train_bounds[0]) + len(test_bounds[0]))

        ax.scatter(x_train_arr, train_bounds[0], s=0.01, label='Train UP')
        ax.scatter(x_train_arr, train_bounds[1], s=0.01, label='Train DOWN')

        ax.scatter(x_test_arr, test_bounds[0], s=0.01, label='Test UP')
        ax.scatter(x_test_arr, test_bounds[1], s=0.01, label='Test DOWN')

        ax.scatter(x_train_arr, yTrain, s=0.01, label='yTrain')
        ax.scatter(x_test_arr, yTest, s=0.01, label='yTest')

        plt.title('PI3NN bounds prediction for flight delay data, bias:{}'.format(bias))
        plt.grid()
        plt.legend()
        plt.savefig(path_fig+'bounds'+'_bias_'+str(bias)+'.png', dpi=300)
        # plt.show()

    # def plot_PICPMPIW




    # def plot_PICPMPIW(self, savingPath=None):
    #   # df = self.plot_df
    #   # df = pd.read_csv('test_results.csv')
    #   # print(df)

    #   fig, (ax1, ax2) = plt.subplots(1, 2)
    #   xlabels = ['700k', '2m', '3m', '4m', '5m']
    #   plt.suptitle('PICP/MPIW for various testing data away from the training data')

    #   ax1.set_xlabel('Starting point of testing data')
    #   ax1.set_ylabel('PICP')
    #   ax1.set_title('PICP')
    #   ax1.plot(df['Test_case'].values, df['PICP'].values, 'o-')
    #   ax1.set_xticks(df['Test_case'].values)
    #   ax1.set_xticklabels(xlabels)
    #   ax1.set_aspect('auto')

    #   ax2.set_xlabel('Starting point of testing data')
    #   ax2.set_ylabel('MPIW')
    #   ax2.set_title('MPIW')

    #   ax2.plot(df['Test_case'].values, df['MPIW'].values, 'o-')
    #   ax2.set_xticks(df['Test_case'].values)
    #   ax2.set_xticklabels(xlabels)
    #   ax2.set_aspect('auto')
    #   # if savingPath is not None:
    #   #   plt.savefig(savingPath)
    #   plt.show()


    #   self.PICP_array = y_all_cap_test


    def loadPlot_confidenceScores(self, flightDelayTestDataList=None, filesPath=None, saveFigPath=None):


        test_labels = ['Martin', '2', '3', '4', '5', 'Memorial', '6', 'Independence', '7', '8', 'Labor', '9', '10', '11', 'Thanksgiving', '12', 'Christmas']

        print(len(test_labels))

        with open(filesPath+'PI3NN_OOD_flight_delay_noMIN_'+ str(0+1) +'_train_np.npy', 'rb') as f1:
            train_loaded_np = np.load(f1)

        tmp_confidence_score_train = train_loaded_np[:, 1]
        plt.hist(tmp_confidence_score_train, bins='auto', facecolor='r',  alpha=0.55, label='Training data (in distribution)')
        plt.xlim([0, 20])
        plt.title('Train distribution')
        plt.savefig(saveFigPath+'train_hist.png')
        # plt.show()
        plt.clf()

        for test_idx in range(len(flightDelayTestDataList)):
            print(test_idx)
            ### load the file
            # with open(filesPath+'PI3NN_OOD_flight_delay_noMIN_'+ str(test_idx+1) +'_train_np.npy', 'rb') as f1:
            #   train_loaded_np = np.load(f1)
            with open(filesPath+'PI3NN_OOD_flight_delay_noMIN_'+ str(test_idx+1) +'_test_np.npy', 'rb') as f2:
                test_loaded_np = np.load(f2)

            # train_loaded_np = np.loadtxt(filesPath+'PI3NN_OOD_flight_delay_noMIN_'+ str(test_idx+1) +'_train_np.txt', delimiter=',')
            # test_loaded_np = np.loadtxt(filesPath+'PI3NN_OOD_flight_delay_noMIN_'+ str(test_idx+1) +'_test_np.txt', delimiter=',')

            ### plot histogram
            # print(train_loaded_np.shape)

            # print(test_loaded_np[:,1])
            # print(train_loaded_np)
            tmp_confidence_score_test = test_loaded_np[:, 1]
            # print(tmp_confidence_score)
            plt.hist(tmp_confidence_score_test, bins='auto', facecolor='b',  alpha=0.55, label='testing data (out of distribution)')
            plt.xlim([0, 20])
            plt.title('Test distribution, month/holiday: {}'.format(test_labels[test_idx]))
            plt.savefig(saveFigPath+'test_hist_'+str(test_idx+1)+'.png')
            # plt.show()
            plt.clf()
        
            # plt.hist(confidence_arr_train, 500, facecolor='r',  alpha=0.55, label='Training data (in distribution)')
            # plt.hist(confidence_arr_test, 500, facecolor='b',  alpha=0.55, label='testing data (out of distribution)')



    def MPIW_hist_save(self, flightDelayTestDataList=None):
        '''Generate the histogram of the MPIW_train and MPIW_test '''
        print('MPIW histrogram...')

        MPIW_array_train = (self.train_output + self.c_up * self.train_output_up).numpy().flatten() - (self.train_output - self.c_down * self.train_output_down).numpy().flatten()

        print('--- Shape of MPIW_array_train: {}'.format(MPIW_array_train.shape))
        with open('./examples/flight_delay/PI3NN_flight_delay/MPIW_hist/PI3NN_flight_delay_MPIW_train.npy', 'wb') as f1:
            np.save(f1, MPIW_array_train)

        if self.configs['data_name'] == 'flight_delay_test_five':
            if flightDelayTestDataList is not None:
                for test_idx in range(len(self.flightDelayTestOutputList)):
                    MPIW_array_test = (self.flightDelayTestOutputList[test_idx][0] + self.c_up * self.flightDelayTestOutputList[test_idx][1]).numpy().flatten() - (self.flightDelayTestOutputList[test_idx][0] - self.c_down * self.flightDelayTestOutputList[test_idx][2]).numpy().flatten()
                    print('--- Test: {}, MPIW_array_test shape: {}'.format(test_idx+1, MPIW_array_test.shape))
                    with open('./examples/flight_delay/PI3NN_flight_delay/MPIW_hist/PI3NN_OOD_flight_delay_test_'+ str(test_idx+1) +'.npy', 'wb') as f2:
                        np.save(f2, MPIW_array_test)

    def MPIW_hist_plot(self, flightDelayTestDataList=None, filesPath=None, saveFigPath=None):
        test_labels = ['Martin', '2', '3', '4', '5', 'Memorial', '6', 'Independence', '7', '8', 'Labor', '9', '10', '11', 'Thanksgiving', '12', 'Christmas']
        print('--- Plotting MPIW_train_array histogram')
        with open(filesPath+'PI3NN_flight_delay_MPIW_train.npy', 'rb') as f1:
            MPIW_train_array = np.load(f1)

        print(MPIW_train_array)
        # plt.hist(MPIW_train_array, bins='auto', facecolor='r',  alpha=0.55, label='Training data MPIW (in distribution)')
        plt.hist(MPIW_train_array, bins='auto', facecolor='r',  alpha=0.55, label='Training data MPIW (in distribution)')
        plt.xlim([0, 200])
        plt.title('Train MPIW distribution')
        plt.savefig(saveFigPath+'MPIW_train_hist.png')
        print('--- Mean MPIW train: {}, AVG: {}'.format(np.mean(MPIW_train_array), np.average(MPIW_train_array)))

        # plt.clf()

        # for test_idx in range(len(flightDelayTestDataList)):
        #   print('--- Plotting MPIW_test_array histogram : {}'.format(test_idx+1))
        #   with open(filesPath+'PI3NN_OOD_flight_delay_test_'+ str(test_idx+1) +'.npy', 'rb') as f2:
        #       MPIW_test_array = np.load(f2)
        #       plt.hist(MPIW_test_array, bins='auto', facecolor='b',  alpha=0.55, label='Testing data MPIW (out of distribution)')
        #       plt.xlim([0, 200])
        #       plt.title('Test MPIW distribution, month/holiday: {}'.format(test_labels[test_idx]))
        #       plt.savefig(saveFigPath+'MPIW_test_hist_'+str(test_idx+1)+'.png')
        #       # plt.show()
        #       plt.clf()
        #   print('--- Mean MPIW test {}: {}, AVG: {}'.format(test_idx+1, np.mean(MPIW_test_array), np.average(MPIW_test_array)))





    def confidenceScoreCalculation(self, flightDelayTestDataList=None):
        ''' confidence score calculation for OOD analysis for PI3NN method '''

        print('--- OOD analysis......')

        MPIW_array_train = (self.train_output + self.c_up * self.train_output_up).numpy().flatten() - (self.train_output - self.c_down * self.train_output_down).numpy().flatten()
        MPIW_train = np.mean(MPIW_array_train)
        # MPIW_array_train = self.train_output_up + self.train_output_down
        # MPIW_array_train = MPIW_array_train.numpy()
        # MPIW_train = np.mean(MPIW_array_train) + 3.92 * np.std(MPIW_array_train)
        

        confidence_arr_train = np.array([min(MPIW_train / train_width, 1.0) for train_width in MPIW_array_train])
        # print(confidence_arr_train)
        print('--- Train conf_scores MEAN: {}, STD: {}'.format(np.mean(confidence_arr_train), np.std(confidence_arr_train)))

        if self.configs['data_name'] == 'flight_delay_test_five':
            if flightDelayTestDataList is not None:
                for test_idx in range(len(self.flightDelayTestOutputList)):
                    # MPIW_array_test = test_output_up + test_output_down
                    # MPIW_array_test = self.flightDelayTestOutputList[test_idx][1] + self.flightDelayTestOutputList[test_idx][2]
                    # MPIW_array_test = MPIW_array_test.numpy()


                    MPIW_array_test = (self.flightDelayTestOutputList[test_idx][0] + self.c_up * self.flightDelayTestOutputList[test_idx][1]).numpy().flatten() - (self.flightDelayTestOutputList[test_idx][0] - self.c_down * self.flightDelayTestOutputList[test_idx][2]).numpy().flatten()

                    # MPIW_array_test = (self.test_output + self.c_up * self.test_output_up).numpy().flatten() - (
     #              test_output - c_down * test_output_down).numpy().flatten()

                    confidence_arr_test = [min(MPIW_train / test_width, 1.0) for test_width in MPIW_array_test]
                    # confidence_arr_train = [min(MPIW_train / train_width, 1.0) for train_width in MPIW_array_train]
                    print('--- Test: {} rank: {} ranconf_scores MEAN: {}, STD: {}'.format(test_idx+1, 4-test_idx, np.mean(confidence_arr_test), np.std(confidence_arr_test)))

                    # confidence_arr_test = [(MPIW_train / test_width) for test_width in MPIW_array_test]
                    # confidence_arr_train = [(MPIW_train / train_width) for train_width in MPIW_array_train]


                    ###### confidence scores shifting

                    ''' Calculate the L2 distance to the mean of training data (x-axis), range from 0-30'''
                    # dist_arr_train = np.sqrt(np.sum(x_train ** 2.0, axis=1))
                    # dist_arr_test = np.sqrt(np.sum(x_test ** 2.0, axis=1))

                    #### calculate dist
                    dist_arr_train = np.sqrt(np.sum(self.xTrain ** 2.0, axis=1))
                    dist_arr_test = np.sqrt(np.sum(flightDelayTestDataList[test_idx][0] ** 2.0, axis=1))
                    # dist_arr_test_list = [np.sqrt(np.sum(xTest ** 2.0, axis=1)) for xTest in flightDelayTestDataList[0]] 

                    # print('dist_arr_train shape: {}'.format(dist_arr_train.shape))
                    # print('confidence arr train len: {}'.format(len(confidence_arr_train)))

                    # print('dist_arr_test shape: {}'.format(dist_arr_test.shape))
                    # print('confidence arr test len: {}'.format(len(confidence_arr_test)))

                    ''' Save to file and plot the results '''
                    confidence_arr_train = np.array(confidence_arr_train)
                    confidence_arr_test = np.array(confidence_arr_test)

                    PI3NN_OOD_train_np = np.hstack(
                        (dist_arr_train.reshape(-1, 1), confidence_arr_train.reshape(-1, 1)))
                    PI3NN_OOD_test_np = np.hstack(
                        (dist_arr_test.reshape(-1, 1), confidence_arr_test.reshape(-1, 1)))

                    # np.savetxt('./examples/flight_delay/PI3NN_flight_delay/confidence_scores_txt/PI3NN_OOD_flight_delay_noMIN_'+ str(test_idx+1) +'_train_np.txt', PI3NN_OOD_train_np, delimiter=',')
                    # np.savetxt('./examples/flight_delay/PI3NN_flight_delay/confidence_scores_txt/PI3NN_OOD_flight_delay_noMIN_'+ str(test_idx+1) +'_test_np.txt', PI3NN_OOD_test_np, delimiter=',')

                    with open('./conf_scores/PI3NN_OOD_flight_delay_'+ str(test_idx+1) +'_train_np.npy', 'wb') as f1:
                        np.save(f1, PI3NN_OOD_train_np)

                    with open('./conf_scores/PI3NN_OOD_flight_delay_'+ str(test_idx+1) +'_test_np.npy', 'wb') as f2:
                        np.save(f2, PI3NN_OOD_test_np)

                    # print('confidence scores saved')

                    # plt.hist(confidence_arr_train, 500, facecolor='r',  alpha=0.55, label='Training data (in distribution)')
                    # plt.hist(confidence_arr_test, 500, facecolor='b',  alpha=0.55, label='testing data (out of distribution)')

                    # plt.xlim([0, 1000])

                    # # plt.plot(dist_arr_train, confidence_arr_train, 'r.', label='Training data (in distribution)')
                    # # plt.plot(dist_arr_test, confidence_arr_test, 'b.',label='testing data (out of distribution')
                    # plt.xlabel('L2 distance to the mean of training data $\{x_i\}_{i=1}^N$')
                    # plt.ylabel('The Confidence Score')
                    # plt.legend(loc='lower left')
                    # plt.title('PI3NN flight delay test case '+ str(test_idx+1))
                    # # plt.ylim(0, 1.2)
                    # plt.savefig('./examples/flight_delay/PI3NN_flight_delay/PI3NN_OOD_flight_delay_monthly_holiday_HIST_'+str(test_idx+1)+'.png')
                    # # plt.savefig('~/Dropbox/NeurIPS_2021_models/flight_delay_results/Confidence_Scores_plots/PI3NN_OOD_flight_delay_'+str(test_idx+1)+'.png')
                    # # plt.show()
                    # plt.clf()

            #### load and print out confidence score stats

        else:
            pass










    def saveResultsToTxt(self):
        ''' Save results to txt file '''
        results_path = './Results_PI3NN/'+ self.configs['data_name'] + '_PI3NN_results.txt'
        with open(results_path, 'a') as fwrite:
            fwrite.write(str(self.configs['experiment_id'])+' '+str(self.configs['split_seed'])+' '+str(self.configs['seed'])+' '+str(round(self.PICP_test,3))+' '+str(round(self.MPIW_test, 3))+ ' '
            +str(round(self.RMSE_test,3))+' '+str(round(self.R2_test, 3))+'\n' )

    def saveResultsToTxt_flightDelay(self, plot=False):
        '''Save flight delay results to txt file from pandas dataframe '''
        results_path = './Results_PI3NN/'+ self.configs['data_name'] + '_PI3NN_results.txt'
        df = pd.DataFrame({
            'Test_case': ['1', '2', '3', '4', '5'],
            'PICP': self.PICP_test_list,
            'MPIW': self.MPIW_test_list,
            'MSE': self.MSE_test_list,
            'RMSE': self.RMSE_test_list,
            'R2_test_list': self.R2_test_list
            })
        df.to_csv(results_path, header=True, index=False, sep='\t', mode='a')
        print('Results saved to: '+ results_path)
        df.to_csv('test_results.csv')
        self.tmp_plot_df = df


    def saveResultsToTxt_flightDelay_monthly_holiday(self, plot=False):
        '''Save flight delay results to txt file from pandas dataframe '''
        results_path = './Results_PI3NN/'+ self.configs['data_name'] + '_monthly_holiday_PI3NN_results.txt'
        df = pd.DataFrame({
            'Test_case': ['Martin', '2', '3', '4', '5', 'Memorial', '6', 'Independence', '7', '8', 'Labor', '9', '10', '11', 'Thanksgiving', '12', 'Christmas'],
            'PICP': self.PICP_test_list,
            'MPIW': self.MPIW_test_list,
            'MSE': self.MSE_test_list,
            'RMSE': self.RMSE_test_list,
            'R2_test_list': self.R2_test_list
            })
        df.to_csv(results_path, header=True, index=False, sep='\t', mode='w')
        print('Results saved to: '+ results_path)
        # df.to_csv('PI3NN_flightDelay_monthly_holiday_results.csv')
        self.tmp_plot_df = df


    def plot_PICPMPIW_flightDelay_monthly_holiday(self, filePath=None, savingPath=None):
        # df = self.plot_df
        # df = pd.read_csv('test_results.csv')  # flight_delay_test_five_monthly_holiday_PI3NN_results.txt
        # print(df)

        df = pd.read_csv(filePath, delim_whitespace=True)
        print(df)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(12, 8))
        xlabels = ['Martin', '2', '3', '4', '5', 'Memorial', '6', 'Independence', '7', '8', 'Labor', '9', '10', '11', 'Thanksgiving', '12', 'Christmas']
        plt.suptitle('PICP/MPIW/RMSE/R2 for various testing data away from the 1st month training data')

        ax1.set_xlabel('Starting point of testing data')
        ax1.set_ylabel('PICP')
        ax1.set_title('PICP')
        ax1.plot(np.arange(len(df)), df['PICP'].values, 'o-')
        ax1.set_xticks(np.arange(len(df)))
        ax1.set_xticklabels(xlabels)
        ax1.set_aspect('auto')
        plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='x-small')
        ax1.grid()

        ax2.set_xlabel('Starting point of testing data')
        ax2.set_ylabel('MPIW')
        ax2.set_title('MPIW')
        ax2.plot(np.arange(len(df)), df['MPIW'].values, 'o-')
        ax2.set_xticks(np.arange(len(df)))
        ax2.set_xticklabels(xlabels)
        ax2.set_aspect('auto')
        plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='x-small')
        ax2.grid()

        ax3.set_xlabel('Starting point of testing data')
        ax3.set_ylabel('RMSE')
        ax3.set_title('RMSE')
        ax3.plot(np.arange(len(df)), df['RMSE'].values, 'o-')
        ax3.set_xticks(np.arange(len(df)))
        ax3.set_xticklabels(xlabels)
        ax3.set_aspect('auto')
        plt.setp(ax3.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='x-small')
        ax3.grid()

        ax4.set_xlabel('Starting point of testing data')
        ax4.set_ylabel('R2')
        ax4.set_title('R2')
        ax4.plot(np.arange(len(df)), df['R2_test_list'].values, 'o-')
        ax4.set_xticks(np.arange(len(df)))
        ax4.set_xticklabels(xlabels)
        ax4.set_aspect('auto')
        plt.setp(ax4.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='small')
        ax4.grid()
        if savingPath is not None:
            plt.savefig(savingPath)
        plt.show()






