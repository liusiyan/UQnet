
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time
import math

from src.Networks.networks import UQ_Net_mean_TF2, UQ_Net_std_TF2
from src.Networks.networks import CL_UQ_Net_train_steps
from src.Visualizations.visualization import CL_plotter
from src.Optimizations.boundary_optimizer import CL_boundary_optimizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class CL_trainer:

    def __init__(self, configs, net_mean, net_std_up, net_std_down,
                 xTrain, yTrain, xValid=None, yValid=None, xTest=None, yTest=None, testDataEvaluationDuringTrain=False, flightDelayTestDataList=None):
        ''' Take all 3 network instance and the trainSteps (CL_UQ_Net_train_steps) instance '''

        self.testDataEvaluationDuringTrain = testDataEvaluationDuringTrain
        self.bool_NaN = False
        self.configs = configs
        self.net_mean = net_mean
        self.net_std_up = net_std_up
        self.net_std_down = net_std_down
        self.xTrain = xTrain
        self.yTrain = yTrain

        if xValid is not None:
            self.xValid = xValid
        if yValid is not None:
            self.yValid = yValid

        if xTest is not None:
            self.xTest = xTest
        if yTest is not None:
            self.yTest = yTest

        if flightDelayTestDataList is not None:
            self.xTest = flightDelayTestDataList[0][0]
            self.yTest = flightDelayTestDataList[0][1]

        self.trainSteps = CL_UQ_Net_train_steps(self.net_mean, self.net_std_up, self.net_std_down,
                                   # self.xTrain, self.yTrain, self.xTest, self.yTest,
                                   optimizers=self.configs['optimizers'], ## 'Adam', 'SGD'
                                   lr=self.configs['lr'],         ## order: mean, up, down
                                   exponential_decay=self.configs['exponential_decay'],
                                   decay_steps=self.configs['decay_steps'],
                                   decay_rate=self.configs['decay_rate'])


        # self.early_stop_start_iter = configs['early_stop_start_iter']
        # self.verbose = 1

        self.plotter = CL_plotter()

        self.train_loss_mean_list = []
        self.valid_loss_mean_list = []
        self.test_loss_mean_list = []
        self.iter_mean_list = []

        self.train_loss_up_list = []
        self.valid_loss_up_list = []
        self.test_loss_up_list = []
        self.iter_up_list = []

        self.train_loss_down_list = []
        self.valid_loss_down_list = []
        self.test_loss_down_list = []
        self.iter_down_list = []

        self.saveFigPrefix = self.configs['data_name']   # prefix for the saved plots



    # def train(self, xTest, yTest, flightDelayTestDataList=None, testDataEvaluation=True):
    def train(self, flightDelayTestDataList=None):
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
            fwrite.write('EXP '+'random_seed '+'PICP_test '+'MPIW_test '+'RMSE '+'R2'+'\n')


        if self.configs['batch_training'] == True:
            #### test tf.data.Dataset for mini-batch training
            train_dataset = tf.data.Dataset.from_tensor_slices((self.xTrain, self.yTrain))
            valid_dataset = tf.data.Dataset.from_tensor_slices((self.xValid, self.yValid))
            test_dataset = tf.data.Dataset.from_tensor_slices((self.xTest, self.yTest))
            if self.configs['batch_shuffle'] == True:
                train_dataset = train_dataset.shuffle(buffer_size=self.configs['batch_shuffle_buffer']).batch(self.configs['batch_size'])
                valid_dataset = valid_dataset.shuffle(buffer_size=self.configs['batch_shuffle_buffer']).batch(self.configs['batch_size'])
                test_dataset = test_dataset.shuffle(buffer_size=self.configs['batch_shuffle_buffer']).batch(self.configs['batch_size'])
            else:
                train_dataset = train_dataset.batch(self.configs['batch_size'])
                valid_dataset = valid_dataset.batch(self.configs['batch_size'])
                test_dataset = test_dataset.batch(self.configs['batch_size'])


        ''' Main training iterations '''

        #######################################
        ##### ''' Training for the MEAN ''' ###
        #######################################

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


        self.trainSteps.train_loss_net_mean.reset_states()
        self.trainSteps.valid_loss_net_mean.reset_states()
        self.trainSteps.test_loss_net_mean.reset_states()

        if self.configs['batch_training'] == False:
            for i in range(self.configs['Max_iter']):
                # self.trainSteps.train_loss_net_mean.reset_states()
                self.trainSteps.valid_loss_net_mean.reset_states()
                self.trainSteps.test_loss_net_mean.reset_states()
                # self.trainSteps.train_step_mean(self.xTrain, self.yTrain, self.xTest, self.yTest, testDataEvaluation=testDataEvaluation)
                # self.trainSteps.train_step_mean(testDataEvaluation=testDataEvaluation)
                if self.testDataEvaluationDuringTrain:
                    self.trainSteps.train_step_mean(self.xTrain, self.yTrain, self.xValid, self.yValid, xTest=self.xTest, yTest=self.yTest, 
                        testDataEvaluationDuringTrain=self.testDataEvaluationDuringTrain)
                if not self.testDataEvaluationDuringTrain:
                    self.trainSteps.train_step_mean(self.xTrain, self.yTrain, self.xValid, self.yValid, xTest=None, yTest=None, 
                        testDataEvaluationDuringTrain=self.testDataEvaluationDuringTrain)

                current_train_loss = self.trainSteps.train_loss_net_mean.result()
                current_valid_loss = self.trainSteps.valid_loss_net_mean.result()

                if math.isnan(current_train_loss) or math.isnan(current_valid_loss):
                    print('--- WARNING: NaN(s) detected, stop or go to next sets of tuning parameters...')
                    break

                if self.testDataEvaluationDuringTrain:
                    current_test_loss = self.trainSteps.test_loss_net_mean.result()
                    self.test_loss_mean_list.append(current_test_loss.numpy())                 
                    if i % 100 == 0: 
                        print('Epoch: {}, train_mean loss: {}, valid_mean loss: {}, test_mean_loss: {}'.format(i, current_train_loss, current_valid_loss, current_test_loss))
                if not self.testDataEvaluationDuringTrain:
                    if i % 100 == 0: 
                        print('Epoch: {}, train_mean loss: {}, valid_mean loss: {}'.format(i, current_train_loss, current_valid_loss))

                self.train_loss_mean_list.append(current_train_loss.numpy())
                self.valid_loss_mean_list.append(current_valid_loss.numpy())
                
                if self.configs['early_stop'] and i >= self.configs['early_stop_start_iter']:
                    if np.less(current_valid_loss - min_delta, best_loss):
                        best_loss = current_valid_loss
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
                                if best_weights is not None:
                                    if self.configs['verbose'] > 0:
                                        print('--- Restoring mean model weights from the end of the best iteration')
                                    self.trainSteps.net_mean.set_weights(best_weights)
                            if self.configs['saveWeights']:
                                print('--- Saving best model weights to h5 file: {}_best_mean_iter_{}.h5'.format(self.configs['data_name'], str(i+1)))
                                self.trainSteps.net_mean.save_weights(os.getcwd()+
                                    '/Results_PI3NN/checkpoints_mean/'+self.configs['data_name']+'_best_mean_iter_' + str(i + 1) + '.h5')
                self.iter_mean_list.append(i)
                if stop_training:
                    if self.testDataEvaluationDuringTrain:
                        print('--- Early stopping criteria met.  Epoch: {}, train_loss:{}, valid_loss:{}, test_loss:{}'.format(i+1, current_train_loss, current_valid_loss, current_test_loss))
                        break
                    if not self.testDataEvaluationDuringTrain:
                        print('--- Early stopping criteria met.  Epoch: {}, train_loss:{}, valid_loss:{}'.format(i+1, current_train_loss, current_valid_loss))
                        break 


        if self.configs['batch_training'] == True:
            bool_found_NaN = False
            for i in range(self.configs['Max_iter']):
                if bool_found_NaN:
                    print('--- Stop or go to next sets of tuning parameters due to NaN(s)')
                    break
                self.trainSteps.train_loss_net_mean.reset_states() ### mean loss of all steps in one epoch
                self.trainSteps.valid_loss_net_mean.reset_states()
                self.trainSteps.test_loss_net_mean.reset_states()

                for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                    self.trainSteps.batch_train_step_mean(x_batch_train, y_batch_train)
                    current_train_loss = self.trainSteps.train_loss_net_mean.result()

                    if math.isnan(current_train_loss):
                        print('--- WARNING: NaN(s) detected, stop or go to next sets of tuning parameters...')
                        bool_found_NaN = True
                        break
                    if (step % 100 == 0) and self.configs['verbose'] > 1:  # if i % 100 == 0:
                        print('Step: {}, train_mean loss: {}'.format(step, current_train_loss))
                self.train_loss_mean_list.append(current_train_loss.numpy())
                ### lr decay
                if self.configs['exponential_decay']:
                    self.trainSteps.global_step_0.assign_add(1)

                #### Validation loop at the end of each epoch
                for x_batch_valid, y_batch_valid in valid_dataset:
                    self.trainSteps.batch_valid_step_mean(x_batch_valid, y_batch_valid)
                    current_valid_loss = self.trainSteps.valid_loss_net_mean.result()
                self.valid_loss_mean_list.append(current_valid_loss.numpy())

                ### (optional) evaluate testing data
                if self.testDataEvaluationDuringTrain:
                    for x_batch_test, y_batch_test in test_dataset:
                        self.trainSteps.batch_test_step_mean(x_batch_test, y_batch_test)
                        current_test_loss = self.trainSteps.test_loss_net_mean.result()
                    self.test_loss_mean_list.append(current_test_loss.numpy())
                    if i % 100 == 0:
                        print('--- Epoch: {}, train loss: {}, validation loss: {}, test loss: {}'.format(i, current_train_loss, current_valid_loss, current_test_loss))
                else:
                    if i % 100 == 0:
                        print('--- Epoch: {}, train loss: {}, validation loss: {}'.format(i, current_train_loss, current_valid_loss))

                if self.configs['early_stop'] and i >= self.configs['early_stop_start_iter']:
                    if np.less(current_valid_loss - min_delta, best_loss):
                        best_loss = current_valid_loss
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
                                if best_weights is not None:
                                    if self.configs['verbose'] > 0:
                                        print('--- Restoring mean model weights from the end of the best iteration')
                                    self.trainSteps.net_mean.set_weights(best_weights)
                            if self.configs['saveWeights']:
                                print('--- Saving best model weights to h5 file: {}_best_mean_iter_{}.h5'.format(self.configs['data_name'], str(i+1)))
                                self.trainSteps.net_mean.save_weights(os.getcwd()+
                                    '/Results_PI3NN/checkpoints_mean/'+self.configs['data_name']+'_best_mean_iter_' + str(i + 1) + '.h5')
                self.iter_mean_list.append(i)
                if stop_training:
                    if self.testDataEvaluationDuringTrain:
                        print('--- Early stopping criteria met.  Epoch: {}, train_loss:{}, valid_loss:{}, test_loss:{}'.format(i+1, current_train_loss, current_valid_loss, current_test_loss))
                        break
                    if not self.testDataEvaluationDuringTrain:
                        print('--- Early stopping criteria met.  Epoch: {}, train_loss:{}, valid_loss:{}'.format(i+1, current_train_loss, current_valid_loss))
                        break 

        if self.configs['plot_loss_history']:
            self.plotter.plotTrainValidationLoss(self.train_loss_mean_list, self.valid_loss_mean_list, # test_loss=self.test_loss_mean_list,
                                            trainPlotLabel='training loss', validPlotLabel='valid loss',
                                            xlabel='iterations', ylabel='Loss', title='('+self.saveFigPrefix+')Train/valid (and test) loss for mean values',
                                            gridOn=True, legendOn=True,
                                            saveFigPath=self.configs['plot_loss_history_path']+self.saveFigPrefix+'_MEAN_loss_seed_'+str(self.configs['split_seed'])+'_'+str(self.configs['seed'])+'.png')
                                            # xlim=[50, len(train_loss_mean_list)])
        if self.configs['save_loss_history']:
            loss_mean_dict = {
                'iter': self.iter_mean_list,
                'train_loss': self.train_loss_mean_list,
                'valid_loss': self.valid_loss_mean_list
            }
            if self.testDataEvaluationDuringTrain:
                loss_mean_dict['test_loss'] = self.test_loss_mean_list

            df_loss_MEAN = pd.DataFrame(loss_mean_dict)
            df_loss_MEAN.to_csv(
                self.configs['save_loss_history_path'] + self.configs['data_name'] + '_MEAN_loss_seed_' + str(self.configs['seed']) + '.csv')



        ''' Generate up and down training/validation data '''
        diff_train = (self.yTrain.reshape(self.yTrain.shape[0], -1) - self.trainSteps.net_mean(self.xTrain, training=False))
        yTrain_up_data = tf.expand_dims(diff_train[diff_train > 0], axis=1)
        xTrain_up_data = self.xTrain[(diff_train > 0).numpy().flatten(), :]
        yTrain_down_data = -1.0 * tf.expand_dims(diff_train[diff_train < 0], axis=1)
        xTrain_down_data = self.xTrain[(diff_train < 0).numpy().flatten(), :]

        self.xTrain_up = xTrain_up_data
        self.yTrain_up = yTrain_up_data.numpy()
        self.xTrain_down = xTrain_down_data
        self.yTrain_down = yTrain_down_data.numpy()

        diff_valid = (self.yValid.reshape(self.yValid.shape[0], -1) - self.trainSteps.net_mean(self.xValid, training=False))
        yValid_up_data = tf.expand_dims(diff_valid[diff_valid > 0], axis=1)
        xValid_up_data = self.xValid[(diff_valid > 0).numpy().flatten(), :]
        yValid_down_data = -1.0 * tf.expand_dims(diff_valid[diff_valid < 0], axis=1)
        xValid_down_data = self.xValid[(diff_valid < 0).numpy().flatten(), :]

        self.xValid_up = xValid_up_data
        self.yValid_up = yValid_up_data.numpy()
        self.xValid_down = xValid_down_data
        self.yValid_down = yValid_down_data.numpy()

        if self.testDataEvaluationDuringTrain:
            diff_test = (self.yTest.reshape(self.yTest.shape[0], -1) - self.trainSteps.net_mean(self.xTest, training=False))
            yTest_up_data = tf.expand_dims(diff_test[diff_test > 0], axis=1)
            xTest_up_data = self.xTest[(diff_test > 0).numpy().flatten(), :]
            yTest_down_data = -1.0 * tf.expand_dims(diff_test[diff_test < 0], axis=1)
            xTest_down_data = self.xTest[(diff_test < 0).numpy().flatten(), :]

            self.xTest_up = xTest_up_data
            self.yTest_up = yTest_up_data.numpy()
            self.xTest_down = xTest_down_data
            self.yTest_down = yTest_down_data.numpy()


        #######################################
        ##### ''' Training for the UP ''' #####
        #######################################
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

        self.trainSteps.train_loss_net_std_up.reset_states()
        self.trainSteps.valid_loss_net_std_up.reset_states()
        self.trainSteps.test_loss_net_std_up.reset_states()

        if self.configs['batch_training'] == False:
            for i in range(self.configs['Max_iter']):
                self.trainSteps.train_loss_net_std_up.reset_states()
                self.trainSteps.valid_loss_net_std_up.reset_states()
                self.trainSteps.test_loss_net_std_up.reset_states()

                if self.testDataEvaluationDuringTrain:
                    self.trainSteps.train_step_up(self.xTrain_up, self.yTrain_up, self.xValid_up, self.yValid_up, xTest=self.xTest_up, yTest=self.yTest_up, 
                        testDataEvaluationDuringTrain=self.testDataEvaluationDuringTrain)
                if not self.testDataEvaluationDuringTrain:
                    self.trainSteps.train_step_up(self.xTrain_up, self.yTrain_up, self.xValid_up, self.yValid_up, xTest=None, yTest=None, 
                        testDataEvaluationDuringTrain=self.testDataEvaluationDuringTrain)

                current_train_loss = self.trainSteps.train_loss_net_std_up.result()
                current_valid_loss = self.trainSteps.valid_loss_net_std_up.result()

                if math.isnan(current_train_loss) or math.isnan(current_valid_loss):
                    print('--- WARNING: NaN(s) detected, stop or go to next sets of tuning parameters...')
                    break

                if self.testDataEvaluationDuringTrain:
                    current_test_loss = self.trainSteps.test_loss_net_std_up.result()
                    self.test_loss_up_list.append(current_test_loss.numpy())  
                    if i % 100 == 0:
                        print('Epoch: {}, train_mean loss: {}, valid_mean loss: {}, test_mean_loss: {}'.format(i, current_train_loss, current_valid_loss, current_test_loss))
                if not self.testDataEvaluationDuringTrain:
                    if i % 100 == 0: 
                        print('Epoch: {}, train_mean loss: {}, valid_mean loss: {}'.format(i, current_train_loss, current_valid_loss))                   

                self.train_loss_up_list.append(current_train_loss.numpy())
                self.valid_loss_up_list.append(current_valid_loss.numpy())
                

                if self.configs['early_stop'] and i >= self.configs['early_stop_start_iter']:
                    if np.less(current_valid_loss - min_delta, best_loss):
                        best_loss = current_valid_loss
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
                                if best_weights is not None:
                                    if self.configs['verbose'] > 0:
                                        print('--- Restoring std_up model weights from the end of the best iteration')
                                    self.trainSteps.net_std_up.set_weights(best_weights)
                            if self.configs['saveWeights']:
                                print('--- Saving best model weights to h5 file: {}_best_std_up_iter_{}.h5'.format(self.configs['data_name'], str(i+1)))
                                self.trainSteps.net_std_up.save_weights(os.getcwd()+
                                    '/Results_PI3NN/checkpoints_up/'+self.configs['data_name']+'_best_std_up_iter_' + str(i + 1) + '.h5')
                self.iter_up_list.append(i)
                if stop_training:
                    if self.testDataEvaluationDuringTrain:
                        print('--- Early stopping criteria met.  Iteration: {}, train_loss:{}, valid_loss:{}, test_loss:{}'.format(i+1, current_train_loss, current_valid_loss, current_test_loss))
                        break
                    if not self.testDataEvaluationDuringTrain:
                        print('--- Early stopping criteria met.  Iteration: {}, train_loss:{}, valid_loss:{}'.format(i+1, current_train_loss, current_valid_loss))
                        break

                ### Test model saving
                # if configs['saveWeights']:
                #     trainSteps.net_std_up.save_weights('./checkpoints_up/up_checkpoint_iter_'+str(i+1)+'.h5')

        if self.configs['batch_training'] == True:
            bool_found_NaN = False
            for i in range(self.configs['Max_iter']):
                if bool_found_NaN:
                    print('--- Stop or go to next sets of tuning parameters due to NaN(s)')
                    break
                self.trainSteps.train_loss_net_std_up.reset_states() ### mean loss of all steps in one epoch
                self.trainSteps.valid_loss_net_std_up.reset_states()
                self.trainSteps.test_loss_net_std_up.reset_states()

                for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                    self.trainSteps.batch_train_step_up(x_batch_train, y_batch_train)
                    current_train_loss = self.trainSteps.train_loss_net_std_up.result()

                    if math.isnan(current_train_loss):
                        print('--- WARNING: NaN(s) detected, stop or go to next sets of tuning parameters...')
                        bool_found_NaN = True
                        break
                    if (step % 100 == 0) and self.configs['verbose'] > 1:  # if i % 100 == 0:
                        print('Step: {}, train_mean loss: {}'.format(step, current_train_loss))

                self.train_loss_up_list.append(current_train_loss.numpy())
                ### lr decay
                if self.configs['exponential_decay']:
                    self.trainSteps.global_step_1.assign_add(1)

                #### Validation loop at the end of each epoch
                for x_batch_valid, y_batch_valid in valid_dataset:
                    self.trainSteps.batch_valid_step_up(x_batch_valid, y_batch_valid)
                    current_valid_loss = self.trainSteps.valid_loss_net_std_up.result()
                self.valid_loss_up_list.append(current_valid_loss.numpy())

                ### (optional) evaluate testing data
                if self.testDataEvaluationDuringTrain:
                    for x_batch_test, y_batch_test in test_dataset:
                        self.trainSteps.batch_test_step_up(x_batch_test, y_batch_test)
                        current_test_loss = self.trainSteps.test_loss_net_std_up.result()
                    self.test_loss_up_list.append(current_test_loss.numpy())
                    if i % 100 == 0: 
                        print('--- Epoch: {}, train loss: {}, validation loss: {}, test loss: {}'.format(i, current_train_loss, current_valid_loss, current_test_loss))
                else:
                    if i % 100 == 0: 
                        print('--- Epoch: {}, train loss: {}, validation loss: {}'.format(i, current_train_loss, current_valid_loss))

                    if self.configs['early_stop'] and i >= self.configs['early_stop_start_iter']:
                        if np.less(current_valid_loss - min_delta, best_loss):
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
                                    if best_weights is not None:
                                        if self.configs['verbose'] > 0:
                                            print('--- Restoring std_up model weights from the end of the best iteration')
                                        self.trainSteps.net_std_up.set_weights(best_weights)
                                if self.configs['saveWeights']:
                                    print('--- Saving best model weights to h5 file: {}_best_std_up_iter_{}.h5'.format(self.configs['data_name'], str(i+1)))
                                    self.trainSteps.net_std_up.save_weights(os.getcwd()+
                                        '/Results_PI3NN/checkpoints_up/'+self.configs['data_name']+'_best_std_up_iter_' + str(i + 1) + '.h5')

                    self.iter_up_list.append(i)
                    if stop_training:
                        if self.testDataEvaluationDuringTrain:
                            print('--- Early stopping criteria met.  Epoch: {}, train_loss:{}, valid_loss:{}, test_loss:{}'.format(i+1, current_train_loss, current_valid_loss, current_test_loss))
                            break
                        if not self.testDataEvaluationDuringTrain:
                            print('--- Early stopping criteria met.  Epoch: {}, train_loss:{}, valid_loss:{}'.format(i+1, current_train_loss, current_valid_loss))
                            break 

        if self.configs['plot_loss_history']:
            self.plotter.plotTrainValidationLoss(self.train_loss_up_list, self.valid_loss_up_list, # test_loss=self.test_loss_up_list,
                                            trainPlotLabel='training loss', validPlotLabel='valid loss',
                                            xlabel='iterations', ylabel='Loss', title='('+self.saveFigPrefix+')Train/valid (and test) loss for UP values',
                                            gridOn=True, legendOn=True,
                                            saveFigPath=self.configs['plot_loss_history_path']+self.saveFigPrefix+'_UP_loss_seed_'+str(self.configs['split_seed'])+'_'+str(self.configs['seed'])+'.png')
                                            # xlim=[50, len(train_loss_up_list)])
        if self.configs['save_loss_history']:
            loss_up_dict = {
                'iter': self.iter_up_list,
                'train_loss': self.train_loss_up_list,
                'valid_loss': self.valid_loss_up_list
            }
            if self.testDataEvaluationDuringTrain:
                loss_up_dict['test_loss'] = self.test_loss_up_list

            df_loss_UP = pd.DataFrame(loss_up_dict)
            df_loss_UP.to_csv(
                self.configs['save_loss_history_path'] + self.configs['data_name'] + '_UP_loss_seed_' + str(self.configs['seed']) + '.csv')


        #######################################
        ##### ''' Training for the DOWN ''' ###
        #######################################
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

        self.trainSteps.train_loss_net_std_down.reset_states()
        self.trainSteps.valid_loss_net_std_down.reset_states()
        self.trainSteps.test_loss_net_std_down.reset_states()


        if self.configs['batch_training'] == False:
            for i in range(self.configs['Max_iter']):
                self.trainSteps.train_loss_net_std_down.reset_states()
                self.trainSteps.valid_loss_net_std_down.reset_states()
                self.trainSteps.test_loss_net_std_down.reset_states()

                if self.testDataEvaluationDuringTrain:
                    self.trainSteps.train_step_down(self.xTrain_down, self.yTrain_down, self.xValid_down, self.yValid_down, xTest=self.xTest_down, yTest=self.yTest_down, 
                        testDataEvaluationDuringTrain=self.testDataEvaluationDuringTrain)
                if not self.testDataEvaluationDuringTrain:
                    self.trainSteps.train_step_down(self.xTrain_down, self.yTrain_down, self.xValid_down, self.yValid_down, xTest=None, yTest=None, 
                        testDataEvaluationDuringTrain=self.testDataEvaluationDuringTrain)

                current_train_loss = self.trainSteps.train_loss_net_std_down.result()
                current_valid_loss = self.trainSteps.valid_loss_net_std_down.result()
                if math.isnan(current_train_loss) or math.isnan(current_valid_loss):
                    print('--- WARNING: NaN(s) detected, stop or go to next sets of tuning parameters...')
                    break

                if self.testDataEvaluationDuringTrain:
                    current_test_loss = self.trainSteps.test_loss_net_std_down.result()
                    self.test_loss_down_list.append(current_test_loss.numpy())
                    if i % 100 == 0:
                        print('Epoch: {}, train_mean loss: {}, valid_mean loss: {}, test_mean_loss: {}'.format(i, current_train_loss, current_valid_loss, current_test_loss))
                if not self.testDataEvaluationDuringTrain:
                    if i % 100 == 0: 
                        print('Epoch: {}, train_mean loss: {}, valid_mean loss: {}'.format(i, current_train_loss, current_valid_loss))

                self.train_loss_down_list.append(current_train_loss.numpy())
                self.valid_loss_down_list.append(current_valid_loss.numpy())
                

                if self.configs['early_stop'] and i >= self.configs['early_stop_start_iter']:
                    if np.less(current_valid_loss - min_delta, best_loss):
                        best_loss = current_valid_loss
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
                                if best_weights is not None:
                                    if self.configs['verbose'] > 0:
                                        print('--- Restoring std_down model weights from the end of the best iteration')
                                    self.trainSteps.net_std_down.set_weights(best_weights)
                            if self.configs['saveWeights']:
                                print('--- Saving best model weights to h5 file: {}_best_std_down_iter_{}.h5'.format(self.configs['data_name'], str(i+1)))
                                self.trainSteps.net_std_down.save_weights(os.getcwd()+
                                    '/Results_PI3NN/checkpoints_down/'+self.configs['data_name']+'_best_std_down_iter_' + str(i + 1) + '.h5')
                self.iter_down_list.append(i)
                if stop_training:
                    if self.testDataEvaluationDuringTrain:
                        print('--- Early stopping criteria met.  Epoch: {}, train_loss:{}, valid_loss:{}, test_loss:{}'.format(i+1, current_train_loss, current_valid_loss, current_test_loss))
                        break
                    if not self.testDataEvaluationDuringTrain:
                        print('--- Early stopping criteria met.  Epoch: {}, train_loss:{}, valid_loss:{}'.format(i+1, current_train_loss, current_valid_loss))
                        break

                ### Test model saving
                # if configs['saveWeights']:
                #     trainSteps.net_std_down.save_weights('./checkpoints_down/down_checkpoint_iter_'+str(i+1)+'.h5')

        if self.configs['batch_training'] == True:
            bool_found_NaN = False
            for i in range(self.configs['Max_iter']):
                if bool_found_NaN:
                    print('--- Stop or go to next sets of tuning parameters due to NaN(s)')
                    break
                self.trainSteps.train_loss_net_std_down.reset_states() ### mean loss of all steps in one epoch
                self.trainSteps.valid_loss_net_std_down.reset_states()
                self.trainSteps.test_loss_net_std_down.reset_states()

                for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                    self.trainSteps.batch_train_step_down(x_batch_train, y_batch_train)
                    current_train_loss = self.trainSteps.train_loss_net_std_down.result()

                    if math.isnan(current_train_loss):
                        print('--- WARNING: NaN(s) detected, stop or go to next sets of tuning parameters...')
                        bool_found_NaN = True
                        break
                    if (step % 100 == 0) and self.configs['verbose'] > 1:  # if i % 100 == 0:
                        print('Step: {}, train_mean loss: {}'.format(step, current_train_loss))

                self.train_loss_down_list.append(current_train_loss.numpy())
                ### lr decay
                if self.configs['exponential_decay']:
                    self.trainSteps.global_step_2.assign_add(1)

                #### Validation loop at the end of each epoch
                for x_batch_valid, y_batch_valid in valid_dataset:
                    self.trainSteps.batch_valid_step_down(x_batch_valid, y_batch_valid)
                    current_valid_loss = self.trainSteps.valid_loss_net_std_down.result()
                self.valid_loss_down_list.append(current_valid_loss.numpy())

                ### (optional) evaluate testing data
                if self.testDataEvaluationDuringTrain:
                    for x_batch_test, y_batch_test in test_dataset:
                        self.trainSteps.batch_test_step_down(x_batch_test, y_batch_test)
                        current_test_loss = self.trainSteps.test_loss_net_std_down.result()
                    self.test_loss_down_list.append(current_test_loss.numpy())
                    if i % 100 == 0: 
                        print('--- Epoch: {}, train loss: {}, validation loss: {}, test loss: {}'.format(i, current_train_loss, current_valid_loss, current_test_loss))
                else:
                    if i % 100 == 0: 
                        print('--- Epoch: {}, train loss: {}, validation loss: {}'.format(i, current_train_loss, current_valid_loss))

                    if self.configs['early_stop'] and i >= self.configs['early_stop_start_iter']:
                        if np.less(current_valid_loss - min_delta, best_loss):
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
                                    if best_weights is not None:
                                        if self.configs['verbose'] > 0:
                                            print('--- Restoring std_down model weights from the end of the best iteration')
                                        self.trainSteps.net_std_down.set_weights(best_weights)
                                if self.configs['saveWeights']:
                                    print('--- Saving best model weights to h5 file: {}_best_std_down_iter_{}.h5'.format(self.configs['data_name'], str(i+1)))
                                    self.trainSteps.net_std_down.save_weights(os.getcwd()+
                                        '/Results_PI3NN/checkpoints_down/'+self.configs['data_name']+'_best_std_down_iter_' + str(i + 1) + '.h5')
                    
                    self.iter_down_list.append(i)
                    if stop_training:
                        if self.testDataEvaluationDuringTrain:
                            print('--- Early stopping criteria met.  Epoch: {}, train_loss:{}, valid_loss:{}, test_loss:{}'.format(i+1, current_train_loss, current_valid_loss, current_test_loss))
                            break
                        if not self.testDataEvaluationDuringTrain:
                            print('--- Early stopping criteria met.  Epoch: {}, train_loss:{}, valid_loss:{}'.format(i+1, current_train_loss, current_valid_loss))
                            break 
                    ### Test model saving
                    # if configs['saveWeights']:
                    #     trainSteps.net_std_down.save_weights('./checkpoints_down/down_checkpoint_iter_'+str(i+1)+'.h5')





        if self.configs['plot_loss_history']:
            self.plotter.plotTrainValidationLoss(self.train_loss_down_list, self.valid_loss_down_list, # test_loss=self.test_loss_down_list,
                                            trainPlotLabel='training loss', validPlotLabel='valid loss',
                                            xlabel='iterations', ylabel='Loss', title='('+self.saveFigPrefix+')Train/valid (and test) loss for DOWN values',
                                            gridOn=True, legendOn=True,
                                            saveFigPath=self.configs['plot_loss_history_path']+self.saveFigPrefix+'_DOWN_loss_seed_'+str(self.configs['seed'])+'.png')
                                            # xlim=[50, len(train_loss_down_list)])
        if self.configs['save_loss_history']:
            loss_down_dict = {
                'iter': self.iter_down_list,
                'train_loss': self.train_loss_down_list,
                'valid_loss': self.valid_loss_down_list
            }
            if self.testDataEvaluationDuringTrain:
                loss_down_dict['test_loss'] = self.test_loss_down_list

            df_loss_DOWN = pd.DataFrame(loss_down_dict)
            df_loss_DOWN.to_csv(
                self.configs['save_loss_history_path']+self.configs['data_name']+'_DOWN_loss_seed_'+str(self.configs['seed'])+'.csv')


        ### test prediction 

        # train_output = self.trainSteps.net_mean(self.xTrain, training=False)
        # train_output_up = self.trainSteps.net_std_up(self.xTrain, training=False)
        # train_output_down = self.trainSteps.net_std_down(self.xTrain, training=False)


    def boundaryOptimization(self, verbose=0):
        Ntrain = self.xTrain.shape[0]
        output = self.trainSteps.net_mean(self.xTrain, training=False)
        output_up = self.trainSteps.net_std_up(self.xTrain, training=False)
        output_down = self.trainSteps.net_std_down(self.xTrain, training=False)

        try:
            self.configs['quantile_list']
            if self.configs['quantile_list'] is not None: ### use quantile_list to override the single quantile'
                if verbose > 0:
                    print('--- Start boundary optimizations for MULTIPLE quantiles...')
                self.quantile_list = self.configs['quantile_list']
                num_outlier_list = [int(Ntrain * (1 - x) / 2) for x in self.quantile_list]
                # if verbose > 0:
                #     print('-- Number of outlier based on the defined quantile:')
                #     print(num_outlier_list)
                boundaryOptimizer = CL_boundary_optimizer(self.yTrain, output, output_up, output_down,
                                          c_up0_ini=0.0,
                                          c_up1_ini=100000.0,
                                          c_down0_ini=0.0,
                                          c_down1_ini=100000.0,
                                          max_iter=1000)

                self.c_up_list = [boundaryOptimizer.optimize_up(outliers=x, verbose=0) for x in num_outlier_list]
                self.c_down_list = [boundaryOptimizer.optimize_down(outliers=x, verbose=0) for x in num_outlier_list]
                if verbose > 0:
                    for idx, item in enumerate(self.quantile_list):
                        print('--- Quantile: {:.4f}, outlisers: {}, c_up: {:.4f}, c_down: {:.4f}'.format(item, num_outlier_list[idx], self.c_up_list[idx], self.c_down_list[idx]))
                    # print('c_up: {}'.format(self.c_up_list))
                    # print('c_down: {}'.format(self.c_down_list))
            else:
                print('--- WARNING: configs[\'quantile_list\'] exist but missing values (list of quantile values) or is None' )
                print('--- Code stop running. Please assign values or comment out the line: configs[\'quantile_list\'] = xxx')
                exit()

        except KeyError:
            num_outlier = int(Ntrain * (1 - self.configs['quantile']) / 2)
            if verbose > 0:
                print('--- Start boundary optimizations for SINGLE quantile: {}'.format(self.configs['quantile']))
                print('--- Number of outlier based on the defined quantile: {}'.format(num_outlier))
            boundaryOptimizer = CL_boundary_optimizer(self.yTrain, output, output_up, output_down, num_outlier=num_outlier,
                                                      c_up0_ini=0.0,
                                                      c_up1_ini=100000.0,
                                                      c_down0_ini=0.0,
                                                      c_down1_ini=100000.0,
                                                      max_iter=1000)
            self.c_up = boundaryOptimizer.optimize_up(verbose=0)
            self.c_down = boundaryOptimizer.optimize_down(verbose=0)
            if verbose > 0:
                print('--- c_up: {}'.format(self.c_up))
                print('--- c_down: {}'.format(self.c_down))

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

            self.valid_output = self.trainSteps.net_mean(self.xValid, training=False)
            self.valid_output_up = self.trainSteps.net_std_up(self.xValid, training=False)
            self.valid_output_down = self.trainSteps.net_std_down(self.xValid, training=False) 

            self.test_output = self.trainSteps.net_mean(self.xTest, training=False)
            self.test_output_up = self.trainSteps.net_std_up(self.xTest, training=False)
            self.test_output_down = self.trainSteps.net_std_down(self.xTest, training=False)            

    def capsCalculation(self, final_evaluation=False, flightDelayTestDataList=None, verbose=0):


        if hasattr(self, 'quantile_list'):
            if self.quantile_list is not None:
                if verbose > 0:
                    print('--- Start caps calculations for MULTIPLE quantiles...')
                    print('**************** For Training data *****************')
                ### caps calculations for multiple quantiles (training data)
                if len(self.yTrain.shape) == 2:
                    self.yTrain = self.yTrain.flatten() # single y output 
                y_U_cap_train_list = [(self.train_output + x * self.train_output_up).numpy().flatten() > self.yTrain for x in self.c_up_list]
                y_L_cap_train_list = [(self.train_output - x * self.train_output_down).numpy().flatten() < self.yTrain for x in self.c_down_list]
                y_all_cap_train_list = [(x * y) for x,y in zip(y_U_cap_train_list, y_L_cap_train_list)]
                self.PICP_train_list = [np.sum(x / y.shape[0]) for x,y in zip(y_all_cap_train_list, y_L_cap_train_list)]
                self.MPIW_train_list = [np.mean((self.train_output + x * self.train_output_up).numpy().flatten() - (
                                        self.train_output - y * self.train_output_down).numpy().flatten()) for x,y in zip(self.c_up_list, self.c_down_list)]
                if verbose > 0:
                    for idx, item in enumerate(self.quantile_list):
                        print('--- Train quantile: {:.4f}, PICP: {:.4f}, MPIW: {:.4f}'.format(item, self.PICP_train_list[idx], self.MPIW_train_list[idx]))

                ### caps calculations for multiple quantiles (validation data)
                if verbose > 0:
                    print('**************** For Validation data *****************')
                if len(self.yValid.shape) == 2:
                    self.yValid = self.yValid.flatten()
                y_U_cap_valid_list = [(self.valid_output + x * self.valid_output_up).numpy().flatten() > self.yValid for x in self.c_up_list]
                y_L_cap_valid_list = [(self.valid_output - x * self.valid_output_down).numpy().flatten() < self.yValid for x in self.c_down_list]
                y_all_cap_valid_list = [(x * y) for x,y in zip(y_U_cap_valid_list, y_L_cap_valid_list)]
                self.PICP_valid_list = [np.sum(x / y.shape[0]) for x,y in zip(y_all_cap_valid_list, y_L_cap_valid_list)]
                self.MPIW_valid_list = [np.mean((self.valid_output + x * self.valid_output_up).numpy().flatten() - (
                                        self.valid_output - y * self.valid_output_down).numpy().flatten()) for x,y in zip(self.c_up_list, self.c_down_list)]
                if verbose > 0:
                    print('--- Quantiles for validation data:')
                    for idx, item in enumerate(self.quantile_list):
                        print('--- Valid quantile: {:.4f}, PICP: {:.4f}, MPIW: {:.4f}'.format(item, self.PICP_valid_list[idx], self.MPIW_valid_list[idx]))

                ### caps calculations for multiple quantiles (testing data, for final evaluation)
                if final_evaluation:
                    if verbose > 0:
                        print('**************** For Testing data *****************')
                    if len(self.yTest.shape) == 2:
                        self.yTest = self.yTest.flatten()
                    y_U_cap_test_list = [(self.test_output + x * self.test_output_up).numpy().flatten() > self.yTest for x in self.c_up_list]
                    y_L_cap_test_list = [(self.test_output - x * self.test_output_down).numpy().flatten() < self.yTest for x in self.c_down_list]
                    y_all_cap_test_list = [(x * y) for x,y in zip(y_U_cap_test_list, y_L_cap_test_list)]
                    self.PICP_test_list = [np.sum(x / y.shape[0]) for x,y in zip(y_all_cap_test_list, y_L_cap_test_list)]
                    self.MPIW_test_list = [np.mean((self.test_output + x * self.test_output_up).numpy().flatten() - (
                                            self.test_output - y * self.test_output_down).numpy().flatten()) for x,y in zip(self.c_up_list, self.c_down_list)]
                    if verbose > 0:
                        print('--- Quantiles for testing data:')
                        for idx, item in enumerate(self.quantile_list):
                            print('--- Test quantile: {:.4f}, PICP: {:.4f}, MPIW: {:.4f}'.format(item, self.PICP_test_list[idx], self.MPIW_test_list[idx]))

                    self.MSE_test = np.mean(np.square(self.test_output.numpy().flatten() - self.yTest))
                    self.RMSE_test = np.sqrt(self.MSE_test)
                    self.R2_test = r2_score(self.yTest, self.test_output.numpy().flatten())
                    if verbose > 0:
                        print('Test MSE: {}'.format(self.MSE_test))
                        print('Test RMSE: {}'.format(self.RMSE_test))
                        print('Test R2: {}'.format(self.R2_test)) 
                # return 
            else:
                print('--- WARNING: configs[\'quantile_list\'] exist but missing values (list of quantile values) or is None' )
                print('--- Code stop running. Please assign values or comment out the line: configs[\'quantile_list\'] = xxx')
                exit()
        else:
            ### caps calculations for single quantile 
            if verbose > 0:
                print('--- Start caps calculations for SINGLE quantile: {}'.format(self.configs['quantile']))
                print('**************** For Training data *****************')
            if len(self.yTrain.shape) == 2:
                self.yTrain = self.yTrain.flatten()
            y_U_cap_train = (self.train_output + self.c_up * self.train_output_up).numpy().flatten() > self.yTrain
            y_L_cap_train = (self.train_output - self.c_down * self.train_output_down).numpy().flatten() < self.yTrain

            y_all_cap_train = y_U_cap_train * y_L_cap_train  # logic_or
            self.PICP_train = np.sum(y_all_cap_train) / y_L_cap_train.shape[0]  # 0-1
            self.MPIW_train = np.mean((self.train_output + self.c_up * self.train_output_up).numpy().flatten() - (
                    self.train_output - self.c_down * self.train_output_down).numpy().flatten())
            if verbose > 0:
                print('Num of train in y_U_cap_train: {}'.format(np.count_nonzero(y_U_cap_train)))
                print('Num of train in y_L_cap_train: {}'.format(np.count_nonzero(y_L_cap_train)))
                print('Num of train in y_all_cap_train: {}'.format(np.count_nonzero(y_all_cap_train)))
                print('np.sum results(train): {}'.format(np.sum(y_all_cap_train)))
                print('PICP_train: {}'.format(self.PICP_train))
                print('MPIW_train: {}'.format(self.MPIW_train))

            ### for validation data
            if verbose > 0:
                print('**************** For Validation data *****************')
            if len(self.yValid.shape) == 2:
                self.yValid = self.yValid.flatten()
            y_U_cap_valid = (self.valid_output + self.c_up * self.valid_output_up).numpy().flatten() > self.yValid
            y_L_cap_valid = (self.valid_output - self.c_down * self.valid_output_down).numpy().flatten() < self.yValid
            y_all_cap_valid = y_U_cap_valid * y_L_cap_valid  # logic_or
            self.PICP_valid = np.sum(y_all_cap_valid) / y_L_cap_valid.shape[0]  # 0-1
            self.MPIW_valid = np.mean((self.valid_output + self.c_up * self.valid_output_up).numpy().flatten() - (
                    self.valid_output - self.c_down * self.valid_output_down).numpy().flatten())
            if verbose > 0:
                print('Num of valid in y_U_cap_valid: {}'.format(np.count_nonzero(y_U_cap_valid)))
                print('Num of valid in y_L_cap_valid: {}'.format(np.count_nonzero(y_L_cap_valid)))
                print('Num of valid in y_all_cap_valid: {}'.format(np.count_nonzero(y_all_cap_valid)))
                print('np.sum results(valid): {}'.format(np.sum(y_all_cap_valid)))
                print('PICP_valid: {}'.format(self.PICP_valid))
                print('MPIW_valid: {}'.format(self.MPIW_valid))

            if final_evaluation:
                if verbose > 0:
                    print('**************** For Testing data *****************')
                if len(self.yTest.shape) == 2:
                    self.yTest = self.yTest.flatten()
                y_U_cap_test = (self.test_output + self.c_up * self.test_output_up).numpy().flatten() > self.yTest
                y_L_cap_test = (self.test_output - self.c_down * self.test_output_down).numpy().flatten() < self.yTest
                y_all_cap_test = y_U_cap_test * y_L_cap_test  # logic_or
                self.PICP_test = np.sum(y_all_cap_test) / y_L_cap_test.shape[0]  # 0-1
                self.MPIW_test = np.mean((self.test_output + self.c_up * self.test_output_up).numpy().flatten() - (
                        self.test_output - self.c_down * self.test_output_down).numpy().flatten())
                # print('y_U_cap: {}'.format(y_U_cap))
                # print('y_L_cap: {}'.format(y_L_cap))
                if verbose > 0:
                    print('Num of true in y_U_cap: {}'.format(np.count_nonzero(y_U_cap_test)))
                    print('Num of true in y_L_cap: {}'.format(np.count_nonzero(y_L_cap_test)))
                    print('Num of true in y_all_cap: {}'.format(np.count_nonzero(y_all_cap_test)))
                    print('np.sum results: {}'.format(np.sum(y_all_cap_test)))
                    print('PICP_test: {}'.format(self.PICP_test))
                    print('MPIW_test: {}'.format(self.MPIW_test))

                # print(y_all_cap)
                # print(np.sum(y_all_cap_test))

                self.MSE_test = np.mean(np.square(self.test_output.numpy().flatten() - self.yTest))
                self.RMSE_test = np.sqrt(self.MSE_test)
                self.R2_test = r2_score(self.yTest, self.test_output.numpy().flatten())
                if verbose > 0:
                    print('Test MSE: {}'.format(self.MSE_test))
                    print('Test RMSE: {}'.format(self.RMSE_test))
                    print('Test R2: {}'.format(self.R2_test))    
            else:
                self.PICP_test = None
                self.MPIW_test = None

            return self.PICP_train, self.PICP_valid, self.PICP_test, self.MPIW_train, self.MPIW_valid, self.MPIW_test

        # if self.configs['data_name'] == 'flight_delay_test_five':
        #     self.PICP_test_list = []
        #     self.MPIW_test_list = []
        #     self.MSE_test_list = []
        #     self.RMSE_test_list = []
        #     self.R2_test_list = []
        #     print('  Test_case       PICP          MPIW           MSE            RMSE           R2   ')
        #     for i in range(len(self.flightDelayTestOutputList)):
        #         y_U_cap_test = (self.flightDelayTestOutputList[i][0] + self.c_up * self.flightDelayTestOutputList[i][1]).numpy().flatten() > flightDelayTestDataList[i][1]
        #         y_L_cap_test = (self.flightDelayTestOutputList[i][0] - self.c_down * self.flightDelayTestOutputList[i][2]).numpy().flatten() < flightDelayTestDataList[i][1]
        #         y_all_cap_test = y_U_cap_test * y_L_cap_test  # logic_or
        #         PICP_test = np.sum(y_all_cap_test) / y_L_cap_test.shape[0]  # 0-1
        #         MPIW_test = np.mean((self.flightDelayTestOutputList[i][0] + self.c_up * self.flightDelayTestOutputList[i][1]).numpy().flatten() - (
        #                 self.flightDelayTestOutputList[i][0] - self.c_down * self.flightDelayTestOutputList[i][2]).numpy().flatten())
        #         MSE_test = np.mean(np.square(self.flightDelayTestOutputList[i][0].numpy().flatten() - flightDelayTestDataList[i][1]))
        #         RMSE_test = np.sqrt(MSE_test)
        #         R2_test = r2_score(flightDelayTestDataList[i][1], self.flightDelayTestOutputList[i][0].numpy().flatten())

        #         self.PICP_test_list.append(PICP_test)
        #         self.MPIW_test_list.append(MPIW_test)
        #         self.MSE_test_list.append(MSE_test)
        #         self.RMSE_test_list.append(RMSE_test)
        #         self.R2_test_list.append(R2_test)

        #         print('{:10d}    {:10.4f}    {:10.4f}    {:10.4f}     {:10.4f}     {:10.4f}   '.format(i+1, PICP_test, MPIW_test, MSE_test, RMSE_test, R2_test))

        #     return self.PICP_test_list, self.MPIW_test_list

    def saveResultsToTxt(self):
        ''' Save results to txt file '''
        results_path = './Results_PI3NN/'+ self.configs['data_name'] + '_PI3NN_results.txt'
        with open(results_path, 'a') as fwrite:
            fwrite.write(str(self.configs['experiment_id'])+' '+str(self.configs['seed'])+' '+str(round(self.PICP_test,3))+' '+str(round(self.MPIW_test, 3))+ ' '
            +str(round(self.RMSE_test,3))+' '+str(round(self.R2_test, 3))+'\n' )



    def save_PI(self, bias=0.0):

        y_U_PI_array_train = (self.train_output + self.c_up * self.train_output_up).numpy().flatten()
        y_L_PI_array_train = (self.train_output - self.c_down * self.train_output_down).numpy().flatten()

        y_U_PI_array_test = (self.test_output + self.c_up * self.test_output_up).numpy().flatten()
        y_L_PI_array_test = (self.test_output - self.c_down * self.test_output_down).numpy().flatten()


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
