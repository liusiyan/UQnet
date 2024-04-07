
''' encoder trainer (currently with LSTM) for time-series data'''
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib
# matplotlib.use("TkAgg")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import math
import copy

num_threads = 1
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)

from pi3nn.Networks.networks_TS import LSTM_mean_TF2, LSTM_PI_TF2
from pi3nn.Networks.networks_TS import CL_LSTM_train_steps
from pi3nn.Visualizations.visualization import CL_plotter
from pi3nn.Optimizations.boundary_optimizer import CL_boundary_optimizer, CL_prediction_shifter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pi3nn.Utils.Utils import CL_Utils
utils = CL_Utils()

# tf.keras.backend.set_floatx('float64') ## to avoid TF casting prediction to float32
class CL_trainer_TS:

    def __init__(self, configs, xTrain, yTrain, net_lstm_mean, net_lstm_up=None, net_lstm_down=None, xValid=None, yValid=None, xTest=None, yTest=None, train_idx=None, valid_idx=None,\
        scalerx=None, scalery=None, testDataEvaluationDuringTrain=False, allTestDataEvaluationDuringTrain=False):

        self.bool_Nan = False
        self.configs = copy.deepcopy(configs)
        self.net_lstm_mean = net_lstm_mean
        self.net_lstm_up = net_lstm_up
        self.net_lstm_down = net_lstm_down

        self.plotter = CL_plotter(self.configs)

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

        if scalerx is not None:
            self.scalerx = scalerx
        if scalery is not None:
            self.scalery = scalery

        if train_idx is not None:
            self.train_idx = train_idx
        if valid_idx is not None:
            self.valid_idx = valid_idx

        self.testDataEvaluationDuringTrain = testDataEvaluationDuringTrain
        self.allTestDataEvaluationDuringTrain = allTestDataEvaluationDuringTrain

        self.trainSteps_lstm = CL_LSTM_train_steps(self.net_lstm_mean, self.net_lstm_up, self.net_lstm_down,
                                                    optimizers_lstm=self.configs['optimizers_lstm'],
                                                    lr_lstm=self.configs['lr_lstm'],
                                                    exponential_decay=self.configs['exponential_decay'],
                                                    decay_steps=self.configs['decay_steps'],
                                                    decay_rate=self.configs['decay_rate'])         

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

        
        print('--- Mean: {}'.format(self.yTrain.mean()))
        print('--- Variance: {}'.format(self.yTrain.var()))
        print('--- STD: {}'.format(self.yTrain.std()))
        print('--- Max: {}'.format(self.yTrain.max()))
        print('--- Min: {}'.format(self.yTrain.min()))

        ### test simple weighted MSE
        self.w_trainY = np.abs(self.yTrain - self.yTrain.mean())

        ### convert train/valid/test Numpy data to Tensors and tf.Dataset
        if self.configs['batch_training'] == True:
            # self.train_dataset = tf.data.Dataset.from_tensor_slices((self.xTrain, self.yTrain))
            # self.valid_dataset = tf.data.Dataset.from_tensor_slices((self.xValid, self.yValid))
            # self.test_dataset = tf.data.Dataset.from_tensor_slices((self.xTest, self.yTest))
            self.train_dataset = tf.data.Dataset.from_tensor_slices((self.xTrain, self.yTrain, self.w_trainY))
            self.valid_dataset = tf.data.Dataset.from_tensor_slices((self.xValid, self.yValid))
            self.test_dataset = tf.data.Dataset.from_tensor_slices((self.xTest, self.yTest))
            if self.configs['batch_shuffle'] == True:
                self.train_dataset = self.train_dataset.shuffle(buffer_size=self.configs['batch_shuffle_buffer']).batch(self.configs['batch_size'])
                self.valid_dataset = self.valid_dataset.shuffle(buffer_size=self.configs['batch_shuffle_buffer']).batch(self.configs['batch_size'])
                self.test_dataset = self.test_dataset.shuffle(buffer_size=self.configs['batch_shuffle_buffer']).batch(self.configs['batch_size'])
            else:
                self.train_dataset = self.train_dataset.batch(self.configs['batch_size'])
                self.valid_dataset = self.valid_dataset.batch(self.configs['batch_size'])
                self.test_dataset = self.test_dataset.batch(self.configs['batch_size'])


        print('--- xTrain_mean shape: {}'.format(self.xTrain.shape))
        print('--- yTrain_mean shape: {}'.format(self.yTrain.shape))

    ### Train pure LSTM encoder and save the outputs of LSTM layer
    def train_LSTM_encoder(self):

        print('--- Start training LSTM encoder ---')
        stop_training = False
        early_stop_wait = 0
        stopped_iter = 0
        min_delta = 0

        if self.allTestDataEvaluationDuringTrain:
            self.trainR2_mean = []
            self.validR2_mean = []
            self.testR2_mean = []

        stopped_baseline = None
        if stopped_baseline is not None:
            best_loss = stopped_baseline
        else:
            best_loss = np.Inf
        best_weights = None

        self.trainSteps_lstm.train_loss_mean.reset_states()
        self.trainSteps_lstm.valid_loss_mean.reset_states()
        self.trainSteps_lstm.test_loss_mean.reset_states()

        if self.configs['batch_training'] == True:
            bool_found_NaN = False
            ### Main training loop
            for i in range(self.configs['Max_lstm_iter']):
                if bool_found_NaN:
                    print('--- Stop or go to next sets of tuning parameters due to NaN(s)')
                    break
                self.trainSteps_lstm.train_loss_mean.reset_states() ### mean loss of all steps in one epoch
                self.trainSteps_lstm.valid_loss_mean.reset_states()
                self.trainSteps_lstm.test_loss_mean.reset_states()

                ### Batch training loop
                # for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                for step, (x_batch_train, y_batch_train, w_trainY_batch) in enumerate(self.train_dataset):
                    if self.configs['weighted_MSE']:
                        self.trainSteps_lstm.batch_train_step_mean_LSTM(x_batch_train, y_batch_train, weights=w_trainY_batch)
                    else:
                        self.trainSteps_lstm.batch_train_step_mean_LSTM(x_batch_train, y_batch_train, weights=None)
                    current_train_loss = self.trainSteps_lstm.train_loss_mean.result()

                    if math.isnan(current_train_loss):
                        print('--- WARNING: NaN(s) detected, stop or go to next sets of tuning parameters...')
                        bool_found_NaN = True
                        break
                    if (step % 100 == 0) and self.configs['verbose'] > 1:  # if i % 100 == 0:
                        print('Step: {}, train_mean loss: {}'.format(step, current_train_loss))
                self.train_loss_mean_list.append(current_train_loss.numpy())
                ### lr decay
                if self.configs['exponential_decay']:
                    self.trainSteps_lstm.global_step_lstm_0.assign_add(1)

                #### Validation loop at the end of each epoch
                for x_batch_valid, y_batch_valid in self.valid_dataset:
                    self.trainSteps_lstm.batch_valid_step_mean_LSTM(x_batch_valid, y_batch_valid)
                    current_valid_loss = self.trainSteps_lstm.valid_loss_mean.result()
                self.valid_loss_mean_list.append(current_valid_loss.numpy())

                # print('--- Epoch: {}, Train loss mean: {}, Valid loss mean: {}'.format(i, current_train_loss.numpy(), current_valid_loss.numpy()))
                if self.configs['plot_evolution']:
                    r2_train, r2_valid, r2_test, train_results_np, valid_results_np, test_results_np = self.LSTMEncoderEvaluation(self.scalerx, self.scalery, save_results=False, return_results=True)
                    self.plotter.plotLSTM_encoder(self.train_idx, self.valid_idx, train_results_np, valid_results_np, test_results_np, \
                        figname='LSTM encoder for {}, iter: {}'.format(self.configs['data_name'], i), ylim_1=self.configs['plot_evo_ylims'][0], ylim_2=self.configs['plot_evo_ylims'][1], show_plot=False, \
                        savefig=self.configs['project']+self.configs['exp']+'/'+self.configs['save_encoder_results']+'/'+self.configs['plot_evo_folder']+'/'+'encoder_iter_{}.png'.format(i))
                        # savefig=self.configs['plot_evo_folder'] + 'encoder_iter_{}.png'.format(i))

                    plt.close()

                ### (optional) evaluate all testing data
                if self.allTestDataEvaluationDuringTrain:
                    train_predictions, train_loss = self.trainSteps_lstm.test_step_mean_LSTM(self.xTrain, self.yTrain)
                    valid_predictions, valid_loss = self.trainSteps_lstm.test_step_mean_LSTM(self.xValid, self.yValid)
                    test_predictions, test_loss = self.trainSteps_lstm.test_step_mean_LSTM(self.xTest, self.yTest)

                    train_predictions = self.scalery.inverse_transform(train_predictions)
                    valid_predictions = self.scalery.inverse_transform(valid_predictions)
                    test_predictions = self.scalery.inverse_transform(test_predictions)
                    yTrain = self.scalery.inverse_transform(self.yTrain)
                    yValid = self.scalery.inverse_transform(self.yValid)
                    yTest = self.scalery.inverse_transform(self.yTest)


                    if self.configs['ylog_trans']:
                        train_predictions = np.exp(train_predictions)
                        valid_predictions = np.exp(valid_predictions)
                        test_predictions = np.exp(test_predictions)
                        yTrain = np.exp(yTrain)
                        yValid = np.exp(yValid)
                        yTest = np.exp(yTest)

                    current_train_r2_all = r2_score(yTrain, train_predictions)
                    current_valid_r2_all = r2_score(yValid, valid_predictions)
                    current_test_r2_all = r2_score(yTest, test_predictions)

                    self.trainR2_mean.append(current_train_r2_all)
                    self.validR2_mean.append(current_valid_r2_all)
                    self.testR2_mean.append(current_test_r2_all)

                    print('- Iter: {}, trainLoss: {:.4f}, validLoss: {:.4f}, testLoss: {:.4f}, trainR2: {:.4f}, validR2: {:.4f}, testR2: {:.4f}'\
                        .format(i, train_loss, valid_loss, test_loss, current_train_r2_all, current_valid_r2_all,current_test_r2_all))

                ### (optional) evaluate batch testing data
                if self.testDataEvaluationDuringTrain:
                    for x_batch_test, y_batch_test in self.test_dataset:
                        self.trainSteps_lstm.batch_test_step_mean_LSTM(x_batch_test, y_batch_test)
                        current_test_loss = self.trainSteps_lstm.test_loss_mean.result()
                    self.test_loss_mean_list.append(current_test_loss.numpy())
                    if i % 1 == 0:
                        print('--- Epoch: {}, train loss: {}, validation loss: {}, test loss: {}'.format(i, current_train_loss, current_valid_loss, current_test_loss))
                else:
                    if i % 10 == 0:
                        print('--- Epoch: {}, train loss: {}, validation loss: {}'.format(i, current_train_loss, current_valid_loss))

                if self.configs['early_stop'] and i >= self.configs['early_stop_start_iter']:
                    if np.less(current_valid_loss - min_delta, best_loss):
                        best_loss = current_valid_loss
                        early_stop_wait = 0
                        if self.configs['restore_best_weights']:
                            best_weights = self.trainSteps_lstm.net_lstm_mean.get_weights()
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
                                    self.trainSteps_lstm.net_lstm_mean.set_weights(best_weights)
                            if self.configs['saveWeights']:
                                print('--- Saving best model weights to h5 file: {}_best_mean_iter_{}.h5'.format(self.configs['data_name'], str(i+1)))
                                self.trainSteps_lstm.net_lstm_mean.save_weights(os.getcwd()+
                                    '/Results_PI3NN/checkpoints_meafn/'+self.cofnfigs['data_name']+'_best_mean_iter_' + str(i + 1) + '.h5')
                self.iter_mean_list.append(i)
                if stop_training:
                    if self.testDataEvaluationDuringTrain:
                        print('--- Early stopping criteria met.  Epoch: {}, train_loss:{}, valid_loss:{}, test_loss:{}'.format(i+1, current_train_loss, current_valid_loss, current_test_loss))
                        break
                    if not self.testDataEvaluationDuringTrain:
                        print('--- Early stopping criteria met.  Epoch: {}, train_loss:{}, valid_loss:{}'.format(i+1, current_train_loss, current_valid_loss))
                        break 

    def testDataPrediction(self):
        self.train_output = self.trainSteps_lstm.net_lstm_mean(self.xTrain, training=False)
        self.train_output_up = self.trainSteps_lstm.net_lstm_up(self.xTrain, training=False)
        self.train_output_down = self.trainSteps_lstm.net_lstm_down(self.xTrain, training=False)

        self.valid_output = self.trainSteps_lstm.net_lstm_mean(self.xValid, training=False)
        self.valid_output_up = self.trainSteps_lstm.net_lstm_up(self.xValid, training=False)
        self.valid_output_down = self.trainSteps_lstm.net_lstm_down(self.xValid, training=False) 

        self.test_output = self.trainSteps_lstm.net_lstm_mean(self.xTest, training=False)
        self.test_output_up = self.trainSteps_lstm.net_lstm_up(self.xTest, training=False)
        self.test_output_down = self.trainSteps_lstm.net_lstm_down(self.xTest, training=False)  


    def LSTMEncoderEvaluation(self, scalerx, scalery, save_results=False, return_results=False):
        train_output = self.trainSteps_lstm.net_lstm_mean(self.xTrain, training=False)
        valid_output = self.trainSteps_lstm.net_lstm_mean(self.xValid, training=False)
        test_output = self.trainSteps_lstm.net_lstm_mean(self.xTest, training=False)

        yTrain = scalery.inverse_transform(self.yTrain)
        yValid = scalery.inverse_transform(self.yValid)
        yTest = scalery.inverse_transform(self.yTest)

        yTrain_pred = scalery.inverse_transform(train_output)
        yValid_pred = scalery.inverse_transform(valid_output)
        yTest_pred = scalery.inverse_transform(test_output)

        if self.configs['ylog_trans']:
            train_results_np = np.hstack((np.exp(yTrain), np.exp(yTrain_pred)))
            valid_results_np = np.hstack((np.exp(yValid), np.exp(yValid_pred)))
            test_results_np = np.hstack((np.exp(yTest), np.exp(yTest_pred)))
        else:
            train_results_np = np.hstack((yTrain, yTrain_pred))
            valid_results_np = np.hstack((yValid, yValid_pred))
            test_results_np = np.hstack((yTest, yTest_pred))

        #### r2 calculations
        if self.configs['ylog_trans']:
            r2_train = r2_score(np.exp(yTrain), np.exp(yTrain_pred))
            r2_valid = r2_score(np.exp(yValid), np.exp(yValid_pred))
            r2_test = r2_score(np.exp(yTest), np.exp(yTest_pred))
        else:
            r2_train = r2_score(yTrain, yTrain_pred)
            r2_valid = r2_score(yValid, yValid_pred)
            r2_test = r2_score(yTest, yTest_pred)            

        if return_results:
            return r2_train, r2_valid, r2_test, train_results_np, valid_results_np, test_results_np
        else:   
            return r2_train, r2_valid, r2_test
