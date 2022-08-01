
''' encoder trainer (for CNN) for rock image data'''
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time
import math
import copy

num_threads = 8
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)

from pi3nn.Networks.networks_CNN import CNN_pure_TF2
from pi3nn.Networks.networks_CNN import CL_CNN_train_steps
from pi3nn.Visualizations.visualization import CL_plotter
from pi3nn.Optimizations.boundary_optimizer import CL_boundary_optimizer, CL_prediction_shifter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pi3nn.Utils.Utils import CL_Utils
utils = CL_Utils()

# Force using CPU globally by hiding GPU(s), comment the line of code below to enable GPU
# tf.config.set_visible_devices([], 'GPU')

# tf.keras.backend.set_floatx('float64') ## to avoid TF casting prediction to float32

##### Note: 3D images are large to fit in RAM at once, so we built tf.Dataset in the data loader 
##### transfer tf.Dataset to here instead of vanilla xTrain, yTrain, ...



class CL_trainer_CNN:


    def __init__(self, configs, train_ds, valid_ds, test_ds, net_cnn_mean, net_cnn_up=None, net_cnn_down=None, scalerx=None, scalery=None, testDataEvaluationDuringTrain=False, allTestDataEvaluationDuringTrain=False):
        self.bool_Nan = False
        self.configs = copy.deepcopy(configs)
        self.net_cnn_mean = net_cnn_mean
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds

        if net_cnn_up is not None:
            self.net_cnn_up = net_cnn_up
        else:
            self.net_cnn_up = None
        if net_cnn_down is not None:
            self.net_cnn_down = net_cnn_down
        else:
            self.net_cnn_down = None
        self.plotter = CL_plotter(self.configs)
        self.testDataEvaluationDuringTrain = testDataEvaluationDuringTrain
        self.allTestDataEvaluationDuringTrain = allTestDataEvaluationDuringTrain

        self.trainSteps_cnn = CL_CNN_train_steps(self.net_cnn_mean, net_cnn_up=self.net_cnn_up, net_cnn_down=self.net_cnn_down,
                                                    optimizers_cnn=self.configs['optimizers_cnn'],
                                                    lr_cnn=self.configs['lr_cnn'],
                                                    exponential_decay=self.configs['exponential_decay'],
                                                    decay_steps=self.configs['decay_steps'],
                                                    decay_rate=self.configs['decay_rate'])   
      

    # # def __init__(self, configs, xTrain, yTrain, net_cnn_mean, net_cnn_up=None, net_cnn_down=None, xValid=None, yValid=None, xTest=None, yTest=None, train_idx=None, valid_idx=None,\
    # #     scalerx=None, scalery=None, testDataEvaluationDuringTrain=False, allTestDataEvaluationDuringTrain=False):

    #     self.bool_Nan = False
    #     self.configs = copy.deepcopy(configs)
    #     self.net_lstm_mean = net_lstm_mean
    #     self.net_lstm_up = net_lstm_up
    #     self.net_lstm_down = net_lstm_down

    #     self.plotter = CL_plotter(self.configs)

    #     self.xTrain = xTrain
    #     self.yTrain = yTrain

    #     if xValid is not None:
    #         self.xValid = xValid
    #     if yValid is not None:
    #         self.yValid = yValid

    #     if xTest is not None:
    #         self.xTest = xTest
    #     if yTest is not None:
    #         self.yTest = yTest

    #     if scalerx is not None:
    #         self.scalerx = scalerx
    #     if scalery is not None:
    #         self.scalery = scalery

    #     if train_idx is not None:
    #         self.train_idx = train_idx
    #     if valid_idx is not None:
    #         self.valid_idx = valid_idx

    #     self.testDataEvaluationDuringTrain = testDataEvaluationDuringTrain
    #     self.allTestDataEvaluationDuringTrain = allTestDataEvaluationDuringTrain

    #     self.trainSteps_cnn = CL_CNN_train_steps(self.net_cnn_mean, self.net_cnn_up, self.net_cnn_down,
    #                                                 optimizers_cnn=self.configs['optimizers_cnn'],
    #                                                 lr_cnn=self.configs['lr_cnn'],
    #                                                 exponential_decay=self.configs['exponential_decay'],
    #                                                 decay_steps=self.configs['decay_steps'],
    #                                                 decay_rate=self.configs['decay_rate'])         

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

    #     self.saveFigPrefix = self.configs['data_name']   # prefix for the saved plots

        
    #     print('--- Mean: {}'.format(self.yTrain.mean()))
    #     print('--- Variance: {}'.format(self.yTrain.var()))
    #     print('--- STD: {}'.format(self.yTrain.std()))
    #     print('--- Max: {}'.format(self.yTrain.max()))
    #     print('--- Min: {}'.format(self.yTrain.min()))

    #     ### test simple weighted MSE
    #     self.w_trainY = np.abs(self.yTrain - self.yTrain.mean())

    #     ### convert train/valid/test Numpy data to Tensors and tf.Dataset
    #     if self.configs['batch_training'] == True:
    #         # self.train_dataset = tf.data.Dataset.from_tensor_slices((self.xTrain, self.yTrain))
    #         # self.valid_dataset = tf.data.Dataset.from_tensor_slices((self.xValid, self.yValid))
    #         # self.test_dataset = tf.data.Dataset.from_tensor_slices((self.xTest, self.yTest))
    #         self.train_dataset = tf.data.Dataset.from_tensor_slices((self.xTrain, self.yTrain, self.w_trainY))
    #         self.valid_dataset = tf.data.Dataset.from_tensor_slices((self.xValid, self.yValid))
    #         self.test_dataset = tf.data.Dataset.from_tensor_slices((self.xTest, self.yTest))
    #         if self.configs['batch_shuffle'] == True:
    #             self.train_dataset = self.train_dataset.shuffle(buffer_size=self.configs['batch_shuffle_buffer']).batch(self.configs['batch_size'])
    #             self.valid_dataset = self.valid_dataset.shuffle(buffer_size=self.configs['batch_shuffle_buffer']).batch(self.configs['batch_size'])
    #             self.test_dataset = self.test_dataset.shuffle(buffer_size=self.configs['batch_shuffle_buffer']).batch(self.configs['batch_size'])
    #         else:
    #             self.train_dataset = self.train_dataset.batch(self.configs['batch_size'])
    #             self.valid_dataset = self.valid_dataset.batch(self.configs['batch_size'])
    #             self.test_dataset = self.test_dataset.batch(self.configs['batch_size'])


    #     print('--- xTrain_mean shape: {}'.format(self.xTrain.shape))
    #     print('--- yTrain_mean shape: {}'.format(self.yTrain.shape))

    ### Train pure CNN encoder and save the outputs of CNN layer
    def train_CNN_encoder(self):

        print('--- Start training CNN encoder ---')
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

        self.trainSteps_cnn.train_loss_mean.reset_states()
        self.trainSteps_cnn.valid_loss_mean.reset_states()
        self.trainSteps_cnn.test_loss_mean.reset_states()

        ### txt file for saving the train/valid/test R2
        r2_saving_handler = open(self.configs['project']+self.configs['exp']+'/'+self.configs['save_encoder_results']+'/'+self.configs['data_name']+'_R2_records.txt', 'w', buffering=1)
        r2_valid_test_r2_list = []
        bool_encoder_saved = None
        if self.configs['batch_training'] == True:
            bool_found_NaN = False
            ### Main training loop
            for i in range(self.configs['Max_cnn_iter']):
                print('--- Training steps')
                if bool_found_NaN:
                    print('--- Stop or go to next sets of tuning parameters due to NaN(s)')
                    break
                self.trainSteps_cnn.train_loss_mean.reset_states() ### mean loss of all steps in one epoch
                self.trainSteps_cnn.valid_loss_mean.reset_states()
                self.trainSteps_cnn.test_loss_mean.reset_states()

                iii = 0
                for step, (_, img, label) in enumerate(self.train_ds):
                    # print('--- Fname: {}'.format(fname.numpy()))
                    # print('--- Image shape: {}'.format(img.numpy().shape))
                    # print('--- Label: {}'.format(label.numpy()))
                    label = tf.reshape(label, [-1, 1])
                    self.trainSteps_cnn.batch_train_step_mean_CNN(img, label, weights=None)
                    current_train_loss = self.trainSteps_cnn.train_loss_mean.result()
                    # if (step % 500 == 0) and self.configs['verbose'] > 1:  # if i % 100 == 0:

                    #     ### evaluate R2 for train
                    #     for tmp_step, (_, tmp_img, tmp_label) in enumerate(self.train_ds):
                    #         tmp_label = tf.reshape(tmp_label, [-1, 1])
                    #         tmp_train_output = self.trainSteps_cnn.net_cnn_mean(tmp_img, training=False)
                    #         if tmp_step == 0:
                    #             tmp_train_output_np = tmp_train_output.numpy()
                    #             tmp_yTrain_np = tmp_label.numpy()
                    #         else:
                    #             tmp_train_output_np = np.vstack((tmp_train_output_np, tmp_train_output.numpy()))
                    #             tmp_yTrain_np = np.vstack((tmp_yTrain_np, tmp_label.numpy()))


                    #     #### check the r2 calclation

                    #     # print('----TTTTTTTTTTTTTTTTTTTT')
                    #     # print(type(tmp_yTrain_np))
                    #     # print(type(tmp_train_output_np))
                    #     # print(tmp_yTrain_np.shape)
                    #     # print(tmp_train_output_np.shape)
                    #     # exit()

                    #     tmp_r2_train = r2_score(tmp_yTrain_np, tmp_train_output_np)
                    #     tmp_mse_train = mean_squared_error(tmp_yTrain_np, tmp_train_output_np)

                    #     ### evaluate R2 for valid
                    #     for tmp_step, (_, tmp_img, tmp_label) in enumerate(self.valid_ds):
                    #         tmp_label = tf.reshape(tmp_label, [-1, 1])
                    #         tmp_valid_output = self.trainSteps_cnn.net_cnn_mean(tmp_img, training=False)
                    #         if tmp_step == 0:
                    #             tmp_valid_output_np = tmp_valid_output.numpy()
                    #             tmp_yValid_np = tmp_label.numpy()
                    #         else:
                    #             tmp_valid_output_np = np.vstack((tmp_valid_output_np, tmp_valid_output.numpy()))
                    #             tmp_yValid_np = np.vstack((tmp_yValid_np, tmp_label.numpy()))

                    #     tmp_r2_valid = r2_score(tmp_yValid_np, tmp_valid_output_np)
                    #     tmp_mse_valid = mean_squared_error(tmp_yValid_np, tmp_valid_output_np)
                    #     print('Epoch: {}, Step: {}, train_mean loss: {}, train R2: {}, valid R2: {}, trainMSE: {:.4f}, validMSE: {:.4f}'.format(i+1, step, current_train_loss, tmp_r2_train, tmp_r2_valid, tmp_mse_train, tmp_mse_valid))
                    if (step % 100 == 0) and self.configs['verbose'] > 1:  # if i % 100 == 0:
                        print('Epoch: {}, Step: {}, train_mean loss: {}'.format(i+1, step, current_train_loss))


                self.train_loss_mean_list.append(current_train_loss.numpy())

                # #### Validation loop at the end of each epoch
                # print('--- Validation steps')
                # for step, (_, img, label) in enumerate(self.valid_ds):
                #     label = tf.reshape(label, [-1, 1])
                #     self.trainSteps_cnn.batch_valid_step_mean_CNN(img, label)
                #     current_valid_loss = self.trainSteps_cnn.valid_loss_mean.result()
                #     if (step % 100 == 0) and self.configs['verbose'] > 1:  # if i % 100 == 0:
                #         print('Epoch: {}, Step: {}, valid_mean loss: {}'.format(i+1, step, current_valid_loss))
                # self.valid_loss_mean_list.append(current_valid_loss.numpy())

                r2_train, r2_valid, r2_test = self.CNNEncoderEvaluation()
                r2_valid_test_tmp = r2_valid + r2_test
                if i > 100:
                    # if r2_valid_test_tmp > np.max(r2_valid_test_r2_list) and i>0:  ### save encoder and mark in in the txt file
                    #     bool_encoder_saved = True
                    #     print('--- Better encoder saved at epoch {}'.format(i+1))
                    #     ### save encoder
                    #     tf.saved_model.save(self.net_cnn_mean, self.configs['project']+self.configs['exp']+'/'+self.configs['save_encoder_results']+'/'+self.configs['data_name']+'_mean_model_encoder')
                    # else:
                    #     bool_encoder_saved = False
                    tf.saved_model.save(self.net_cnn_mean, self.configs['project']+self.configs['exp']+'/'+self.configs['save_encoder_results']+'/'+self.configs['data_name']+'_mean_model_encoder'+'_iter_'+str(i+1))


                r2_valid_test_r2_list.append(r2_valid_test_tmp)

                r2_saving_handler.write('{} {:.4f} {:.4f} {:.4f} save encoder: {} \n'.format(i+1, r2_train, r2_valid, r2_test, bool_encoder_saved))
                ### save r2 to files

                ### check valid, test R2, if they are better than previous best, save the encoder.





                #### test batch training loop (batch size should be 1)


                # ### Batch training loop
                # # for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                # for step, (x_batch_train, y_batch_train, w_trainY_batch) in enumerate(self.train_dataset):
                #     if self.configs['weighted_MSE']:
                #         self.trainSteps_cnn.batch_train_step_mean_CNN(x_batch_train, y_batch_train, weights=w_trainY_batch)
                #     else:
                #         self.trainSteps_cnn.batch_train_step_mean_CNN(x_batch_train, y_batch_train, weights=None)
                #     current_train_loss = self.trainSteps_cnn.train_loss_mean.result()

                #     if math.isnan(current_train_loss):
                #         print('--- WARNING: NaN(s) detected, stop or go to next sets of tuning parameters...')
                #         bool_found_NaN = True
                #         break
                #     if (step % 100 == 0) and self.configs['verbose'] > 1:  # if i % 100 == 0:
                #         print('Step: {}, train_mean loss: {}'.format(step, current_train_loss))
                # self.train_loss_mean_list.append(current_train_loss.numpy())
                # ### lr decay
                # if self.configs['exponential_decay']:
                #     self.trainSteps_cnn.global_step_cnn_0.assign_add(1)

                # #### Validation loop at the end of each epoch
                # for x_batch_valid, y_batch_valid in self.valid_dataset:
                #     self.trainSteps_cnn.batch_valid_step_mean_CNN(x_batch_valid, y_batch_valid)
                #     current_valid_loss = self.trainSteps_cnn.valid_loss_mean.result()
                # self.valid_loss_mean_list.append(current_valid_loss.numpy())

                # # print('--- Epoch: {}, Train loss mean: {}, Valid loss mean: {}'.format(i, current_train_loss.numpy(), current_valid_loss.numpy()))
                # if self.configs['plot_evolution']:
                #     r2_train, r2_valid, r2_test, train_results_np, valid_results_np, test_results_np = self.CNNEncoderEvaluation(self.scalerx, self.scalery, save_results=False, return_results=True)
                #     self.plotter.plotCNN_encoder(self.train_idx, self.valid_idx, train_results_np, valid_results_np, test_results_np, \
                #         figname='CNN encoder for {}, iter: {}'.format(self.configs['data_name'], i), ylim_1=self.configs['plot_evo_ylims'][0], ylim_2=self.configs['plot_evo_ylims'][1], show_plot=False, \
                #         savefig=self.configs['project']+self.configs['exp']+'/'+self.configs['save_encoder_results']+'/'+self.configs['plot_evo_folder']+'/'+'encoder_iter_{}.png'.format(i))
                #         # savefig=self.configs['plot_evo_folder'] + 'encoder_iter_{}.png'.format(i))

                #     plt.close()

                # ### (optional) evaluate all testing data
                # if self.allTestDataEvaluationDuringTrain:
                #     train_predictions, train_loss = self.trainSteps_cnn.test_step_mean_CNN(self.xTrain, self.yTrain)
                #     valid_predictions, valid_loss = self.trainSteps_cnn.test_step_mean_CNN(self.xValid, self.yValid)
                #     test_predictions, test_loss = self.trainSteps_cnn.test_step_mean_CNN(self.xTest, self.yTest)

                #     train_predictions = self.scalery.inverse_transform(train_predictions)
                #     valid_predictions = self.scalery.inverse_transform(valid_predictions)
                #     test_predictions = self.scalery.inverse_transform(test_predictions)
                #     yTrain = self.scalery.inverse_transform(self.yTrain)
                #     yValid = self.scalery.inverse_transform(self.yValid)
                #     yTest = self.scalery.inverse_transform(self.yTest)


                #     if self.configs['ylog_trans']:
                #         train_predictions = np.exp(train_predictions)
                #         valid_predictions = np.exp(valid_predictions)
                #         test_predictions = np.exp(test_predictions)
                #         yTrain = np.exp(yTrain)
                #         yValid = np.exp(yValid)
                #         yTest = np.exp(yTest)

                #     current_train_r2_all = r2_score(yTrain, train_predictions)
                #     current_valid_r2_all = r2_score(yValid, valid_predictions)
                #     current_test_r2_all = r2_score(yTest, test_predictions)

                #     self.trainR2_mean.append(current_train_r2_all)
                #     self.validR2_mean.append(current_valid_r2_all)
                #     self.testR2_mean.append(current_test_r2_all)

                #     print('- Iter: {}, trainLoss: {:.4f}, validLoss: {:.4f}, testLoss: {:.4f}, trainR2: {:.4f}, validR2: {:.4f}, testR2: {:.4f}'\
                #         .format(i, train_loss, valid_loss, test_loss, current_train_r2_all, current_valid_r2_all,current_test_r2_all))

                # ### (optional) evaluate batch testing data
                # if self.testDataEvaluationDuringTrain:
                #     for x_batch_test, y_batch_test in self.test_dataset:
                #         self.trainSteps_cnn.batch_test_step_mean_CNN(x_batch_test, y_batch_test)
                #         current_test_loss = self.trainSteps_cnn.test_loss_mean.result()
                #     self.test_loss_mean_list.append(current_test_loss.numpy())
                #     if i % 1 == 0:
                #         print('--- Epoch: {}, train loss: {}, validation loss: {}, test loss: {}'.format(i, current_train_loss, current_valid_loss, current_test_loss))
                # else:
                #     if i % 10 == 0:
                #         print('--- Epoch: {}, train loss: {}, validation loss: {}'.format(i, current_train_loss, current_valid_loss))

                # if self.configs['early_stop'] and i >= self.configs['early_stop_start_iter']:
                #     if np.less(current_valid_loss - min_delta, best_loss):
                #         best_loss = current_valid_loss
                #         early_stop_wait = 0
                #         if self.configs['restore_best_weights']:
                #             best_weights = self.trainSteps_cnn.net_cnn_mean.get_weights()
                #     else:
                #         early_stop_wait += 1
                #         # print('--- Iter: {}, early_stop_wait: {}'.format(i+1, early_stop_wait))
                #         if early_stop_wait >= self.configs['wait_patience']:
                #             stopped_iter = i
                #             stop_training = True
                #             if self.configs['restore_best_weights']:
                #                 if best_weights is not None:
                #                     if self.configs['verbose'] > 0:
                #                         print('--- Restoring mean model weights from the end of the best iteration')
                #                     self.trainSteps_cnn.net_cnn_mean.set_weights(best_weights)
                #             if self.configs['saveWeights']:
                #                 print('--- Saving best model weights to h5 file: {}_best_mean_iter_{}.h5'.format(self.configs['data_name'], str(i+1)))
                #                 self.trainSteps_cnn.net_cnn_mean.save_weights(os.getcwd()+
                #                     '/Results_PI3NN/checkpoints_meafn/'+self.cofnfigs['data_name']+'_best_mean_iter_' + str(i + 1) + '.h5')
                # self.iter_mean_list.append(i)
                # if stop_training:
                #     if self.testDataEvaluationDuringTrain:
                #         print('--- Early stopping criteria met.  Epoch: {}, train_loss:{}, valid_loss:{}, test_loss:{}'.format(i+1, current_train_loss, current_valid_loss, current_test_loss))
                #         break
                #     if not self.testDataEvaluationDuringTrain:
                #         print('--- Early stopping criteria met.  Epoch: {}, train_loss:{}, valid_loss:{}'.format(i+1, current_train_loss, current_valid_loss))
                #         break 

    def testDataPrediction(self):
        self.train_output = self.trainSteps_cnn.net_cnn_mean(self.xTrain, training=False)
        self.train_output_up = self.trainSteps_cnn.net_cnn_up(self.xTrain, training=False)
        self.train_output_down = self.trainSteps_cnn.net_cnn_down(self.xTrain, training=False)

        self.valid_output = self.trainSteps_cnn.net_cnn_mean(self.xValid, training=False)
        self.valid_output_up = self.trainSteps_cnn.net_cnn_up(self.xValid, training=False)
        self.valid_output_down = self.trainSteps_cnn.net_cnn_down(self.xValid, training=False) 

        self.test_output = self.trainSteps_cnn.net_cnn_mean(self.xTest, training=False)
        self.test_output_up = self.trainSteps_cnn.net_cnn_up(self.xTest, training=False)
        self.test_output_down = self.trainSteps_cnn.net_cnn_down(self.xTest, training=False)  

    # def CNNEncoderEvaluation(self, save_results=False, return_results=False):
    #     train_output = self.trainSteps_cnn.net_cnn_mean(self.xTrain, training=False)
    #     valid_output = self.trainSteps_cnn.net_cnn_mean(self.xValid, training=False)
    #     test_output = self.trainSteps_cnn.net_cnn_mean(self.xTest, training=False)


    import matplotlib.pyplot as plt
    def CNNEncoderEvaluation(self, scalery=None, save_results=False, return_results=False):
        print('--- Start CNN Encoder Evaluations...')

        # print('-------print  net_CNN_mean instance again at encoder evaluation step')
        # print(self.trainSteps_cnn.net_cnn_mean)
        # exit()
        ## Evaluate training data
        for step, (_, img, label) in enumerate(self.train_ds):
            label = tf.reshape(label, [-1, 1])
            train_output = self.trainSteps_cnn.net_cnn_mean(img, training=False)
            if (step % 100 == 0) and self.configs['verbose'] > 1:  # if i % 100 == 0:
                print('Evaluating training data, Step: {}'.format(step))
            if step == 0:
                train_output_np = train_output.numpy()
                yTrain_np = label.numpy()
            else:
                train_output_np = np.vstack((train_output_np, train_output.numpy()))
                yTrain_np = np.vstack((yTrain_np, label.numpy()))

            # print(img.numpy().shape)
            # # print(plt.imshow(img.numpy()[0, :, :, 64, 0]))
            # # plt.show()
            # print(step)
            # print(np.sum(img.numpy()))
            # print(train_output.numpy())
            # print(label.numpy())

        ## Evaluate valid data
        for step, (_, img, label) in enumerate(self.valid_ds):
            label = tf.reshape(label, [-1, 1])
            valid_output = self.trainSteps_cnn.net_cnn_mean(img, training=False)
            if (step % 100 == 0) and self.configs['verbose'] > 1:  # if i % 100 == 0:
                print('Evaluating validation data, Step: {}'.format(step))
            if step == 0:
                valid_output_np = valid_output.numpy()
                yValid_np = label.numpy()
            else:
                valid_output_np = np.vstack((valid_output_np, valid_output.numpy()))
                yValid_np = np.vstack((yValid_np, label.numpy()))  

        ## Evaluate testing data      
        for step, (_, img, label) in enumerate(self.test_ds):
            label = tf.reshape(label, [-1, 1])
            test_output = self.trainSteps_cnn.net_cnn_mean(img, training=False)
            if (step % 100 == 0) and self.configs['verbose'] > 1:  # if i % 100 == 0:
                print('Evaluating test data, Step: {}'.format(step))
            if step == 0:
                test_output_np = test_output.numpy()
                yTest_np = label.numpy()
            else:
                test_output_np = np.vstack((test_output_np, test_output.numpy()))
                yTest_np = np.vstack((yTest_np, label.numpy()))  

        #### above code running OK
        yTrain = yTrain_np
        yValid = yValid_np
        yTest = yTest_np

        if scalery is not None:
            yTrain = scalery.inverse_transform(yTrain)
            yValid = scalery.inverse_transform(yValid)
            yTest = scalery.inverse_transform(yTest)

            yTrain_pred = scalery.inverse_transform(train_output_np)
            yValid_pred = scalery.inverse_transform(valid_output_np)
            yTest_pred = scalery.inverse_transform(test_output_np)
        else:
            yTrain_pred = train_output_np
            yValid_pred = valid_output_np
            yTest_pred = test_output_np

        if self.configs['ylog_trans']:
            train_results_np = np.hstack((np.exp(yTrain), np.exp(yTrain_pred)))
            valid_results_np = np.hstack((np.exp(yValid), np.exp(yValid_pred)))
            test_results_np = np.hstack((np.exp(yTest), np.exp(yTest_pred)))
        else:
            train_results_np = np.hstack((yTrain, yTrain_pred))
            valid_results_np = np.hstack((yValid, yValid_pred))
            test_results_np = np.hstack((yTest, yTest_pred))

        # print('EEEEEEEEEEEEEEEVVVVVVVVVVVVVVVVVVVV')
        # print(yTrain)
        # print('WWWWWWWWWWWWWW')
        # print(yTrain_pred)

        # print('VVVVVVVVVVVVVVVVVVVVVVVVVV')
        # print(yValid)
        # print('WWWWWW22222222222222222222')
        # print(yValid_pred)

        # print('VVVVVVVVVVVVVVVVVVVV33333333333333333333')
        # print(yTest)
        # print('WWWWWWWWWW3333333333333333333333')
        # print(yTest_pred)
        # # exit()

        # print('yTrain shape: {}'.format(yTrain.shape))
        # print('yTrain_pred shape: {}'.format(yTrain_pred.shape))
        # print('yValid shape: {}'.format(yValid.shape))
        # print('yValid_pred shape: {}'.format(yValid_pred.shape))
        # print('yTest shape: {}'.format(yTest.shape))
        # print('yTest_pred shape: {}'.format(yTest_pred.shape))


        #### r2 calculations
        if self.configs['ylog_trans']:
            r2_train = r2_score(np.exp(yTrain), np.exp(yTrain_pred))
            r2_valid = r2_score(np.exp(yValid), np.exp(yValid_pred))
            r2_test = r2_score(np.exp(yTest), np.exp(yTest_pred))
        else:
            r2_train = r2_score(yTrain, yTrain_pred)
            r2_valid = r2_score(yValid, yValid_pred)
            r2_test = r2_score(yTest, yTest_pred) 

        print('train R2: {}'.format(r2_train))
        print('valid R2: {}'.format(r2_valid))    
        print('test R2: {}'.format(r2_test))               

        if return_results:
            return r2_train, r2_valid, r2_test, train_results_np, valid_results_np, test_results_np
        else:   
            return r2_train, r2_valid, r2_test