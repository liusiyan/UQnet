import os
import tensorflow as tf
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
num_threads = 8
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)
from tqdm import tqdm

# Force using CPU globally by hiding GPU(s), comment the line of code below to enable GPU
tf.config.set_visible_devices([], 'GPU')

from pathlib import Path
import datetime
from tqdm import tqdm
import time
import json
import pickle
# import keras

import argparse
import json
import itertools
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

from pi3nn.DataLoaders.data_loaders import CL_dataLoader
from pi3nn.Networks.networks import UQ_Net_mean_TF2, UQ_Net_std_TF2
from pi3nn.Networks.networks import CL_UQ_Net_train_steps
from pi3nn.Networks.networks_TS import LSTM_mean_TF2, LSTM_PI_TF2, LSTM_pure_TF2, CL_LSTM_train_steps
from pi3nn.Networks.networks_CNN import CNN_pure_TF2, CL_CNN_train_steps
from pi3nn.Trainers.trainers import CL_trainer
from pi3nn.Trainers.trainer_TS import CL_trainer_TS
from pi3nn.Trainers.trainer_CNN import CL_trainer_CNN
from pi3nn.Optimizations.boundary_optimizer import CL_boundary_optimizer
from pi3nn.Visualizations.visualization import CL_plotter
from pi3nn.Optimizations.params_optimizer import CL_params_optimizer, CL_params_optimizer_LSTM
from pi3nn.Utils.Utils import CL_Utils

# from hyperopt import fmin, hp, Trials, STATUS_OK, tpe, rand
# tf.keras.backend.set_floatx('float64') ## to avoid TF casting prediction to float32

utils = CL_Utils()
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='boston', help='example data names: boston, concrete, energy, kin8nm, wine, yacht')
parser.add_argument('--mode', type=str, default='auto', help='auto or manual mode')
parser.add_argument('--quantile', type=float, default=0.90)
parser.add_argument('--streamflow_inputs', type=int, default=3)
parser.add_argument('--plot_evo', type=bool, default=False, help='plot prediction for each iteration for analysis')

parser.add_argument('--project', type=utils.dir_path_proj, default=None, help='Please specify the project path')
parser.add_argument('--exp', type=utils.dir_path_exp, default=None, help='Please specify the experiment name')
parser.add_argument('--exp_overwrite', type=bool, default=False, help='Overwrite existing experiment or not')
parser.add_argument('--configs', type=utils.dir_path_configs, help='Please specify the configs file')

''' 
If you would like to customize the data loading and pre-processing, we recommend you to write the 
functions in pi3nn/DataLoaders/data_loaders.py and call it from here. Or write them directly here.
'''
args = parser.parse_args()

if args.project is None:
    print('Please specify project path as argument, example: --project /myproject')
    exit()
if args.exp is None:
    print('Please specify experiment name/folder_name as argument: example, --exp my_exp_1')
    exit()

print('-----------------------------------------------------------------')
print('--- Project:                {}'.format(args.project))
print('--- Experiment:             {}'.format(args.exp))
print('--- Mode:                   {}'.format(args.mode))
print('--- Configs file:           {}'.format(args.configs))
print('--- Overwrite existing exp: {}'.format(args.exp_overwrite))
print('-----------------------------------------------------------------')

configs = {}
configs['data_name'] = args.data

if args.mode == 'cnn_encoder':
    print('-------------------------------------------------------')
    print('------------- Training for CNN encoder ----------------')
    print('-------------------------------------------------------')  

    with open(args.configs, 'r') as json_file_r:
        loaded_json = json.load(json_file_r)

    configs = loaded_json
    configs['project'] = args.project
    configs['exp'] = args.exp

    utils.check_encoder_folder(args.project+args.exp+'/'+configs['save_encoder_folder']) ## create a folder for encoder results, if not exist
    print('--- Dataset: {}'.format(configs['data_name']))
    random.seed(configs['seed'])
    np.random.seed(configs['seed'])
    tf.random.set_seed(configs['seed'])
    data_dir = configs['data_dir'] # './datasets/Rock_Imgs/'

    print(data_dir)

    dataLoader = CL_dataLoader(original_data_path=data_dir, configs=configs)
    ### load images
    ### test using tf.py_function wrapper
    train_ds, valid_ds, test_ds = dataLoader.load_rock_imgs_C() ### Load Berea_1 data
    print('--- Images and perm data loaded !!!')
    print('--- Preparing CNN trainer...')
    num_outputs = 1

    if configs['load_encoder']:
        ## test model loading 
        # loaded_model = tf.saved_model.load(os.getcwd()+'/Results_PI3NN/checkpoints_mean/'+configs['data_name']+'_mean_model')
        loaded_model = tf.saved_model.load(args.project+args.exp+'/'+configs['save_encoder_folder']+'/'+configs['load_encoder_name']+'_mean_model_encoder')
        print('--- Encoder model: {} loaded for intermeadiate prediction of data: {} !!!!'.format(configs['load_encoder_name'], configs['data_name']))
        ### evaluate train, valid and test dataset using the loaded model
        # loaded_model.trainable = False
        ### test if we are still train the model

        ### CNN layer predictions example (first train/valid/test image), and save as compressed .npz
        savez_dict = dict()
        for step, (fname, img, label) in enumerate(test_ds):
            print(step)
            # print(label.numpy())
            # print(type(label.numpy()))
            ### CNN layers ONLY predictions
            label = tf.reshape(label, [-1, 1])
            xx = loaded_model.conv_1(img)
            xx = loaded_model.maxpool_1(xx)
            xx = loaded_model.flatten(xx)
            xx = loaded_model.fc_1(xx)
            # ### add Y labels to the matrix
            tmp_arr = xx.numpy()
            tmp_arr = np.concatenate((tmp_arr, label.numpy().reshape(-1,1)), axis=1)
            savez_dict[fname.numpy()[0].decode("utf-8")] = tmp_arr

        np.savez_compressed(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_'+configs['load_encoder_name']+'_test_ds.npz', **savez_dict)  

        exit()




        # ### test loading npz_compressed
        # loaded_npz = np.load(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_TTTEST_X.npz')
        # ### list all keys
        # keys_extract = list(loaded_npz.keys())

        # ### need to find the corresponding Y labels

        # ### extract and vstack all CNN encoder predictions
        # cnn_layers_pred_list = []
        # for ii, key in enumerate(keys_extract):
        #     print(ii)
        #     cnn_layers_pred_list.append(loaded_npz[key])

        # cnn_layers_pred_np = np.vstack(cnn_layers_pred_list)
        # print(cnn_layers_pred_np)

        # pi3nn_cnn_X = cnn_layers_pred_np[:, :-1]
        # pi3nn_cnn_Y = cnn_layers_pred_np[:, -1]

        ### we can do normal PI3NN training and testing here








            # print(step)
            # print(type(fname.numpy()[0]))
            # print(fname.numpy()[0])
            # print(type(fname.numpy()[0].decode("utf-8")))
            # print(fname.numpy()[0].decode("utf-8"))
        #     if step == 0:
        #         print(np.sum(img.numpy()))
        #         xx = loaded_model.conv_1(img)
        #         xx = loaded_model.maxpool_1(xx)
        #         xx = loaded_model.flatten(xx)
        #         print('SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSs')
        #         print(xx)
        #         print(xx.numpy())
        #         print(np.sum(xx.numpy()))
        #         print(label.numpy())
        #         print(xx.numpy().shape)
        #         print(label.numpy().shape)

        #         np.savez_compressed(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_test_X.npz', xx.numpy())
        #         np.savez_compressed(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_test_Y.npz', label.numpy())

        #         # np.save(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_test_X.npy', xx.numpy())
        #         # np.save(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_test_Y.npy', label.numpy())
        #         # np.savetxt(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_test_X.txt', xx.numpy())
        #         # np.savetxt(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_test_Y.txt', label.numpy())
        #     else:
        #         break

        # ### load the saved CNN layer preditions

        # ### Test loading the CNN encoder predicted data
        # # test_X = np.loadtxt(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_encoder_test_X.txt')
        # # test_Y = np.loadtxt(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_encoder_test_Y.txt')

        # test_X = np.load(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_encoder_test_X.npy')
        # test_Y = np.load(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_encoder_test_Y.npy')

        # print('--- Finished CNN prediction loading...')
        # print(test_X.shape)
        # # print(test_Y.shape)
        # print(np.sum(test_X))
        # print(test_Y)





        # trainValid_X = np.loadtxt(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_encoder_trainValid_X.txt')
        # test_X = np.loadtxt(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_encoder_test_X.txt')
        # trainValid_Y = np.loadtxt(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_encoder_trainValid_Y.txt')
        # test_Y = np.loadtxt(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_encoder_test_Y.txt')
        # trainValid_idx = np.loadtxt(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_trainValid_idx.txt').astype(int)  ### original



        # args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'
        # np.savetxt(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_trainValid_X.txt', np.vstack((pred_train_LSTM, pred_valid_LSTM)))
        # np.savetxt(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_test_X.txt', pred_test_LSTM)
        # np.savetxt(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_trainValid_Y.txt', np.vstack((ori_yTrain, ori_yValid)))
        # np.savetxt(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_test_Y.txt', ori_yTest)
        # np.savetxt(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_trainValid_idx.txt', np.append(xTrain_idx.values, xValid_idx.values).astype(int), fmt='%s')





        # pred_train_LSTM = loaded_model.lstm_mean(xTrain)
        # pred_valid_LSTM = loaded_model.lstm_mean(xValid)
        # pred_test_LSTM = loaded_model.lstm_mean(xTest)  


        # print(list(loaded_model.signatures.keys()))  # ['serving_default']
        # infer = loaded_model.signatures['serving_default']
        # print(infer.structured_outputs)
        # loading_pred = infer


        # for step, (_, img, label) in enumerate(train_ds):
        #     label = tf.reshape(label, [-1, 1])
        #     if step == 0:
        #         train_output = loaded_model(img)  # infer
        #         print(train_output)
        #         train_output = loaded_model(img)  # infer
        #         print(train_output)

        #         break


        # exit()



        # # print(type(loaded_model.__dict__))
        # # print(loaded_model.__dict__.keys())
        # # print(type(loaded_model.conv_1))
        # # print(loaded_model.conv_1)


        # ##### give any SINGLE image from train or test data to predict intermediate parameters


        # for name, img, label in train_ds.take(1):
        #     print(name)
        #     print(np.sum(img.numpy()))
        #     print(label)

        # for name, img, label in train_ds.take(2):
        #     print(name)
        #     print(np.sum(img.numpy()))
        #     print(label)

        # exit()



        # import time

        # t0 = time.time()
        # for step, (_, img, label) in enumerate(train_ds):
        #     label = tf.reshape(label, [-1, 1])
        #     if step == 0:
        #         print(np.sum(img.numpy()))
        #         train_output = loaded_model(img)
        #         print(train_output)
        #     else:
        #         break

        # for step, (_, img, label) in enumerate(train_ds):
        #     label = tf.reshape(label, [-1, 1])
        #     if step == 0:
        #         print(np.sum(img.numpy()))
        #         xx = loaded_model.conv_1(img)
        #         xx = loaded_model.maxpool_1(xx)
        #         xx = loaded_model.flatten(xx)
        #         # xx = loaded_model.fc_2(xx)
        #         # xx = loaded_model.outputLayer(xx)
        #         print(xx)
        #     else:
        #         break

        # print('t: {:.4f}'.format(time.time()-t0))

        exit()



        for step, (_, img, label) in enumerate(train_ds):
            label = tf.reshape(label, [-1, 1])
            if step == 0:
                print(np.sum(img.numpy()))
                train_output = loaded_model(img)
                print(train_output)
            else:
                pass

        # for step, (_, img, label) in enumerate(train_ds):
        #     label = tf.reshape(label, [-1, 1])
        #     if step == 0:
        #         print(np.sum(img.numpy()))
        #         train_output = loaded_model(img)
        #         print(train_output)
        #     else:
        #         pass
            


        # # exit()

        # for step, (_, img, label) in enumerate(train_ds):
        #     label = tf.reshape(label, [-1, 1])
        #     # print(type(img))
        #     # print(img.numpy())
        #     # print(img.numpy().shape)
        #     # # img = img[]
        #     # # exit()
        #     train_output = loaded_model(img)
        #     if (step % 100 == 0) and configs['verbose'] > 1:  # if i % 100 == 0:
        #         print('Evaluating training data, Step: {}'.format(step))
        #     if step == 0:
        #         train_output_np = train_output.numpy()
        #         yTrain_np = label.numpy()
        #     else:
        #         train_output_np = np.vstack((train_output_np, train_output.numpy()))
        #         yTrain_np = np.vstack((yTrain_np, label.numpy()))
        #     if (step % 100 == 0) and configs['verbose'] > 1: 
        #         r2_train = r2_score(yTrain_np, train_output_np)
        #         mse_train = mean_squared_error(yTrain_np, train_output_np)

        #         print('--- Step: {}, trainR2: {:.4f}, trainMSE: {:.4f}'.format(step, r2_train, mse_train))

        # print('--- second round of training started .....')

        # for step, (_, img, label) in enumerate(train_ds):
        #     label = tf.reshape(label, [-1, 1])
        #     # print(type(img))
        #     # print(img.numpy())
        #     # print(img.numpy().shape)
        #     # # img = img[]
        #     # # exit()
        #     train_output = loaded_model(img)
        #     if (step % 100 == 0) and configs['verbose'] > 1:  # if i % 100 == 0:
        #         print('Evaluating training data, Step: {}'.format(step))
        #     if step == 0:
        #         train_output_np = train_output.numpy()
        #         yTrain_np = label.numpy()
        #     else:
        #         train_output_np = np.vstack((train_output_np, train_output.numpy()))
        #         yTrain_np = np.vstack((yTrain_np, label.numpy()))
        #     if (step % 100 == 0) and configs['verbose'] > 1: 
        #         r2_train = r2_score(yTrain_np, train_output_np)
        #         mse_train = mean_squared_error(yTrain_np, train_output_np)

        #         print('--- Step: {}, trainR2: {:.4f}, trainMSE: {:.4f}'.format(step, r2_train, mse_train))


        # exit()












            # if (step % 100 == 0) and configs['verbose'] > 1:  # if i % 100 == 0:
            #     print('Evaluating training data, Step: {}'.format(step))
            # if step == 0:
            #     train_output_np = train_output.numpy()
            #     yTrain_np = label.numpy()
            # else:
            #     train_output_np = np.vstack((train_output_np, train_output.numpy()))
            #     yTrain_np = np.vstack((yTrain_np, label.numpy()))


            # if (step % 100 == 0) and configs['verbose'] > 1: 
            #     r2_train = r2_score(yTrain_np, train_output_np)
            #     mse_train = mean_squared_error(yTrain_np, train_output_np)

            #     print('--- Step: {}, trainR2: {:.4f}, trainMSE: {:.4f}'.format(step, r2_train, mse_train))


        # train_results_np = np.hstack((yTrain, yTrain_pred))
        # valid_results_np = np.hstack((yValid, yValid_pred))
        # test_results_np = np.hstack((yTest, yTest_pred))

        # train_output = loaded_model(xTrain, training=False)
        # valid_output = loaded_model(xValid, training=False)
        # test_output = loaded_model(xTest, training=False)

        # yTrain = scalery.inverse_transform(yTrain)
        # yValid = scalery.inverse_transform(yValid)
        # yTest = scalery.inverse_transform(yTest)

        # yTrain_pred = scalery.inverse_transform(train_output)
        # yValid_pred = scalery.inverse_transform(valid_output)
        # yTest_pred = scalery.inverse_transform(test_output)

        # if configs['ylog_trans']:
        #     yTrain, yTrain_pred = np.exp(yTrain), np.exp(yTrain_pred)
        #     yValid, yValid_pred = np.exp(yValid), np.exp(yValid_pred)
        #     yTest, yTest_pred = np.exp(yTest), np.exp(yTest_pred)

        # train_results_np = np.hstack((yTrain, yTrain_pred))
        # valid_results_np = np.hstack((yValid, yValid_pred))
        # test_results_np = np.hstack((yTest, yTest_pred))

    else:
        net_CNN_mean = CNN_pure_TF2(configs, num_outputs)
        print('net_CNN_mean 1:')
        print(net_CNN_mean)
        ### trainer instance
        trainer_CNN = CL_trainer_CNN(configs, train_ds, valid_ds, test_ds, net_CNN_mean, net_cnn_up=None, net_cnn_down=None)
        ### train CNN
        trainer_CNN.train_CNN_encoder()

        print('DDDDDDDDDDDDONE!!!!')

        if configs['save_encoder']:
            print('--- Saving CNN model to {}_mean_model'.format(configs['data_name']))

            # trainer_CNN.net_cnn_mean.save(args.project+args.exp+'/'+configs['save_encoder_folder']+'/'+configs['data_name']+'_mean_model_encoder', save_format="tf")
            tf.saved_model.save(trainer_CNN.net_cnn_mean, args.project+args.exp+'/'+configs['save_encoder_folder']+'/'+configs['data_name']+'_mean_model_encoder')

        r2_train, r2_valid, r2_test, train_results_np, valid_results_np, test_results_np = trainer_CNN.CNNEncoderEvaluation(scalery=None, save_results=False, return_results=True)






        # net_lstm_mean = LSTM_pure_TF2(configs, num_outputs)
        # trainer_lstm = CL_trainer_TS(configs, xTrain, yTrain, net_lstm_mean, net_lstm_up=None, net_lstm_down=None, xValid=xValid, yValid=yValid, xTest=xTest, yTest=yTest, train_idx=xTrain_idx, valid_idx=xValid_idx,\
        #     scalerx=scalerx, scalery=scalery, testDataEvaluationDuringTrain=False, allTestDataEvaluationDuringTrain=True)
        # trainer_lstm.train_LSTM_encoder()

        # if configs['save_encoder']:
        #     print('--- Saving LSTM model to {}_mean_model'.format(configs['data_name']))
        #     tf.saved_model.save(trainer_lstm.net_lstm_mean, args.project+args.exp+'/'+configs['save_encoder_folder']+'/'+configs['data_name']+'_mean_model_encoder' )  
        # r2_train, r2_valid, r2_test, train_results_np, valid_results_np, test_results_np = trainer_lstm.LSTMEncoderEvaluation(scalerx, scalery, save_results=False, return_results=True)

        # ### LSTM layer predictions
        # pred_train_LSTM = trainer_lstm.net_lstm_mean.lstm_mean(xTrain)
        # pred_valid_LSTM = trainer_lstm.net_lstm_mean.lstm_mean(xValid)
        # pred_test_LSTM = trainer_lstm.net_lstm_mean.lstm_mean(xTest)






        exit()
        dataLoader.load_rock_imgs_B()
        exit()
        ## Trial (1) direct data loading
        dataLoader.load_rock_imgs()
        print('end of test !!!!!!!!!!!!!')
        #### test tf.keras.utils.image_dataset_from_directory
        exit()



    ### save encoder results and the encoder prediction (CNN layer ONLY predictions)
    if configs['save_encoder_results'] is not None:
        ### combine train/valid/test data
        print('--- Saving the encoder results...')
        # trainValid_np = np.zeros((len(train_results_np)+len(valid_results_np), 2))
        # for i in range(2):
        #     trainValid_np[xTrain_idx, i] = train_results_np[:, i]
        #     trainValid_np[xValid_idx, i] = valid_results_np[:, i]

        # np.save(args.project+args.exp+'/'+configs['save_encoder_results']+'/'+'trainValid_np.npy', trainValid_np)
        # np.save(args.project+args.exp+'/'+configs['save_encoder_results']+'/'+'test_np.npy', test_results_np)
        # print('--- Encoder results saved to:')
        # print(args.project+args.exp+'/'+configs['save_encoder_results']+'/')


    if configs['save_encoder_pred']:
        print('--- Saving the encoder predictions...')

        # args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'
        # np.savetxt(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_trainValid_X.txt', np.vstack((pred_train_LSTM, pred_valid_LSTM)))
        # np.savetxt(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_test_X.txt', pred_test_LSTM)
        # np.savetxt(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_trainValid_Y.txt', np.vstack((ori_yTrain, ori_yValid)))
        # np.savetxt(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_test_Y.txt', ori_yTest)
        # np.savetxt(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_trainValid_idx.txt', np.append(xTrain_idx.values, xValid_idx.values).astype(int), fmt='%s')



if args.mode == 'lstm_encoder':
    print('-------------------------------------------------------')
    print('------------- Training for LSTM encoder ---------------')
    print('-------------------------------------------------------')

    with open(args.configs, 'r') as json_file_r:
        loaded_json = json.load(json_file_r)

    configs = loaded_json
    configs['project'] = args.project
    configs['exp'] = args.exp

    utils.check_encoder_folder(args.project+args.exp+'/'+configs['save_encoder_folder']) ## create a folder for encoder results, if not exist

    print('--- Dataset: {}'.format(configs['data_name']))
    random.seed(configs['seed'])
    np.random.seed(configs['seed'])
    tf.random.set_seed(configs['seed'])

    streamflow_data = ['Bradley','Copper','Gothic','Qui','Rock','RUS','EAQ','ph']
    if args.data in streamflow_data:
        data_dir = configs['data_dir'] # './datasets/Timeseries/StreamflowData/'
        dataLoader = CL_dataLoader(original_data_path=data_dir)
        xTrain, xValid, xTest, yTrain, yValid, yTest, num_inputs, num_outputs, scalerx, scalery, ori_yTrain, ori_yValid, ori_yTest, xTrain_idx, xValid_idx = \
        dataLoader.load_streamflow_timeseries(configs, return_original_ydata=True)

    num_inputs = utils.getNumInputsOutputs(xTrain)
    num_outputs = utils.getNumInputsOutputs(yTrain)

    print('--- Num inputs: {}'.format(num_inputs))
    print('--- num_outputs: {}'.format(num_outputs))

    if configs['load_encoder']:
        ## test model loading 
        # loaded_model = tf.saved_model.load(os.getcwd()+'/Results_PI3NN/checkpoints_mean/'+configs['data_name']+'_mean_model')
        loaded_model = tf.saved_model.load(args.project+args.exp+'/'+configs['save_encoder_folder']+'/'+configs['data_name']+'_mean_model_encoder')
        print('--- Encoder model loaded for {} !!!!'.format(configs['data_name']))
        train_output = loaded_model(xTrain, training=False)
        valid_output = loaded_model(xValid, training=False)
        test_output = loaded_model(xTest, training=False)

        yTrain = scalery.inverse_transform(yTrain)
        yValid = scalery.inverse_transform(yValid)
        yTest = scalery.inverse_transform(yTest)

        yTrain_pred = scalery.inverse_transform(train_output)
        yValid_pred = scalery.inverse_transform(valid_output)
        yTest_pred = scalery.inverse_transform(test_output)

        if configs['ylog_trans']:
            yTrain, yTrain_pred = np.exp(yTrain), np.exp(yTrain_pred)
            yValid, yValid_pred = np.exp(yValid), np.exp(yValid_pred)
            yTest, yTest_pred = np.exp(yTest), np.exp(yTest_pred)

        train_results_np = np.hstack((yTrain, yTrain_pred))
        valid_results_np = np.hstack((yValid, yValid_pred))
        test_results_np = np.hstack((yTest, yTest_pred))

        ### LSTM layer predictions
        pred_train_LSTM = loaded_model.lstm_mean(xTrain)
        pred_valid_LSTM = loaded_model.lstm_mean(xValid)
        pred_test_LSTM = loaded_model.lstm_mean(xTest)       

    else:
        net_lstm_mean = LSTM_pure_TF2(configs, num_outputs)
        trainer_lstm = CL_trainer_TS(configs, xTrain, yTrain, net_lstm_mean, net_lstm_up=None, net_lstm_down=None, xValid=xValid, yValid=yValid, xTest=xTest, yTest=yTest, train_idx=xTrain_idx, valid_idx=xValid_idx,\
            scalerx=scalerx, scalery=scalery, testDataEvaluationDuringTrain=False, allTestDataEvaluationDuringTrain=True)
        trainer_lstm.train_LSTM_encoder()

        if configs['save_encoder']:
            print('--- Saving LSTM model to {}_mean_model'.format(configs['data_name']))
            tf.saved_model.save(trainer_lstm.net_lstm_mean, args.project+args.exp+'/'+configs['save_encoder_folder']+'/'+configs['data_name']+'_mean_model_encoder' )  
        r2_train, r2_valid, r2_test, train_results_np, valid_results_np, test_results_np = trainer_lstm.LSTMEncoderEvaluation(scalerx, scalery, save_results=False, return_results=True)

        ### LSTM layer predictions
        pred_train_LSTM = trainer_lstm.net_lstm_mean.lstm_mean(xTrain)
        pred_valid_LSTM = trainer_lstm.net_lstm_mean.lstm_mean(xValid)
        pred_test_LSTM = trainer_lstm.net_lstm_mean.lstm_mean(xTest)

    plotter = CL_plotter(configs)
    plotter.plotLSTM_raw_Y(yTrain, yValid, yTest, xTrain_idx, xValid_idx, scalery=scalery, \
        ylim_1=None, ylim_2=None, savefig=args.project+args.exp+'/'+configs['save_encoder_folder']+'/'+'original_split.png') 
    # configs['plot_ylims'] = [[0., 0.9], [0., 0.9]]
    plotter.plotLSTM_encoder(xTrain_idx, xValid_idx, train_results_np, valid_results_np, test_results_np, figname='LSTM encoder for {}'.format(configs['data_name']), \
        show_plot=False, ylim_1=None, ylim_2=None, savefig=args.project+args.exp+'/'+configs['save_encoder_folder']+'/'+'encoder.png')
    # ylim_1=configs['plot_ylims'][0], ylim_2=configs['plot_ylims'][1]
    

    if configs['save_encoder_results'] is not None:
        ### combine train/valid data
        trainValid_np = np.zeros((len(train_results_np)+len(valid_results_np), 2))
        for i in range(2):
            trainValid_np[xTrain_idx, i] = train_results_np[:, i]
            trainValid_np[xValid_idx, i] = valid_results_np[:, i]

        np.save(args.project+args.exp+'/'+configs['save_encoder_results']+'/'+'trainValid_np.npy', trainValid_np)
        np.save(args.project+args.exp+'/'+configs['save_encoder_results']+'/'+'test_np.npy', test_results_np)
        print('--- Encoder results saved to:')
        print(args.project+args.exp+'/'+configs['save_encoder_results']+'/')


    if configs['ylog_trans']:
        ori_yTrain = np.exp(ori_yTrain)
        ori_yValid = np.exp(ori_yValid)
        ori_yTest = np.exp(ori_yTest)

    if configs['save_encoder_pred']:
        args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'
        np.savetxt(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_trainValid_X.txt', np.vstack((pred_train_LSTM, pred_valid_LSTM)))
        np.savetxt(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_test_X.txt', pred_test_LSTM)
        np.savetxt(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_trainValid_Y.txt', np.vstack((ori_yTrain, ori_yValid)))
        np.savetxt(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_encoder_test_Y.txt', ori_yTest)
        np.savetxt(args.project+args.exp+'/'+configs['save_encoder_pred_folder']+'/'+configs['data_name']+'_trainValid_idx.txt', np.append(xTrain_idx.values, xValid_idx.values).astype(int), fmt='%s')


elif args.mode == 'PI3NN_MLP':
    print('----------------------------------------------------------------')
    print('--- Training for MLP-PI3NN based on LSTM encoder predictions ---')
    print('----------------------------------------------------------------')

    with open(args.configs, 'r') as json_file_r:
        loaded_json = json.load(json_file_r)

    configs = loaded_json
    configs['project'] = args.project
    configs['exp'] = args.exp

    ### For .txt files only
    # # (1) check and PI3NN results folder, create one if not exist
    # utils.check_PI3NN_folder(args.project+args.exp+'/'+configs['PI3NN_results_folder'])
    # # (2) check the encoder results, check if the 5 files are available, if not, stop the program
    # print('--- Checking encoder prediction results...')
    # utils.check_encoder_predictions(args.project+args.exp+'/'+configs['encoder_path'], data_name=configs['data_name'])

    if configs['stop_losses'][1] is not None:
        end_up_train_loss = configs['stop_losses'][1]
        print('--- Assigned stopping train loss for UP: {:.4f}'.format(end_up_train_loss))
    else:
        end_up_train_loss = None
    if configs['stop_losses'][2] is not None:
        end_down_train_loss = configs['stop_losses'][2]
        print('--- Assigned stopping train loss for DOWN: {:.4f}'.format(end_down_train_loss))
    else:
        end_down_train_loss = None

    if configs['test_biases']:
        bias_list = configs['test_biases_list']
        end_up_train_loss = None
        end_down_train_loss = None
    else: # single run with one pair of pre-assigned biases
        bias_list = [0.0]

    for ii in range(len(bias_list)):
        if configs['test_biases']:
            tmp_bias = bias_list[ii]
            configs['bias_up'] = tmp_bias
            configs['bias_down'] = tmp_bias
            configs['lr'] = [0.001, 0.005, 0.005]
            configs['optimizers'] = ['Adam', 'Adam', 'Adam']
        if ii > 0:
            configs['stop_losses'] = [None, None, None]  #[None, end_up_train_loss, end_down_train_loss]
            for i in range(2):
                if configs['stop_losses'][i+1] is not None:
                    configs['Max_iter'][i+1] = 10000000
            # configs['Max_iter'][0] = configs['Max_iter'][0]

            # configs['optimizers'] = ['Adam', 'Adam', 'Adam']   # ['Adam', 'SGD', 'SGD'] ['Adam', 'Adam', 'Adam'] 
            # configs['lr'] = [0.001, 0.002, 0.002]            # [0.005, 0.005, 0.005]

        random.seed(configs['seed'])
        np.random.seed(configs['seed'])
        tf.random.set_seed(configs['seed'])


        ### Test loading the CNN encoder predicted data
        loaded_npz = np.load(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_encoder_'+configs['load_encoder_name']+'_test_ds.npz')
        ### list all keys
        keys_extract = list(loaded_npz.keys())

        ### need to find the corresponding Y labels

        ### extract and vstack all CNN encoder predictions
        print('--- Loading '+args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_encoder_'+configs['load_encoder_name']+'_test_ds.npz')


        cnn_layers_pred_list = []
        for ii, key in enumerate(tqdm(keys_extract)):
            # print(ii)
            cnn_layers_pred_list.append(loaded_npz[key])

        cnn_layers_pred_np = np.vstack(cnn_layers_pred_list)
        print(cnn_layers_pred_np)

        pi3nn_cnn_X = cnn_layers_pred_np[:, :-1]
        pi3nn_cnn_Y = cnn_layers_pred_np[:, -1]


        print('PI3NN----ccccccccccccccccccc')

        print(pi3nn_cnn_X.shape)
        print(pi3nn_cnn_Y.shape)

        ### trainvalid/test split
        print(pi3nn_cnn_X.shape)
        print(pi3nn_cnn_Y.shape)
        xTrainValid, xTest, yTrainValid, yTest = train_test_split(pi3nn_cnn_X, pi3nn_cnn_Y, test_size=0.1, random_state=0)

        print(xTrainValid.shape)
        print(xTest.shape)
        print(yTrainValid.shape)
        print(yTest.shape)

        # exit()
        ### train/valid split
        xTrain, xValid, yTrain, yValid = train_test_split(xTrainValid, yTrainValid, test_size=0.1, random_state=0)

        yTrain = yTrain.reshape(-1, 1)
        yValid = yValid.reshape(-1, 1)
        yTest = yTest.reshape(-1, 1)

        # xTrain = xTrain[:, :50]
        # xValid = xValid[:, :50]
        # xTest = xTest[:, :50]




        # print(xTrainValid.shape)
        # print(xTest.shape)
        # print(yTrainValid.shape)
        # print(yTest.shape)

        print('sssssssssss')
        print(xTrain.shape)
        print(xValid.shape)
        print(xTest.shape)
        print(yTrain.shape)
        print(yValid.shape)
        print(yTest.shape)
        # exit()



        print('WWWWWWWWWWWWWWWWWWWWW')
        print(np.average(yTrain))
        print(np.average(yValid))
        print(np.average(yTest))
        print(np.std(yTrain))
        print(np.std(yValid))
        print(np.std(yTest))
        # exit()






        # ### Load the LSTM encoder predicted data
        # trainValid_X = np.loadtxt(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_encoder_trainValid_X.txt')
        # test_X = np.loadtxt(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_encoder_test_X.txt')
        # trainValid_Y = np.loadtxt(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_encoder_trainValid_Y.txt')
        # test_Y = np.loadtxt(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_encoder_test_Y.txt')
        # trainValid_idx = np.loadtxt(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_trainValid_idx.txt').astype(int)  ### original

        # ### re-order the data (FOR time-series data)
        # trainValid_Y = trainValid_Y[np.argsort(trainValid_idx)]
        # trainValid_X = trainValid_X[np.argsort(trainValid_idx), :]
        # trainValid_idx = np.arange(len(trainValid_Y))

        # trainValid_Y = trainValid_Y.reshape(-1, 1)
        # test_Y = test_Y.reshape(-1, 1)

        # if configs['ylog_trans']:
        #     trainValid_Y = np.log(trainValid_Y)
        #     test_Y = np.log(test_Y)

        # if configs['ypower_root_trans'][0]:
        #     # print(test_Y)
        #     trainValid_Y = np.power(trainValid_Y, (1/configs['ypower_root_trans'][1]))
        #     test_Y = np.power(test_Y, (1/configs['ypower_root_trans'][1]))

        num_inputs = utils.getNumInputsOutputs(xTrainValid)
        num_outputs = utils.getNumInputsOutputs(yTrainValid)

        # num_inputs = utils.getNumInputsOutputs(trainValid_X)
        # num_outputs = utils.getNumInputsOutputs(trainValid_Y)


        print('--- Num inputs: {}'.format(num_inputs))
        print('--- num_outputs: {}'.format(num_outputs))


        # ### train/valid split
        # xTrain, xValid, yTrain, yValid = train_test_split(pd.DataFrame(trainValid_X), pd.DataFrame(trainValid_Y), test_size=20, random_state=0)

        print('xTrain shape: {}'.format(xTrain.shape))
        print('xValid shape: {}'.format(xValid.shape))
        print('xTest shape: {}'.format(xTest.shape))

        print('yTrain shape: {}'.format(yTrain.shape))
        print('yValid shape: {}'.format(yValid.shape))
        print('yTest shape: {}'.format(yTest.shape))
        
        # xTrain_idx = xTrain.index
        # xValid_idx = xValid.index

        #### to float32
        xTrain, yTrain = xTrain.astype(np.float32), yTrain.astype(np.float32)
        xValid, yValid = xValid.astype(np.float32), yValid.astype(np.float32)
        xTest, yTest = xTest.astype(np.float32), yTest.astype(np.float32)


        print(np.isnan(np.min(xTrain)))
        print(np.isnan(np.min(xValid)))
        print(np.isnan(np.min(xTest)))
        print(np.isnan(np.min(yTrain)))
        print(np.isnan(np.min(yValid)))
        print(np.isnan(np.min(yTest)))

        print('TTTTTTTTTTTT')
        # exit()

        #### scaling        
        if configs['load_PI3NN_MLP']: ### load the scalers
            scalerx, scalery = utils.load_scalers(args.project+args.exp+'/'+configs['load_PI3NN_MLP_folder'])
        else:
            scalerx = StandardScaler()
            scalery = StandardScaler()

        xTrain = scalerx.fit_transform(xTrain)
        xValid = scalerx.transform(xValid)
        xTest = scalerx.transform(xTest)

        yTrain = scalery.fit_transform(yTrain)
        yValid = scalery.transform(yValid)
        yTest = scalery.transform(yTest)

        print(np.isnan(np.min(xTrain)))
        print(np.isnan(np.min(xValid)))
        print(np.isnan(np.min(xTest)))
        print(np.isnan(np.min(yTrain)))
        print(np.isnan(np.min(yValid)))
        print(np.isnan(np.min(yTest)))

        print('AAAAAAAFFFFFFFFTTTTTTTTTTTT')
        # exit()

        # yTest = scalery.transform(yTest.reshape(-1, 1))

        if configs['load_PI3NN_MLP']:  ### load the saved PI3NN model
            net_mean = UQ_Net_mean_TF2(configs, num_inputs, num_outputs)
            net_up = UQ_Net_std_TF2(configs, num_inputs, num_outputs, net='up', bias=configs['bias_up'])
            net_down = UQ_Net_std_TF2(configs, num_inputs, num_outputs, net='down', bias=configs['bias_down'])
            trainer = CL_trainer(configs, net_mean, net_up, net_down, xTrain, yTrain, xValid=xValid, yValid=yValid, xTest=xTest, yTest=yTest, \
                train_idx=xTrain_idx, valid_idx=xValid_idx, scalerx=scalerx, scalery=scalery, testDataEvaluationDuringTrain=True, allTestDataEvaluationDuringTrain=True)

            tmp_mean, tmp_up, tmp_down = utils.load_PI3NN_saved_models(args.project+args.exp+'/'+configs['load_PI3NN_MLP_folder'], data_name=configs['data_name'])
            trainer.trainSteps.net_mean = tmp_mean
            trainer.trainSteps.net_std_up = tmp_up
            trainer.trainSteps.net_std_down = tmp_down

        else:
            net_mean = UQ_Net_mean_TF2(configs, num_inputs, num_outputs)
            net_up = UQ_Net_std_TF2(configs, num_inputs, num_outputs, net='up', bias=configs['bias_up'])
            net_down = UQ_Net_std_TF2(configs, num_inputs, num_outputs, net='down', bias=configs['bias_down'])

            # print('sssssssssss22222222222222222222222')
            # print(xTrain.shape)
            # print(xValid.shape)
            # print(xTest.shape)
            # print(yTrain.shape)
            # print(yValid.shape)
            # print(yTest.shape)
            # exit()

            # trainer_CNN = CL_trainer_CNN(configs, train_ds, valid_ds, test_ds, net_CNN_mean, net_cnn_up=None, net_cnn_down=None)
            trainer = CL_trainer(configs, net_mean, net_up, net_down, xTrain, yTrain, xValid=xValid, yValid=yValid, xTest=xTest, yTest=yTest, \
                train_idx=None, valid_idx=None, scalerx=scalerx, scalery=scalery, testDataEvaluationDuringTrain=True, allTestDataEvaluationDuringTrain=True)

            trainer.train()

            # exit()              

            if configs['save_PI3NN_MLP']:
                ###  save tf models and scalers
                utils.save_PI3NN_models(args.project+args.exp+'/'+configs['load_PI3NN_MLP_folder'], 
                    trainer.trainSteps.net_mean,
                    trainer.trainSteps.net_std_up,
                    trainer.trainSteps.net_std_down,
                    scalerx=scalerx, scalery=scalery, data_name=configs['data_name'])

        trainer.boundaryOptimization(verbose=1)  # boundary optimization
        trainer.testDataPrediction()    # evaluation of the trained nets on testing data
        trainer.capsCalculation(final_evaluation=True, verbose=1)       # metrics calculation
        # trainer.saveResultsToTxt()      # save results to txt file




        ### Below code is designed originally for PI3NN-LSTM, I will modifiy them later for CNN cases.



        r2_train, r2_valid, r2_test, train_results_np, valid_results_np, test_results_np = trainer.modelEvaluation_MLPI3NN(scalerx, scalery, save_results=True, return_results=True)
        
        print('--- Final eval r2_train: {:.4f}'.format(r2_train))
        print('--- Final eval r2_valid: {:.4f}'.format(r2_valid))
        print('--- Final eval r2_test: {:.4f}'.format(r2_test))

        # exit()
        # plotter = CL_plotter(configs)




 
        if configs['save_PI3NN_MLP_pred']:
            ### save predicted results to npy
            trainValid_np = np.zeros((len(train_results_np)+len(valid_results_np), 4))
            for i in range(4):
                trainValid_np[xTrain_idx, i] = train_results_np[:, i]
                trainValid_np[xValid_idx, i] = valid_results_np[:, i]

            if configs['test_biases']:
                np.save(args.project+args.exp+'/'+configs['save_PI3NN_MLP_pred_folder']+'/'+'trainValid_np_bias_{}.npy'.format(tmp_bias), trainValid_np)
                np.save(args.project+args.exp+'/'+configs['save_PI3NN_MLP_pred_folder']+'/'+'test_np_bias_{}.npy'.format(tmp_bias), test_results_np)
                print('--- PI3NN prediction on testing data saved to:')
                print(args.project+args.exp+'/'+configs['save_PI3NN_MLP_pred_folder']+'/'+'trainValid_np_bias_{}.npy'.format(tmp_bias)) 
                print(args.project+args.exp+'/'+configs['save_PI3NN_MLP_pred_folder']+'/'+'test_np_bias_{}.npy'.format(tmp_bias)) 
            else:
                np.save(args.project+args.exp+'/'+configs['save_PI3NN_MLP_pred_folder']+'/'+'trainValid_np.npy', trainValid_np)
                np.save(args.project+args.exp+'/'+configs['save_PI3NN_MLP_pred_folder']+'/'+'test_np.npy', test_results_np)
                print('--- PI3NN prediction on testing data saved to:')
                print(args.project+args.exp+'/'+configs['save_PI3NN_MLP_pred_folder']+'/'+'trainValid_np.npy') 
                print(args.project+args.exp+'/'+configs['save_PI3NN_MLP_pred_folder']+'/'+'test_np.npy') 


        # #### One time runnning
        # plotter.plot_MLPI3NN(xTrain_idx, xValid_idx, train_results_np, valid_results_np, test_results_np, figname='PI3NN-LSTM results for data: {}, biases={}, {}'.format(configs['data_name'], configs['bias_up'], configs['bias_down']), \
        # train_PIW_quantile=configs['train_PIW_quantile'], gaussian_filter=True, 
        #  savefig=args.project+args.exp+'/'+configs['PI3NN_results_folder']+'/'+'PI3NN_{}_bias_{}_{}_q_{}.png'.format(configs['data_name'], configs['bias_up'], configs['bias_down'], configs['train_PIW_quantile']),\
        #  save_results=None)

        if ii == 0 and len(bias_list)>1:
            end_up_train_loss = trainer.end_up_train_loss
            end_down_train_loss = trainer.end_down_train_loss

            print('--- end up train loss: {}'.format(end_up_train_loss))
            print('--- end down train loss: {}'.format(end_down_train_loss))

