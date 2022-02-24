import os
import tensorflow as tf
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
num_threads = 2
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)

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
from pi3nn.Trainers.trainers import CL_trainer
from pi3nn.Trainers.trainer_TS import CL_trainer_TS
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

    # (1) check and PI3NN results folder, create one if not exist
    utils.check_PI3NN_folder(args.project+args.exp+'/'+configs['PI3NN_results_folder'])
    # (2) check the encoder results, check if the 5 files are available, if not, stop the program
    print('--- Checking encoder prediction results...')
    utils.check_encoder_predictions(args.project+args.exp+'/'+configs['encoder_path'], data_name=configs['data_name'])


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
            configs['Max_iter'] = [configs['Max_iter'][0], 10000000, 10000000]
            configs['stop_losses'] = [None, end_up_train_loss, end_down_train_loss]
            # configs['optimizers'] = ['Adam', 'Adam', 'Adam']   # ['Adam', 'SGD', 'SGD'] ['Adam', 'Adam', 'Adam'] 
            # configs['lr'] = [0.001, 0.002, 0.002]            # [0.005, 0.005, 0.005]

        random.seed(configs['seed'])
        np.random.seed(configs['seed'])
        tf.random.set_seed(configs['seed'])

        ### Load the LSTM encoder predicted data
        trainValid_X = np.loadtxt(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_encoder_trainValid_X.txt')
        test_X = np.loadtxt(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_encoder_test_X.txt')
        trainValid_Y = np.loadtxt(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_encoder_trainValid_Y.txt')
        test_Y = np.loadtxt(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_encoder_test_Y.txt')
        trainValid_idx = np.loadtxt(args.project+args.exp+'/'+configs['encoder_path']+'/'+configs['data_name']+'_trainValid_idx.txt').astype(int)  ### original

        ### re-order the data
        trainValid_Y = trainValid_Y[np.argsort(trainValid_idx)]
        trainValid_X = trainValid_X[np.argsort(trainValid_idx), :]
        trainValid_idx = np.arange(len(trainValid_Y))

        trainValid_Y = trainValid_Y.reshape(-1, 1)
        test_Y = test_Y.reshape(-1, 1)

        if configs['ylog_trans']:
            trainValid_Y = np.log(trainValid_Y)
            test_Y = np.log(test_Y)

        if configs['ypower_root_trans'][0]:
            # print(test_Y)
            trainValid_Y = np.power(trainValid_Y, (1/configs['ypower_root_trans'][1]))
            test_Y = np.power(test_Y, (1/configs['ypower_root_trans'][1]))

        num_inputs = utils.getNumInputsOutputs(trainValid_X)
        num_outputs = utils.getNumInputsOutputs(trainValid_Y)

        print('--- Num inputs: {}'.format(num_inputs))
        print('--- num_outputs: {}'.format(num_outputs))

        ### train/valid split
        xTrain, xValid, yTrain, yValid = train_test_split(pd.DataFrame(trainValid_X), pd.DataFrame(trainValid_Y), test_size=20, random_state=0)

        print('xTrain shape: {}'.format(xTrain.shape))
        print('xValid shape: {}'.format(xValid.shape))
        print('yTrain shape: {}'.format(yTrain.shape))
        print('yValid shape: {}'.format(yValid.shape))
        
        xTrain_idx = xTrain.index
        xValid_idx = xValid.index

        #### to float32
        xTrain, yTrain = xTrain.astype(np.float32), yTrain.astype(np.float32)
        xValid, yValid = xValid.astype(np.float32), yValid.astype(np.float32)
        xTest, yTest = test_X.astype(np.float32), test_Y.astype(np.float32)

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
        yTest = scalery.transform(yTest.reshape(-1, 1))

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

            trainer = CL_trainer(configs, net_mean, net_up, net_down, xTrain, yTrain, xValid=xValid, yValid=yValid, xTest=xTest, yTest=yTest, \
                train_idx=xTrain_idx, valid_idx=xValid_idx, scalerx=scalerx, scalery=scalery, testDataEvaluationDuringTrain=True, allTestDataEvaluationDuringTrain=True)
            trainer.train()              

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
        r2_train, r2_valid, r2_test, train_results_np, valid_results_np, test_results_np = trainer.modelEvaluation_MLPI3NN(scalerx, scalery, save_results=False, return_results=True)
        plotter = CL_plotter(configs)

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


        #### One time runnning
        plotter.plot_MLPI3NN(xTrain_idx, xValid_idx, train_results_np, valid_results_np, test_results_np, figname='PI3NN-LSTM results for data: {}, biases={}, {}'.format(configs['data_name'], configs['bias_up'], configs['bias_down']), \
        train_PIW_quantile=configs['train_PIW_quantile'], gaussian_filter=True, 
         savefig=args.project+args.exp+'/'+configs['PI3NN_results_folder']+'/'+'PI3NN_{}_bias_{}_{}_q_{}.png'.format(configs['data_name'], configs['bias_up'], configs['bias_down'], configs['train_PIW_quantile']),\
         save_results=None)

        if ii == 0 and len(bias_list)>1:
            end_up_train_loss = trainer.end_up_train_loss
            end_down_train_loss = trainer.end_down_train_loss

            print('--- end up train loss: {}'.format(end_up_train_loss))
            print('--- end down train loss: {}'.format(end_down_train_loss))

