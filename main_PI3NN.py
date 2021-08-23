
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import datetime
from tqdm import tqdm
import time
# import keras

import itertools
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Force using CPU globally by hiding GPU(s)
# tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

from src.DataLoaders.benchmark_data_loader_V2 import CL_dataLoader
from src.Networks.network_V2 import UQ_Net_mean_TF2, UQ_Net_std_TF2
from src.Networks.network_V2 import CL_UQ_Net_train_steps
from src.Trainers.trainers import CL_trainer
from src.Optimizations.boundary_optimizer import CL_boundary_optimizer
from src.Visualizations.visualization import CL_plotter


'''
PI3NN method for regression problems
'''
configs = {'data_name':'flight_delay_test_1',    # flight_delay_test_1 # yacht
           'original_data_path': '/datasets/UCI_datasets/',
           'splitted_data_path': '/datasets/UCI_TrainTest_Split/',
           'split_seed': 1,                 
           'split_test_ratio': 0.1,         # ratio of the testing data during random split
           'seed': 10,                      # general random seed
           'num_neurons_mean_net': 100,     # number of neurons in hidden layer for the 'MEAN' network
           'num_neurons_up_down_net': 100,  # number of neurons in hidden layer for the 'UP'and 'DOWN' network
           'quantile': 0.95,                # target percentile for optimization step# target percentile for optimization step
           'Max_iter': 5000,
           'lr': [0.02, 0.02, 0.02],           # learning rate
           'optimizers': ['Adam', 'Adam', 'Adam'],
           'learning_decay': True,
           'exponential_decay': True,
           'decay_steps': 15,  # 10
           'decay_rate': 0.6,  # 0.8
           'saveWeights': False,
           'loadWeights_test': False,
           'early_stop': True,
           'early_stop_start_iter': 60, # 10
           'wait_patience': 80,
           'restore_best_weights': True,
           'save_loss_history': False,
           'save_loss_history_path': './Results_PI3NN/loss_history/',
           'plot_loss_history' : False,
           'plot_loss_history_path':'./Results_PI3NN/loss_curves/',
           'verbose': 1,
           'experiment_id': 1,

           'batch_training': False,
           'batch_size': 1024,
           'batch_shuffle': True,
           'batch_shuffle_buffer': 1024
          }


''' Data selection '''
# print('--- Dataset: {}'.format(configs['data_name']))
# print('--- Splitting seed and random seed: {}, {}'.format(configs['split_seed'], configs['seed']))
# random.seed(configs['seed'])
# np.random.seed(configs['seed'])
# tf.random.set_seed(configs['seed'])


##### ----- (1) UCI data sets (Random splitting) ----- #####
'''Available data sets (keywords):
boston, concrete, energy-efficiency, kin8nm, naval, powerplant, protein, wine, yacht, MSD '''

# dataLoader = CL_dataLoader(os.getcwd()+configs['original_data_path'])
# X_data_load, Y_data_load = dataLoader.load_single_dataset(configs['data_name'])
# xTrain, xTest, yTrain, yTest = train_test_split(X_data_load, Y_data_load, 
#                                                 test_size=configs['split_test_ratio'],
#                                                 random_state=configs['split_seed'],
#                                                 shuffle=True)  

# print('wwwwwwwwwwwwwwwwwwwwwwwwwwwwww')
# print(xTrain.shape)
# print(yTrain.shape)
# print(xTest.shape)
# print(yTest.shape)

# print(yTest)

##### ----- (2) Pre-split UCI data sets  ----- #####
# dataLoader = CL_dataLoader(os.getcwd()+configs['original_data_path'])
# xyTrain_load, xyTest_load = dataLoader.LoadData_Splitted_UCI(configs['data_name'], os.getcwd()+configs['splitted_data_path'], configs['split_seed'])
# saveFigPrefix = configs['data_name']

# ''' Inputs/output data selection ---  '''
# if configs['data_name'] == 'energy' or configs['data_name'] == 'naval':
#     xTrain = xyTrain_load[:, :-2] ## all columns except last two columns as inputs
#     yTrain = xyTrain_load[:, -1] ## last column as output
#     xTest = xyTest_load[:, :-2]
#     yTest = xyTest_load[:, -1]
# else:
#     xTrain = xyTrain_load[:, :-1]
#     yTrain = xyTrain_load[:, -1]
#     xTest = xyTest_load[:, :-1]
#     yTest = xyTest_load[:, -1]



''' Standardize inputs '''
# xTrain, xTrain_mean, xTrain_std = dataLoader.standardizer(xTrain)
# xTest = (xTest - xTrain_mean) / xTrain_std

# yTrain, yTrain_mean, yTrain_std = dataLoader.standardizer(yTrain)
# yTest = (yTest - yTrain_mean) / yTrain_std

# num_inputs = dataLoader.getNumInputsOutputs(xTrain)
# num_outputs = dataLoader.getNumInputsOutputs(yTrain)

# ''' Create network instances'''
# net_mean = UQ_Net_mean_TF2(num_inputs, num_outputs, num_neurons=configs['num_neurons_mean_net'])
# net_std_up = UQ_Net_std_TF2(num_inputs, num_outputs,  num_neurons=configs['num_neurons_up_down_net'])
# net_std_down = UQ_Net_std_TF2(num_inputs, num_outputs, num_neurons=configs['num_neurons_up_down_net'])

# ''' Initialize trainer '''
# trainer = CL_trainer(configs, net_mean, net_std_up, net_std_down, xTrain, yTrain, xTest=xTest, yTest=yTest)

# trainer.train()                 # training for 3 networks
# trainer.boundaryOptimization()  # boundary optimization
# trainer.testDataPrediction()    # evaluation of the trained nets on testing data
# trainer.capsCalculation()       # metrics calculation
# trainer.saveResultsToTxt()      # save results to txt file





# ##### ----- (3) Flight delays data sets ----- #####
# dataLoader = CL_dataLoader(os.getcwd()+'/datasets/flight_delays_data/')
# df_train, df_test_1, df_test_2, df_test_3, df_test_4, df_test_5 = dataLoader.load_flight_delays_df()
# # print(df_train)

# ''' selected features
# Inputs:
# (1) (Distance): distance of the flight in miles
# (2) (Month): DayOfMonth
# (3) (DayofMonth):
# (4) (DayOfWeek)
# (5) (DepTime): actual departure time
# (6) (ArrTime): actual arrival time
# (7) (AirTime): air time in minutes

# Outputs:
# (1) (ArrDelay): the arrival delay in minutes
# '''

### test new flight delay loader
dataLoader = CL_dataLoader(os.getcwd()+'/datasets/flight_delays_data/')
xTrain, yTrain, test_data_list  = dataLoader.load_flight_delays_df()

# # print(df_train.head(3).T)
print('--- Dataset: Flight delays')
random.seed(configs['seed'])
np.random.seed(configs['seed'])
tf.random.set_seed(configs['seed'])

num_inputs = dataLoader.getNumInputsOutputs(xTrain)
num_outputs = dataLoader.getNumInputsOutputs(yTrain)

# for i in range(2):
    # ''' Create network instances'''
net_mean = UQ_Net_mean_TF2(num_inputs, num_outputs, num_neurons=configs['num_neurons_mean_net'])
net_std_up = UQ_Net_std_TF2(num_inputs, num_outputs,  num_neurons=configs['num_neurons_up_down_net'])
net_std_down = UQ_Net_std_TF2(num_inputs, num_outputs, num_neurons=configs['num_neurons_up_down_net'])

# ''' Initialize trainer '''
trainer = CL_trainer(configs, net_mean, net_std_up, net_std_down, xTrain, yTrain)
trainer.train(flightDelayData=True, flightDelayTestDataList=test_data_list)                 # training for 3 networks
trainer.boundaryOptimization()  # boundary optimization
trainer.testDataPrediction(flightDelayData=True, flightDelayTestDataList=test_data_list)    # evaluation of the trained nets on testing data
trainer.capsCalculation(flightDelayData=True, flightDelayTestDataList=test_data_list)       # metrics calculation
# trainer.saveResultsToTxt()      # save results to txt file

trainer.saveResultsToTxt_flightDelay(plot=True)

trainer.plot_PICPMPIW(savingPath='./Results_PI3NN/flightDelay_MPIW.svg')



##### ----- (X) User input data sets  ----- #####