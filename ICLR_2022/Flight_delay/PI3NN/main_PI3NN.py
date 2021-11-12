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
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

from src.DataLoaders.benchmark_data_loader_V2 import CL_dataLoader
from src.Networks.network_V2 import UQ_Net_mean_TF2, UQ_Net_std_TF2
from src.Networks.network_V2 import CL_UQ_Net_train_steps
from src.Trainers.trainers import CL_trainer
from src.Optimizations.boundary_optimizer import CL_boundary_optimizer
from src.Visualizations.visualization import CL_plotter


'''
PI3NN method for flight delay example
'''
configs = {'data_name':'flight_delay_test_five',  
           'original_data_path': '/datasets/UCI_datasets/',
           'splitted_data_path': '/datasets/UCI_TrainTest_Split/',
           'split_seed': 1,                 
           'split_test_ratio': 0.1,         # ratio of the testing data during random split
           'seed': 10,                      # general random seed
           'num_neurons_mean_net': 100,     # number of neurons in hidden layer for the 'MEAN' network
           'num_neurons_up_down_net': 100,  # number of neurons in hidden layer for the 'UP'and 'DOWN' network
           'quantile': 0.95,                # target percentile for optimization step# target percentile for optimization step
           'Max_iter': 1000, # 50000, # 5000,
           'lr': [0.02, 0.02, 0.02],  # 0.02         # learning rate
           'optimizers': ['Adam', 'Adam', 'Adam'], # ['Adam', 'Adam', 'Adam'],
           'learning_decay': True,
           'exponential_decay': False,
           'decay_steps': 100,  # 10
           'decay_rate': 0.9,  # 0.6
           'saveWeights': False,
           'loadWeights_test': False,
           'early_stop': True,
           'early_stop_start_iter': 100, # 60
           'wait_patience': 200,
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




# ##### ----- (3) Flight delays data sets ----- #####

# ''' selected features
# Inputs:
# (1) (Distance): distance of the flight in miles
# (2) (DayofMonth)
# (3) (DayOfWeek)
# (4) (DepDelay) departure delay
# (5) (AirTime) air time in minutes
# (6) (TaxiOut)

# Outputs:
# (1) (ArrDelay): the arrival delay in minutes
# '''
dataLoader = CL_dataLoader('../flight_delay_data/', configs)

xTrain, yTrain, test_data_list = dataLoader.load_flight_delays_df_2()

# # print(df_train.head(3).T)
print('--- Dataset: {}'.format(configs['data_name']))
random.seed(configs['seed'])
np.random.seed(configs['seed'])
tf.random.set_seed(configs['seed'])

num_inputs = dataLoader.getNumInputsOutputs(xTrain)
num_outputs = dataLoader.getNumInputsOutputs(yTrain)

net_mean = UQ_Net_mean_TF2(num_inputs, num_outputs, num_neurons=configs['num_neurons_mean_net'])
net_std_up = UQ_Net_std_TF2(num_inputs, num_outputs,  num_neurons=configs['num_neurons_up_down_net'], bias=2.0)  ## With OOD detection, change to 0. for No OOD detection
net_std_down = UQ_Net_std_TF2(num_inputs, num_outputs, num_neurons=configs['num_neurons_up_down_net'], bias=2.0) ## With OOD detection, change to 0. for No OOD detection

# # ''' Initialize trainer '''
trainer = CL_trainer(configs, net_mean, net_std_up, net_std_down, xTrain, yTrain, flightDelayTestDataList=test_data_list)
trainer.train(flightDelayTestDataList=test_data_list, testDataEvaluation=True)  
trainer.boundaryOptimization()  # boundary optimization
trainer.testDataPrediction(flightDelayTestDataList=test_data_list)    # evaluation of the trained nets on testing data
trainer.capsCalculation(flightDelayTestDataList=test_data_list)       # metrics calculation
trainer.confidenceScoreCalculation(flightDelayTestDataList=test_data_list)
