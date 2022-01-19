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

# Force using CPU globally by hiding GPU(s), comment the line of code below to enable GPU
tf.config.set_visible_devices([], 'GPU')

from pathlib import Path
import datetime
from tqdm import tqdm
import time
# import keras

import argparse
import json
import itertools
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

from pi3nn.DataLoaders.data_loaders import CL_dataLoader
from pi3nn.Networks.networks import UQ_Net_mean_TF2, UQ_Net_std_TF2
from pi3nn.Networks.networks import CL_UQ_Net_train_steps
from pi3nn.Trainers.trainers import CL_trainer
from pi3nn.Optimizations.boundary_optimizer import CL_boundary_optimizer
from pi3nn.Visualizations.visualization import CL_plotter
from pi3nn.Optimizations.params_optimizer import CL_params_optimizer
from pi3nn.Utils.Utils import CL_Utils

utils = CL_Utils()
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='boston', help='example data names: boston, concrete, energy, kin8nm, wine, yacht')
parser.add_argument('--mode', type=str, default='auto', help='auto or manual mode')
parser.add_argument('--quantile', type=float, default=0.95)
args = parser.parse_args()

''' 
If you would like to customize the data loading and pre-processing, we recommend you to write the 
functions in src/DataLoaders/data_loaders.py and call it from here. Or write them directly here.
We provide an example on 'boston_housing.txt' dataset below:      

'''


##########################################################
################## Data Loading Section ##################
##########################################################
data_dir = './datasets/UCI_datasets/'
dataLoader = CL_dataLoader(original_data_path=data_dir)
if args.data == 'energy' or args.data == 'naval':
    X, Y, _ = dataLoader.load_single_dataset(args.data) 
else:
    X, Y = dataLoader.load_single_dataset(args.data)
if len(Y.shape) == 1:
    Y = Y.reshape(-1, 1)


# ### Split train/test data  (manual specification or random split using sklearn)
# # Ntest = 100
# # xTrain, xTest = X[:-Ntest, :], X[-Ntest:, :]
# # yTrain, yTest = Y[:-Ntest, :], Y[-Ntest:, :]

# ## or random split
xTrainValid, xTest, yTrainValid, yTest = train_test_split(X, Y, test_size=0.1, random_state=1, shuffle=True)
## Split the validation data
xTrain, xValid, yTrain, yValid = train_test_split(xTrainValid, yTrainValid, test_size=0.1, random_state=1, shuffle=True)  


### Data normalization
scalar_x = StandardScaler()
scalar_y = StandardScaler()

xTrain = scalar_x.fit_transform(xTrain)
xValid = scalar_x.fit_transform(xValid)
xTest = scalar_x.transform(xTest)

yTrain = scalar_y.fit_transform(yTrain)
yValid = scalar_y.fit_transform(yValid)
yTest = scalar_y.transform(yTest)

#########################################################
############## End of Data Loading Section ##############
#########################################################

num_inputs = utils.getNumInputsOutputs(xTrain)
num_outputs = utils.getNumInputsOutputs(yTrain)

configs = {}
### Some other general input info
configs['data_name'] = 'bostonHousing'
configs['quantile'] = args.quantile  # # target percentile for optimization step# target percentile for optimization step, 
                                     # 0.95 by default if not specified 
configs['experiment_id'] = 1
configs['verbose'] = 1
configs['save_loss_history'] = False
configs['save_loss_history_path'] = './Results_PI3NN/loss_history/'
configs['plot_loss_history'] = False
configs['plot_loss_history_path'] ='./Results_PI3NN/loss_curves/'


######################################################################################
# Multiple quantiles, comment out this line in order to run single quantile estimation
# configs['quantile_list'] = np.arange(0.05, 1.00, 0.05) # 0.05-0.95
######################################################################################


if args.mode == 'manual':
    print('--- Running on manual mode.')
    ### specify hypar-parameters for the training
    configs['seed'] = 10                # general random seed
    configs['num_neurons_mean'] = [50]  # hidden layer(s) for the 'MEAN' network
    configs['num_neurons_up'] = [50]    # hidden layer(s) for the 'UP' network
    configs['num_neurons_down'] = [50]  # hidden layer(s) for the 'DOWN' network
    configs['Max_iter'] = 5000 # 5000,
    configs['lr'] = [0.02, 0.02, 0.02]  # 0.02         # learning rate
    configs['optimizers'] = ['Adam', 'Adam', 'Adam'] # ['SGD', 'SGD', 'SGD'],
    configs['exponential_decay'] = True
    configs['decay_steps'] = 3000 # 3000  # 10
    configs['decay_rate'] = 0.9  # 0.6
    configs['saveWeights'] = False
    configs['loadWeights_test'] = False
    configs['early_stop'] = True
    configs['early_stop_start_iter'] = 100 # 60
    configs['wait_patience'] = 300
    configs['restore_best_weights'] = True
    configs['batch_training'] = False 
    configs['batch_size'] = 256
    configs['batch_shuffle'] = True
    configs['batch_shuffle_buffer'] = 1024         
    print('--- Dataset: {}'.format(configs['data_name']))
    random.seed(configs['seed'])
    np.random.seed(configs['seed'])
    tf.random.set_seed(configs['seed'])

    

    ''' Create network instances'''
    net_mean = UQ_Net_mean_TF2(configs, num_inputs, num_outputs)
    net_up = UQ_Net_std_TF2(configs, num_inputs, num_outputs, net='up')
    net_down = UQ_Net_std_TF2(configs, num_inputs, num_outputs, net='down')

    # ''' Initialize trainer and conduct training/optimizations '''
    trainer = CL_trainer(configs, net_mean, net_up, net_down, xTrain, yTrain, xValid=xValid, yValid=yValid, xTest=xTest, yTest=yTest, testDataEvaluationDuringTrain=False)
    trainer.train()                 # training for 3 networks
    trainer.boundaryOptimization(verbose=1)  # boundary optimization
    trainer.testDataPrediction()    # evaluation of the trained nets on testing data
    trainer.capsCalculation(final_evaluation=True, verbose=1)       # metrics calculation
    # trainer.saveResultsToTxt()      # save results to txt file

else:
    print('--- Running on auto mode.')

    configs['batch_training'] = False
    params_optimizer = CL_params_optimizer(configs, xTrain, yTrain, xValid, yValid, xTest, yTest)
    optim_configs = params_optimizer.auto_params_optimization(upper_limit_iters=1000, num_trials=50, rnd_state=2)


    print('-------------------------------------------------------------')
    print('--- Train model based on the optimized parameters:')
    print('-------------------------------------------------------------')

    ### apply the optimized parameters and run training one more time
    random.seed(optim_configs['seed'])
    np.random.seed(optim_configs['seed'])
    tf.random.set_seed(optim_configs['seed'])

    ''' Create network instances'''
    net_mean = UQ_Net_mean_TF2(optim_configs, num_inputs, num_outputs)
    net_up = UQ_Net_std_TF2(optim_configs, num_inputs, num_outputs, net='up')
    net_down = UQ_Net_std_TF2(optim_configs, num_inputs, num_outputs, net='down')

    # ''' Initialize trainer and conduct training/optimizations '''
    trainer = CL_trainer(optim_configs, net_mean, net_up, net_down, xTrain, yTrain, xValid=xValid, yValid=yValid, xTest=xTest, yTest=yTest, testDataEvaluationDuringTrain=True) # for final evaluation
    trainer.train()                 # training for 3 networks
    trainer.boundaryOptimization(verbose=1)  # boundary optimization
    trainer.testDataPrediction()    # evaluation of the trained nets on testing data
    trainer.capsCalculation(final_evaluation=True, verbose=1)       # metrics calculation
    # trainer.saveResultsToTxt()      # save results to txt file
    print('-------------------------------------------------------------')
    print('--- Optimized parameters: ')
    print('-------------------------------------------------------------')
    for key, value in optim_configs.items():
        print('- {}: {}'.format(key, value))





