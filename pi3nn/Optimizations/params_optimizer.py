
''' Hyper-parameters optimizations '''


### hyperopt package used for fast hypar-parameters tuning 

import random
import numpy as np
import pandas as pd
import tensorflow as tf

##### Hyperopt hyperparameter tuning test
from hyperopt import fmin, hp, Trials, STATUS_OK, tpe, rand
from pi3nn.Networks.networks import UQ_Net_mean_TF2, UQ_Net_std_TF2
from pi3nn.Trainers.trainers import CL_trainer
from pi3nn.Networks.networks_TS import LSTM_mean_TF2, LSTM_PI_TF2, CL_LSTM_train_steps
from pi3nn.Trainers.trainer_TS import CL_trainer_TS
from pi3nn.Utils.Utils import CL_Utils
utils = CL_Utils()

class CL_params_optimizer:
    def __init__(self, configs, xTrain, yTrain, xValid, yValid, xTest, yTest, upper_limit_iters=[500,20,20], num_trials=10, rstate=None):
        self.configs = configs
        self.target_PICP = self.configs['quantile']
        self.configs['restore_best_weights'] = True
        self.configs['batch_training'] = False
        if self.configs['batch_training']:
            self.configs['batch_shuffle'] = True
            self.configs['batch_shuffle_buffer'] = 1024
        self.configs['saveWeights'] = False
        self.configs['loadWeights_test'] = False
        self.num_trials = num_trials
        self.batch_size_options = [8, 16, 32, 64, 128, 256]

        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xValid = xValid
        self.yValid = yValid
        self.xTest = xTest
        self.yTest = yTest

        self.layer_options = self.configs['layer_options']
        # self.layer_options = [1, 2, 3] ### 1, 2, and 3 layers of network will be tested
        self.min_nodes = 1
        self.max_nodes = 128

        print('--- Auto hyper-parameters tuning...')



    def structure_params(self, num_layers, label):
        params = {}
        for n in range(num_layers):
            params['n_nodes_layer_{}_{}'.format(label, n)] = hp.randint('n_nodes_{}_{}_{}'.format(label, num_layers, n), self.min_nodes, self.max_nodes)
        return params
 

    def auto_params_optimization(self, upper_limit_iters=[500,20,20], num_trials=20, rnd_state=None):

        space = {
        'seed': hp.randint('seed', 0, 100),
        'Max_iter_mean': hp.randint('Max_iter_mean', 1, upper_limit_iters[0]),
        'Max_iter_up': hp.randint('Max_iter_up', 1, upper_limit_iters[1]), 
        'Max_iter_down': hp.randint('Max_iter_down', 1, upper_limit_iters[2]),  
        'lr_mean': hp.uniform('lr_mean', 0.0001, 0.01), 
        'lr_up': hp.uniform('lr_up', 0.0001, 0.01),
        'lr_down': hp.uniform('lr_down', 0.0001, 0.01),
        'optimizer': hp.choice('optimizer', ['Adam', 'SGD']),
        'exponential_decay': hp.choice('exponential_decay', [True, False]),
        'decay_steps': hp.randint('decay_steps', 1, upper_limit_iters[0]),
        'decay_rate': hp.uniform('decay_rate', 0.5, 0.999),
        # 'batch_training': hp.choice('batch_training', [True, False]),
        # 'batch_shuffle': hp.choice('batch_shuffle', [True, False]),
        # 'batch_size': hp.choice('batch_size', [8, 16, 32, 64, 128, 256]),
        'early_stop': hp.choice('early_stop', [True, False]),
        'early_stop_start_iter': hp.randint('early_stop_start_iter', 1, upper_limit_iters[0]),
        'wait_patience': hp.randint('wait_patience', 1, upper_limit_iters[0]),
        # 'Nnode': hp.choice('Nnode', [8, 16, 32, 64, 128]),  ## apply to all three networks, single hidden layer for now
        }

        ### add layers/nodes info to the 'space'
        space['layers_mean'] = hp.choice('layers_mean', [self.structure_params(n, 'mean') for n in self.layer_options])
        space['layers_up'] = hp.choice('layers_up', [self.structure_params(n, 'up') for n in self.layer_options])
        space['layers_down'] = hp.choice('layers_down', [self.structure_params(n, 'down') for n in self.layer_options])

        if self.configs['batch_training']:
            # space['batch_shuffle'] = hp.choice('batch_shuffle', [True, False])
            space['batch_size'] = hp.choice('batch_size', self.batch_size_options)


        trials = Trials()
        if rnd_state is None:
            rstate = np.random.RandomState()
        if rnd_state is not None:
            rstate = np.random.RandomState(rnd_state)
        optim_params = fmin(self.params_optim_target_function, space, algo=tpe.suggest, max_evals=num_trials, trials=trials, rstate=rstate)

        # print('--- Optimized parameters: ')
        # for key, value in optim_params.items():
        #   if key == 'Nnode':
        #       print('- {}: {}'.format(key, [8, 16, 32, 64, 128][value]))
        #   elif key == 'early_stop':
        #       print('- {}: {}'.format(key, [True, False][value]))
        #   elif key == 'exponential_decay':
        #       print('- {}: {}'.format(key, [True, False][value]))
        #   elif key == 'optimizer':
        #       print('- {}: {}'.format(key, ['Adam', 'SGD'][value]))
        #   else:
        #       print('- {}: {}'.format(key, value))

        ## assign the optimized params
        self.configs['seed'] = optim_params['seed']

        self.configs['Max_iter'] = []
        self.configs['Max_iter'].append(optim_params['Max_iter_mean'])
        self.configs['Max_iter'].append(optim_params['Max_iter_up'])
        self.configs['Max_iter'].append(optim_params['Max_iter_down'])

        layers_mean = optim_params['layers_mean'] + 1
        layers_up = optim_params['layers_up'] + 1
        layers_down = optim_params['layers_down'] + 1

        optim_mean_layers = []
        for i in range(layers_mean):
            optim_mean_layers.append(optim_params['n_nodes_mean_{}_{}'.format(layers_mean, i)])

        optim_up_layers = []
        for i in range(layers_up):
            optim_up_layers.append(optim_params['n_nodes_up_{}_{}'.format(layers_up, i)])

        optim_down_layers = []
        for i in range(layers_down):
            optim_down_layers.append(optim_params['n_nodes_down_{}_{}'.format(layers_down, i)])

        self.configs['num_neurons_mean'] = optim_mean_layers
        self.configs['num_neurons_up'] = optim_up_layers
        self.configs['num_neurons_down'] = optim_down_layers

        self.configs['lr'][0] = optim_params['lr_mean']
        self.configs['lr'][1] = optim_params['lr_up']
        self.configs['lr'][2] = optim_params['lr_down']
        self.configs['optimizers'][0] = ['Adam', 'SGD'][optim_params['optimizer']]
        self.configs['optimizers'][1] = ['Adam', 'SGD'][optim_params['optimizer']]
        self.configs['optimizers'][2] = ['Adam', 'SGD'][optim_params['optimizer']]
        self.configs['exponential_decay'] = [True, False][optim_params['exponential_decay']]
        self.configs['decay_steps'] = optim_params['decay_steps']
        self.configs['decay_rate'] = optim_params['decay_rate']
        self.configs['early_stop'] = [True, False][optim_params['early_stop']]
        self.configs['early_stop_start_iter'] = optim_params['early_stop_start_iter']
        self.configs['wait_patience'] = optim_params['wait_patience']

        if self.configs['batch_training']:
            # self.configs['batch_shuffle'] = optim_params['batch_shuffle']
            self.configs['batch_size'] = self.batch_size_options[optim_params['batch_size']]

        # self.configs['num_neurons_mean_net'] = [8, 16, 32, 64, 128][optim_params['Nnode']]     
        # self.configs['num_neurons_up_down_net'] = [8, 16, 32, 64, 128][optim_params['Nnode']]
        return self.configs


    def params_optim_target_function(self, params):

        self.configs['seed'] = params['seed']
        self.configs['Max_iter'] = params['Max_iter']

        # params['layers_mean'] --> {'n_nodes_layer_mean_0': 116, 'n_nodes_layer_mean_1': 41} 

        self.configs['num_neurons_mean'] = [params['layers_mean'][nodes] for nodes in params['layers_mean']]
        self.configs['num_neurons_up'] = [params['layers_up'][nodes] for nodes in params['layers_up']]
        self.configs['num_neurons_down'] = [params['layers_down'][nodes] for nodes in params['layers_down']]

        # print('--- MEAN:')
        # print(params['layers_mean'])
        # for key, nodes in enumerate(params['layers_mean']):
        #   print('--layer: {} nodes: {}'.format(key, params['layers_mean'][nodes]))

        # print([params['layers_mean'][nodes] for nodes in params['layers_mean']])


        # print('--- UP:')
        # print(params['layers_up'])
        # for key, nodes in enumerate(params['layers_up']):
        #   print('--layer: {} nodes: {}'.format(key, params['layers_up'][nodes]))

        # print([params['layers_up'][nodes] for nodes in params['layers_up']])

        # print('--- DOWN:')
        # print(params['layers_down'])
        # for key, nodes in enumerate(params['layers_down']):
        #   print('--layer: {} nodes: {}'.format(key, params['layers_down'][nodes]))

        # # print([params['layers_down'][nodes] for nodes in params['layers_down']])
        # # # exit()

        self.configs['lr'] = []
        self.configs['lr'].append(params['lr_mean'])
        self.configs['lr'].append(params['lr_up'])
        self.configs['lr'].append(params['lr_down'])
        self.configs['optimizers'] = []
        self.configs['optimizers'].append(params['optimizer'])
        self.configs['optimizers'].append(params['optimizer'])
        self.configs['optimizers'].append(params['optimizer'])
        self.configs['exponential_decay'] = params['exponential_decay']
        self.configs['decay_steps'] = params['decay_steps']
        self.configs['decay_rate'] = params['decay_rate']
        self.configs['early_stop'] = params['early_stop']
        self.configs['early_stop_start_iter'] = params['early_stop_start_iter']
        self.configs['wait_patience'] = params['wait_patience']

        if self.configs['batch_training']:
            self.configs['batch_size'] = params['batch_size']
            # self.configs['batch_shuffle'] = params['batch_shuffle']
            # self.configs['batch_shuffle_buffer'] = params['batch_shuffle_buffer']

        random.seed(self.configs['seed'])
        np.random.seed(self.configs['seed'])
        tf.random.set_seed(self.configs['seed'])

        num_inputs = utils.getNumInputsOutputs(self.xTrain)
        num_outputs = utils.getNumInputsOutputs(self.yTrain)

        ''' Create network instances'''
        net_mean = UQ_Net_mean_TF2(self.configs, num_inputs, num_outputs)
        net_up = UQ_Net_std_TF2(self.configs, num_inputs, num_outputs, net='up')
        net_down = UQ_Net_std_TF2(self.configs, num_inputs, num_outputs, net='down')

        # ''' Initialize trainer and conduct training/optimizations '''
        trainer = CL_trainer(self.configs, net_mean, net_up, net_down, self.xTrain, self.yTrain, xValid=self.xValid, yValid=self.yValid, xTest=self.xTest, yTest=self.yTest, testDataEvaluationDuringTrain=False)
        trainer.train()        # training for 3 networks
        trainer.boundaryOptimization()  # boundary optimization
        trainer.testDataPrediction()    # evaluation of the trained nets on testing data
        PICP_train, PICP_valid, PICP_test, MPIW_train, MPIW_valid, MPIW_test, R2_test = trainer.capsCalculation(final_evaluation=True)       # metrics calculation

        # customized_loss = np.abs(PICP_valid - self.target_PICP)
        # customized_loss = np.abs(PICP_test - self.target_PICP)
        customized_loss = np.abs(1 - R2_test)
        # customized_loss = np.abs(PICP_valid - self.target_PICP) * 0.1* np.abs(MPIW_valid)

        return {'loss': customized_loss, 'status': STATUS_OK}



class CL_params_optimizer_LSTM:
    def __init__(self, configs, xTrain, yTrain, xValid, yValid, xTest, yTest, upper_limit_iters=[500,20,20], num_trials=10, rstate=None, scalerx=None, scalery=None):
        self.configs = configs
        self.target_PICP = self.configs['quantile']
        self.configs['restore_best_weights'] = True
        self.configs['batch_training'] = True 
        self.configs['saveWeights'] = False
        self.configs['loadWeights_test'] = False
        self.num_trials = num_trials
        self.scalerx = scalerx
        self.scalery = scalery
        self.batch_size_options = [8, 16, 32, 64, 128, 256]

        if self.configs['batch_training']:
            self.configs['batch_shuffle'] = True
            self.configs['batch_shuffle_buffer'] = 1024

        self.configs['save_LSTM_model'] = False
        # self.configs['load_LSTM_model'] = False

        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xValid = xValid
        self.yValid = yValid
        self.xTest = xTest
        self.yTest = yTest

        self.layer_options = self.configs['layer_options']
        # self.layer_options = [1, 2, 3] ### 1, 2, and 3 layers of network will be tested
        self.layer_options_updown = self.configs['layer_options_updown']
        self.min_nodes = 1
        self.max_nodes = 256

        ### record the Final R2s for all trials
        self.final_r2s = [] ## list of list [ [trainR2, validR2, testR2],[...],... ]
        ### record train/valid/test loss
        self.all_losses = [] ## list of list [ [train_loss, valid_loss, test_loss], [...], ...]

        print('--- Auto hyper-parameters tuning for PI3NN-LSTM...')

    def structure_params(self, num_layers, label):
        params = {}
        for n in range(num_layers):
            params['n_nodes_layer_{}_{}'.format(label, n)] = hp.randint('n_nodes_{}_{}_{}'.format(label, num_layers, n), self.min_nodes, self.max_nodes)
        return params

    def auto_params_optimization_LSTM(self, upper_limit_iters=[500,20,20], num_trials=20, rnd_state=None):

        space = {
        'seed': hp.randint('seed', 0, 100),
        'LSTM_nodes': hp.randint('LSTM_nodes', 1, 256),
        # 'fc1_nodes': hp.randint('fc1_nodes', 1, 256),
        'Max_lstm_iter_mean': hp.randint('Max_lstm_iter_mean', 10, upper_limit_iters[0]),
        'Max_lstm_iter_up': hp.randint('Max_lstm_iter_up', 10, upper_limit_iters[1]),
        'Max_lstm_iter_down': hp.randint('Max_lstm_iter_down', 10, upper_limit_iters[2]), 
        'lr_lstm_mean': hp.uniform('lr_lstm_mean', 0.0001, 0.01),
        'lr_lstm_up': hp.uniform('lr_lstm_up', 0.0001, 0.01), 
        'lr_lstm_down': hp.uniform('lr_lstm_down', 0.0001, 0.01), 
        'optimizer': hp.choice('optimizer', ['Adam', 'SGD']),
        'exponential_decay': hp.choice('exponential_decay', [True, False]),
        'decay_steps': hp.randint('decay_steps', 1, upper_limit_iters[0]),
        'decay_rate': hp.uniform('decay_rate', 0.5, 0.999),
        # 'batch_training': hp.choice('batch_training', [True, False]),
        # 'batch_shuffle': hp.choice('batch_shuffle', [True, False]),
        # 'batch_size': hp.choice('batch_size', [8, 16, 32, 64, 128, 256]),
        'early_stop': hp.choice('early_stop', [True, False]),
        'early_stop_start_iter': hp.randint('early_stop_start_iter', 1, upper_limit_iters[0]),
        'wait_patience': hp.randint('wait_patience', 1, upper_limit_iters[0]),
        }

        ### add layers/nodes info to the 'space'
        space['layers_mean'] = hp.choice('layers_mean', [self.structure_params(n, 'mean') for n in self.layer_options])
        space['layers_up'] = hp.choice('layers_up', [self.structure_params(n, 'up') for n in self.layer_options])
        space['layers_down'] = hp.choice('layers_down', [self.structure_params(n, 'down') for n in self.layer_options])

        if self.configs['batch_training']:
            space['batch_size'] = hp.choice('batch_size', self.batch_size_options)

        trials = Trials()
        if rnd_state is None:
            rstate = np.random.RandomState()
        if rnd_state is not None:
            rstate = np.random.RandomState(rnd_state)

        ## run optimization
        optim_params = fmin(self.params_optim_target_function_LSTM, space, algo=tpe.suggest, max_evals=num_trials, trials=trials, rstate=rstate)
        
        ## assign the optimized params
        self.configs['seed'] = optim_params['seed']
        self.configs['LSTM_nodes'] = optim_params['LSTM_nodes']

        self.configs['Max_lstm_iter'] = []
        self.configs['Max_lstm_iter'].append(optim_params['Max_lstm_iter_mean'])
        self.configs['Max_lstm_iter'].append(optim_params['Max_lstm_iter_up'])
        self.configs['Max_lstm_iter'].append(optim_params['Max_lstm_iter_down'])

        layers_mean = optim_params['layers_mean'] + 1
        layers_up = optim_params['layers_up'] + 1
        layers_down = optim_params['layers_down'] + 1

        optim_mean_layers = []
        for i in range(layers_mean):
            optim_mean_layers.append(optim_params['n_nodes_mean_{}_{}'.format(layers_mean, i)])

        optim_up_layers = []
        for i in range(layers_up):
            optim_up_layers.append(optim_params['n_nodes_up_{}_{}'.format(layers_up, i)])

        optim_down_layers = []
        for i in range(layers_down):
            optim_down_layers.append(optim_params['n_nodes_down_{}_{}'.format(layers_down, i)])

        self.configs['num_neurons_mean'] = optim_mean_layers
        self.configs['num_neurons_up'] = optim_up_layers
        self.configs['num_neurons_down'] = optim_down_layers

        self.configs['lr_lstm'] = []
        self.configs['lr_lstm'].append(optim_params['lr_lstm_mean'])
        self.configs['lr_lstm'].append(optim_params['lr_lstm_up'])
        self.configs['lr_lstm'].append(optim_params['lr_lstm_down'])

        self.configs['optimizers_lstm'] = []
        self.configs['optimizers_lstm'].append(['Adam', 'SGD'][optim_params['optimizer']])
        self.configs['optimizers_lstm'].append(['Adam', 'SGD'][optim_params['optimizer']])
        self.configs['optimizers_lstm'].append(['Adam', 'SGD'][optim_params['optimizer']])
        self.configs['exponential_decay'] = [True, False][optim_params['exponential_decay']]
        self.configs['decay_steps'] = optim_params['decay_steps']
        self.configs['decay_rate'] = optim_params['decay_rate']
        self.configs['early_stop'] = [True, False][optim_params['early_stop']]
        self.configs['early_stop_start_iter'] = optim_params['early_stop_start_iter']
        self.configs['wait_patience'] = optim_params['wait_patience']

        if self.configs['batch_training']:
            self.configs['batch_size'] = self.batch_size_options[optim_params['batch_size']]

        optimized_configs = self.configs.copy()
        ### Convert the dtype from numpy.float64, numpy.int64 to python float, int
        ### after using Hyperopt package
        optimized_configs = utils.convertDtype(optimized_configs)
        return optimized_configs, trials

        # return self.configs  ## optimized configs

    def params_optim_target_function_LSTM(self, params):

        self.configs['seed'] = params['seed']
        self.configs['Max_lstm_iter'] = []
        self.configs['Max_lstm_iter'].append(params['Max_lstm_iter_mean'])
        self.configs['Max_lstm_iter'].append(params['Max_lstm_iter_up'])
        self.configs['Max_lstm_iter'].append(params['Max_lstm_iter_down'])

        self.configs['LSTM_nodes'] = params['LSTM_nodes']
        self.configs['num_neurons_mean'] = [params['layers_mean'][nodes] for nodes in params['layers_mean']]
        self.configs['num_neurons_up'] = [params['layers_up'][nodes] for nodes in params['layers_up']]
        self.configs['num_neurons_down'] = [params['layers_down'][nodes] for nodes in params['layers_down']]

        self.configs['lr_lstm'] = []
        self.configs['lr_lstm'].append(params['lr_lstm_mean'])
        self.configs['lr_lstm'].append(params['lr_lstm_up'])
        self.configs['lr_lstm'].append(params['lr_lstm_down'])

        self.configs['optimizers_lstm'] = []
        self.configs['optimizers_lstm'].append(params['optimizer'])
        self.configs['optimizers_lstm'].append(params['optimizer'])
        self.configs['optimizers_lstm'].append(params['optimizer'])

        self.configs['exponential_decay'] = params['exponential_decay']
        self.configs['decay_steps'] = params['decay_steps']
        self.configs['decay_rate'] = params['decay_rate']
        self.configs['early_stop'] = params['early_stop']
        self.configs['early_stop_start_iter'] = params['early_stop_start_iter']
        self.configs['wait_patience'] = params['wait_patience']
        
        if self.configs['batch_training']:
            self.configs['batch_size'] = params['batch_size']
            # print('---batchsize: {}'.format(self.configs['batch_size']))
            # self.configs['batch_shuffle'] = params['batch_shuffle']
            # self.configs['batch_shuffle_buffer'] = params['batch_shuffle_buffer']

        # if self.configs['batch_training']:
        #   self.configs['batch_size'] = params['batch_size']

        random.seed(self.configs['seed'])
        np.random.seed(self.configs['seed'])
        tf.random.set_seed(self.configs['seed'])

        num_inputs = utils.getNumInputsOutputs(self.xTrain)
        num_outputs = utils.getNumInputsOutputs(self.yTrain)

        #### RUN LSTM
        net_lstm_mean = LSTM_mean_TF2(self.configs, num_outputs)
        net_lstm_up = LSTM_PI_TF2(self.configs, num_outputs, net='up')
        net_lstm_down = LSTM_PI_TF2(self.configs, num_outputs, net='down')

        trainer_lstm = CL_trainer_TS(self.configs, net_lstm_mean, net_lstm_up, net_lstm_down, self.xTrain, self.yTrain, xValid=self.xValid, yValid=self.yValid, xTest=self.xTest, yTest=self.yTest, testDataEvaluationDuringTrain=True)
        trainer_lstm.train_LSTM()
        trainer_lstm.boundaryOptimization(verbose=1)
        trainer_lstm.testDataPrediction()    # evaluation of the trained nets on testing data
        r2_train, r2_valid, r2_test = trainer_lstm.modelEvaluation(self.scalerx, self.scalery, save_results=False)
        customized_loss = np.abs(1 - r2_test)
        print('*** Printing TMP results ***')
        for key, value in self.configs.items():
            print('- {}: {}'.format(key, value))
        print('***************************************************************************')
        print('--- Current R2, train: {:.4f}, valid: {:.4f}, test: {:.4f}'.format(r2_train, r2_valid, r2_test))
        print('***************************************************************************')

        self.final_r2s.append([r2_train, r2_valid, r2_test])
        self.all_losses.append([trainer_lstm.train_loss_mean_list, trainer_lstm.valid_loss_mean_list, trainer_lstm.test_loss_mean_list])
        return {'loss': customized_loss, 'status': STATUS_OK}

    def auto_params_optimization_LSTM_updown(self, upper_limit_iters=[100,100], num_trials=20, rnd_state=None):
        space = {
        # 'seed': hp.randint('seed', 0, 100),
        # 'LSTM_nodes': hp.randint('LSTM_nodes', 1, 256),
        # 'fc1_nodes': hp.randint('fc1_nodes', 1, 256),
        # 'Max_lstm_iter_mean': hp.randint('Max_lstm_iter_mean', 10, upper_limit_iters[0]),
        'Max_lstm_iter_up': hp.randint('Max_lstm_iter_up', 10, upper_limit_iters[0]),
        'Max_lstm_iter_down': hp.randint('Max_lstm_iter_down', 10, upper_limit_iters[1]), 
        # 'lr_lstm_mean': hp.uniform('lr_lstm_mean', 0.0001, 0.01),
        'lr_lstm_up': hp.uniform('lr_lstm_up', 0.0001, 0.01), 
        'lr_lstm_down': hp.uniform('lr_lstm_down', 0.0001, 0.01), 
        # 'optimizer': hp.choice('optimizer', ['Adam', 'SGD']),
        # 'exponential_decay': hp.choice('exponential_decay', [True, False]),
        # 'decay_steps': hp.randint('decay_steps', 1, upper_limit_iters[0]),
        # 'decay_rate': hp.uniform('decay_rate', 0.5, 0.999),
        # 'batch_training': hp.choice('batch_training', [True, False]),
        # 'batch_shuffle': hp.choice('batch_shuffle', [True, False]),
        # 'batch_size': hp.choice('batch_size', [8, 16, 32, 64, 128, 256]),
        # 'early_stop': hp.choice('early_stop', [True, False]),
        # 'early_stop_start_iter': hp.randint('early_stop_start_iter', 1, upper_limit_iters[0]),
        # 'wait_patience': hp.randint('wait_patience', 1, upper_limit_iters[0]),
        }

        ### add layers/nodes info to the 'space'
        space['layers_up'] = hp.choice('layers_up', [self.structure_params(n, 'up') for n in self.layer_options_updown])
        space['layers_down'] = hp.choice('layers_down', [self.structure_params(n, 'down') for n in self.layer_options_updown])

        trials = Trials()
        if rnd_state is None:
            rstate = np.random.RandomState()
        if rnd_state is not None:
            rstate = np.random.RandomState(rnd_state)

        ## run optimization
        optim_params = fmin(self.params_optim_target_function_LSTM_updown, space, algo=tpe.suggest, max_evals=num_trials, trials=trials, rstate=rstate)
        ## assign the optimized params
        self.configs['Max_lstm_iter'][1] = optim_params['Max_lstm_iter_up']
        self.configs['Max_lstm_iter'][2] = optim_params['Max_lstm_iter_down']

        layers_up = optim_params['layers_up'] + 1
        layers_down = optim_params['layers_down'] + 1
        optim_up_layers = []
        for i in range(layers_up):
            optim_up_layers.append(optim_params['n_nodes_up_{}_{}'.format(layers_up, i)])
        optim_down_layers = []
        for i in range(layers_down):
            optim_down_layers.append(optim_params['n_nodes_down_{}_{}'.format(layers_down, i)])

        self.configs['num_neurons_up'] = optim_up_layers
        self.configs['num_neurons_down'] = optim_down_layers
        self.configs['lr_lstm'][1] = optim_params['lr_lstm_up']
        self.configs['lr_lstm'][2] = optim_params['lr_lstm_down']

        optimized_configs = self.configs.copy()
        optimized_configs = utils.convertDtype(optimized_configs)
        return optimized_configs, trials


    def params_optim_target_function_LSTM_updown(self, params):

        self.configs['Max_lstm_iter'][1] = params['Max_lstm_iter_up']
        self.configs['Max_lstm_iter'][2] = params['Max_lstm_iter_down']

        self.configs['lr_lstm'][1] = params['lr_lstm_up']
        self.configs['lr_lstm'][2] = params['lr_lstm_down']

        self.configs['num_neurons_up'] = [params['layers_up'][nodes] for nodes in params['layers_up']]
        self.configs['num_neurons_down'] = [params['layers_down'][nodes] for nodes in params['layers_down']]

        random.seed(self.configs['seed'])
        np.random.seed(self.configs['seed'])
        tf.random.set_seed(self.configs['seed'])

        num_inputs = utils.getNumInputsOutputs(self.xTrain)
        num_outputs = utils.getNumInputsOutputs(self.yTrain)

        random.seed(self.configs['seed'])
        np.random.seed(self.configs['seed'])
        tf.random.set_seed(self.configs['seed'])

        #### RUN LSTM for only UP/DOWN networks

        ### Test LSTM   --- original one-time run
        net_lstm_mean = LSTM_mean_TF2(self.configs, num_outputs)
        net_lstm_up = LSTM_PI_TF2(self.configs, num_outputs, net='up')
        net_lstm_down = LSTM_PI_TF2(self.configs, num_outputs, net='down')

        trainer_lstm = CL_trainer_TS(self.configs, net_lstm_mean, net_lstm_up, net_lstm_down, self.xTrain, self.yTrain, xValid=self.xValid, yValid=self.yValid, xTest=self.xTest, yTest=self.yTest, \
            testDataEvaluationDuringTrain=False, allTestDataEvaluationDuringTrain=False)
        trainer_lstm.train_LSTM()
        trainer_lstm.boundaryOptimization(verbose=0)
        trainer_lstm.testDataPrediction()    # evaluation of the trained nets on testing data
        r2_train, r2_valid, r2_test, train_results_np, valid_results_np, test_results_np \
        = trainer_lstm.modelEvaluation(self.scalerx, self.scalery, save_results=False, return_results=True)

        ### evaluate the customized loss
        # train_results_np, valid_results_np, test_results_np have been converted back from log process (if have)
        train_in = 0
        train_in_obs = 0
        test_in = 0
        test_in_obs = 0
        for i in range(len(train_results_np)):
            if (train_results_np[i, 1] >= train_results_np[i, 3]) and (train_results_np[i, 1] <= train_results_np[i, 2]):
                train_in+=1
            if (train_results_np[i, 0] >= train_results_np[i, 3]) and (train_results_np[i, 0] <= train_results_np[i, 2]):
                train_in_obs+=1

        for i in range(len(test_results_np)):
            if (test_results_np[i, 1] >= test_results_np[i, 3]) and (test_results_np[i, 1] <= test_results_np[i, 2]):
                test_in+=1
            if (test_results_np[i, 0] >= test_results_np[i, 3]) and (test_results_np[i, 0] <= test_results_np[i, 2]):
                test_in_obs+=1

        frac_in_train = train_in_obs/len(train_results_np)
        frac_in_test = test_in_obs/len(test_results_np)
        customized_loss = np.abs(self.configs['quantile']-frac_in_test)

        for key, value in self.configs.items():
            print('- {}: {}'.format(key, value))
        print('***************************************************************************')
        print('--- Current R2, train: {:.4f}, valid: {:.4f}, test: {:.4f}'.format(r2_train, r2_valid, r2_test))
        print('--- Current data coverage, train: {:.2f}%, test: {:.2f}%'.format(frac_in_train*100, frac_in_test*100))
        print('***************************************************************************')

        return {'loss': customized_loss, 'status': STATUS_OK}