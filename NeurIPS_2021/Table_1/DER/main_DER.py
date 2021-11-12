import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from scipy import stats
from sklearn.metrics import r2_score
import math

# import edl
import evidential_deep_learning as edl
import data_loader
import trainers
import models
from models.toy.h_params import h_params
import itertools
tf.config.threading.set_intra_op_parallelism_threads(1)
import random

''' 
This is the code for evidential regression (DER) method used for our comparison which was adjusted from the original DER code 
(also included in the upper level folder) or you can find it at https://github.com/aamini/evidential-deep-learning

All of our experiments on DER method with UCI datasets were conducted on a single Ubuntu workstation, and we use Intel I9-10980xe CPU generated 
all the results instead of using a GPU because the training are relatively fast on CPU.

To reproduce the results, simply assign the data set name to the "data_name" variable at the beginning of this code before running the main file.
Accepted data names are: 'boston', 'concrete', 'energy', 'kin8nm', 'naval', 'power', 'protein', 'wine', 'yacht', 'MSD'
The results will be generated in the ./Results_DER/ including the summary of the training results (.txt files), plotted loss curves (./Results_DER/loss_curves/)
and loss history for each case (.csv format in ./Results_DER/loss_history/)

We also prepared pre-generated results for your reference (in ./Pre_generated_results/)

The results for DER method from our Table 1 can be obtained by running this code or using our pre-generated results.

Have fun!
'''

# dataset_list = ['boston', 'concrete', 'energy', 'kin8nm', 'naval', 'power', 'protein', 'wine', 'yacht', 'MSD']

data_name = 'boston'
original_data_path = '../../UCI_datasets/'          ## original UCI data sets
splitted_data_path = '../../UCI_TrainTest_Split/'   ## pre-split data
results_path = './Results_DER/'+data_name + '_DER_results.txt'

save_loss_history = True
save_loss_history_path = './Results_DER/loss_history/'
plot_loss_history = True
plot_loss_history_path = './Results_DER/loss_curves/'

parser = argparse.ArgumentParser()
parser.add_argument("--num-trials", default=20, type=int,
                    help="Number of trials to repreat training for \
                    statistically significant results.")
parser.add_argument("--num-epochs", default=40, type=int)
parser.add_argument('--datasets', nargs='+', default=["yacht"],
                    choices=['boston', 'concrete', 'energy',
                            'kin8nm', 'naval', 'power', 'protein',
                            'wine', 'yacht', 'MSD'])

split_seed_list  = [1, 2, 3, 4, 5]
random_seed_list = [10, 20, 30, 40, 50]
seed_combination_list = list(itertools.product(split_seed_list, random_seed_list))
print('-- The splitting and random seed combination list: {}'.format(seed_combination_list))

dataset = data_name
learning_rate = h_params[dataset]["learning_rate"]
batch_size = h_params[dataset]["batch_size"]

with open(results_path, 'a') as fwrite:
    fwrite.write('EXP '+'split_seed '+'random_seed '+'Mean_PICP_test '+'Mean_MPIW_test '+'Mean_RMSE '+'Mean_NLL '+'Mean_R2 '+'num_trials '+'data '+'lr '+'batch_size '+'\n')

### used for train specific case with singe or a series of index (0-24 for random seed combinations)
iii_low = 0    #
iii_high = 24
for iii in range(len(seed_combination_list)):
    if iii >= iii_low and iii <=iii_high:

        split_seed = seed_combination_list[iii][0]
        seed = seed_combination_list[iii][1]
        print('--- Running EXP {}/{}'.format(iii + 1, len(seed_combination_list)))
        print('--- Dataset: {}'.format(data_name))
        print('--- Splitting seed and random seed: {}, {}'.format(split_seed, seed))
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        args = parser.parse_args()
        args.datasets[0] = data_name
        training_schemes = [trainers.Evidential]
        datasets = args.datasets
        # print('--- Printing datasets:')
        # print(datasets)
        num_trials = args.num_trials
        print('num_trials:{}'.format(num_trials))
        # num_trials = 3
        num_epochs = args.num_epochs
        dev = "/cpu:0" # for small datasets/models cpu is faster than gpu
        """" ================================================"""

        RMSE = np.zeros((len(datasets), len(training_schemes), num_trials))
        NLL = np.zeros((len(datasets), len(training_schemes), num_trials))

        PICP_arr = np.zeros(num_trials)
        MPIW_arr = np.zeros(num_trials)
        R2_arr = np.zeros(num_trials)

        for di, dataset in enumerate(datasets):
            # print(di)
            # print(dataset)
            for ti, trainer_obj in enumerate(training_schemes):
                for n in range(num_trials):
                    print('*********************************************')
                    print('--- data: {}, trial: {}, iii range:{}, {} '.format(data_name, n+1, iii_low, iii_high))
                    print('*********************************************')

                    # (x_train, y_train), (x_test, y_test), y_scale = data_loader.load_dataset(dataset, return_as_tensor=False)
                    ''' Load the customized data with fixed splitting train/test '''
                    x_train, y_train, x_test, y_test, y_scale = data_loader.LoadData_Splitted_UCI(data_name, original_data_path, splitted_data_path, split_seed)
                    batch_size = h_params[dataset]["batch_size"]
                    num_iterations = num_epochs * x_train.shape[0]//batch_size
                    done = False
                    while not done:
                        with tf.device(dev):
                            model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
                            model, opts = model_generator.create(input_shape=x_train.shape[1:], tf_seed=seed)

                            trainer = trainer_obj(model, opts, dataset, learning_rate=h_params[dataset]["learning_rate"])
                            model, rmse, nll, loss = trainer.train(x_train, y_train, x_test, y_test, y_scale, batch_size=batch_size, iters=num_iterations,
                                                             verbose=True, data_name=data_name, split_seed=split_seed, rnd_seed=seed, trial_num=n,
                                                                   bool_plot_loss=True, bool_save_loss=True,
                                                                   save_loss_path=save_loss_history_path,
                                                                   plot_loss_path=plot_loss_history_path)

                            ''' Evaluate the PICP and MPIW for each trial '''
                            ### taken from the 'plot_ng' function from the original evidential regression code
                            x_test_input_tf = tf.convert_to_tensor(x_test, tf.float32)
                            outputs = model(x_test_input_tf)

                            mu, v, alpha, beta = tf.split(outputs, 4, axis=1)
                            epistemic_var = np.sqrt(beta / (v * (alpha - 1)))
                            epistemic_var = np.minimum(epistemic_var, 1e3)

                            y_pred_U = mu.numpy() + epistemic_var * 1.96
                            y_pred_L = mu.numpy() - epistemic_var * 1.96
                            # print('y_pred_U: {}'.format(y_pred_U))
                            # print('y_pred_L: {}'.format(y_pred_L))
                            if np.isnan(y_pred_U).any() or np.isnan(y_pred_L).any():
                                PICP = math.nan
                                MPIW = math.nan
                                R2 = math.nan
                                rmse = math.nan
                                nll = math.nan
                                print('--- the y_pred_U/L contains NaN(s) in current trial')
                            else:
                                y_U_cap = y_pred_U > y_test
                                y_L_cap = y_pred_L < y_test
                                # print('y_U_cap: {}'.format(y_U_cap))
                                # print('y_L_cap: {}'.format(y_L_cap))
                                # print('y_L_cap: {}'.format(y_L_cap))
                                y_all_cap = y_U_cap * y_L_cap
                                PICP = np.sum(y_all_cap) / y_L_cap.shape[0]
                                MPIW = np.mean(y_pred_U - y_pred_L)
                                R2 = r2_score(y_test, mu.numpy())
                                print('PICP: {}, MPIW: {}, R2: {}'.format(PICP, MPIW, R2))

                            del model
                            tf.keras.backend.clear_session()
                            done = False if np.isinf(nll) or np.isnan(nll) else True

                            ### new added done criteria
                            if np.isnan(loss):
                                done = True
                    print("saving {} {}".format(rmse, nll))
                    RMSE[di, ti, n] = rmse
                    NLL[di, ti, n] = nll
                    PICP_arr[n] = PICP
                    MPIW_arr[n] = MPIW
                    R2_arr[n] = R2


        print('PICP_arr: {}'.format(PICP_arr))
        print('MPIW_arr: {}'.format(MPIW_arr))
        print('R2_arr: {}'.format(R2_arr))

        PICP_mean = np.nanmean(PICP_arr)
        MPIW_mean = np.nanmean(MPIW_arr)
        RMSE_mean = np.nanmean(RMSE)
        NLL_mean = np.nanmean(NLL)
        R2_mean = np.nanmean(R2_arr)

        print('--- Mean PICP: {}'.format(PICP_mean))
        print('--- Mean MPIW: {}'.format(MPIW_mean))
        print('--- Mean RMSE: {}'.format(RMSE_mean))
        print('--- Mean NLL: {}'.format(NLL_mean))
        print('--- Mean R2: {}'.format(R2_mean))

        RESULTS = np.hstack((RMSE, NLL))
        print('RESULTS: {}'.format(RESULTS))

        mu = RESULTS.mean(axis=-1)
        error = np.std(RESULTS, axis=-1)

        print("==========================")
        print("[{}]: {} pm {}".format(dataset, mu, error))
        print("==========================")

        print("TRAINERS: {}\nDATASETS: {}".format([trainer.__name__ for trainer in training_schemes], datasets))
        print("MEAN: \n{}".format(mu))
        print("ERROR: \n{}".format(error))

        ''' Save results to dat file '''
        with open(results_path, 'a') as fwrite:
            fwrite.write(str(iii + 1) + ' ' + str(split_seed) + ' ' + str(seed) + ' ' + str(round(PICP_mean, 3)) + ' ' + str(round(MPIW_mean, 3)) + ' '+ \
                         str(round(RMSE_mean, 3)) + ' ' + str(round(NLL_mean, 3))+ ' ' +str(round(R2_mean, 3)) + ' ' + str(num_trials) + ' ' + data_name + ' ' + \
                         str(learning_rate) + ' ' + str(batch_size) +'\n')

    # import pdb; pdb.set_trace()


# h_params = {
#     'yacht': {'learning_rate': 5e-4, 'batch_size': 1},
#     'naval': {'learning_rate': 5e-4, 'batch_size': 1},
#     'concrete': {'learning_rate': 5e-3, 'batch_size': 1},
#     'energy': {'learning_rate': 2e-3, 'batch_size': 1},
#     'kin8nm': {'learning_rate': 1e-3, 'batch_size': 1},
#     'power': {'learning_rate': 1e-3, 'batch_size': 2},
#     'boston': {'learning_rate': 1e-3, 'batch_size': 8},
#     'wine': {'learning_rate': 1e-4, 'batch_size': 32},
#     'protein': {'learning_rate': 1e-3, 'batch_size': 64},
#     'MSD': {'learning_rate': 1e-3, 'batch_size': 256} # we added this dataset
# }