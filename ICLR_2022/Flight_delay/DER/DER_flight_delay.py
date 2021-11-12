import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from scipy import stats
from sklearn.metrics import r2_score
import math
# Force using CPU globally by hiding GPU(s)
tf.config.set_visible_devices([], 'GPU')

# import edl
import evidential_deep_learning as edl
import data_loader
import trainers
import models
from models.toy.h_params import h_params
import itertools
tf.config.threading.set_intra_op_parallelism_threads(1)
import random



data_name = 'flight_delay'
original_data_path = '../flight_delay_data/'         
results_path = './Results_DER/'+data_name + '_DER_results.txt'



save_loss_history = False
save_loss_history_path = './Results_DER/loss_history/'
plot_loss_history = False
plot_loss_history_path = './Results_DER/loss_curves/'

parser = argparse.ArgumentParser()
parser.add_argument("--num-trials", default=1, type=int,
                    help="Number of trials to repreat training for \
                    statistically significant results.")
parser.add_argument("--num-epochs", default=100, type=int)
parser.add_argument('--datasets', nargs='+', default=["flight_delay"],
                    choices=['flight_delay'])


dataset = data_name
# learning_rate = h_params[dataset]["learning_rate"]
# batch_size = h_params[dataset]["batch_size"]
learning_rate = 1e-4
batch_size = 512
neurons = 100


### New flight delay data loader for customized train/test data same with PI3NN method
xTrain, yTrain, yTrain_scale, test_data_list = data_loader.load_flight_delays('../flight_delay_data/')

# '''choose the train/test dataset '''
x_train = xTrain
y_train = yTrain
y_scale = yTrain_scale
test_idx = 0   # [0, 1, 2, 3] for test 1,2,3,4
x_test = test_data_list[test_idx][0]
y_test = test_data_list[test_idx][1]


seed = 12345
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

args = parser.parse_args()
args.datasets[0] = data_name
training_schemes = [trainers.Evidential]
datasets = args.datasets
print('--- Printing datasets:')
print(datasets)
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
            print('--- data: {}, trial: {}'.format(data_name, n+1))
            print('*********************************************')

            # batch_size = h_params[dataset]["batch_size"]
            num_iterations = num_epochs * x_train.shape[0]//batch_size
            print('num_epochs: {}, num_x_data: {}, batch_size: {}, total iters {} = {} * {} // {}'.format(num_epochs, x_train.shape[0], batch_size, num_iterations, num_epochs, x_train.shape[0], batch_size))
            done = False
            while not done:
                with tf.device(dev):
                    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
                    model, opts = model_generator.create(input_shape=x_train.shape[1:], num_neurons=neurons, tf_seed=seed)

                    trainer = trainer_obj(model, opts, dataset, learning_rate=learning_rate)
                    model, rmse, nll, loss = trainer.train(x_train, y_train, x_test, y_test, y_scale, batch_size=batch_size, iters=num_iterations,
                                                     verbose=True, data_name=data_name, rnd_seed=seed, trial_num=n,
                                                           bool_plot_loss=False, bool_save_loss=True,
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

                    ''' Do same thing for training data in order to do OOD analysis '''
                    x_train_input_tf = tf.convert_to_tensor(x_train, tf.float32)
                    outputs_train = model(x_train_input_tf)

                    mu_train, v_train, alpha_train, beta_train = tf.split(outputs_train, 4, axis=1)
                    epistemic_var_train = np.sqrt(beta_train / (v_train * (alpha_train - 1)))
                    epistemic_var_train = np.minimum(epistemic_var_train, 1e3)

                    y_pred_U_train = mu_train.numpy() + epistemic_var_train * 1.96
                    y_pred_L_train = mu_train.numpy() - epistemic_var_train * 1.96

                    if np.isnan(y_pred_U).any() or np.isnan(y_pred_L).any():
                        PICP = math.nan
                        MPIW = math.nan
                        R2 = math.nan
                        rmse = math.nan
                        nll = math.nan
                        print('--- the y_pred_U/L contains NaN(s) in current trial')
                    else:

                        ''' Calculate the confidence scores (y-axis) range from 0-1'''

                        y_U_cap_train = y_pred_U_train.flatten() > y_train
                        y_L_cap_train = y_pred_L_train.flatten() < y_train
                        MPIW_array_train = y_pred_U_train.flatten() - y_pred_L_train.flatten()
                        MPIW_train = np.mean(MPIW_array_train)

                        #### for test (evaluate each y_U_cap - y_L_cap in the pre-calculated MPIW_train single value
                        # for the confidence score)
                        print(y_pred_U.shape)
                        print(y_pred_L.shape)
                        print(y_test.reshape(-1).shape)

                        y_pred_U = y_pred_U.reshape(-1)
                        y_pred_L = y_pred_L.reshape(-1)


                        y_U_cap = y_pred_U > y_test
                        y_L_cap = y_pred_L < y_test

                        # print('y_U_cap: {}'.format(y_U_cap))
                        # print('y_L_cap: {}'.format(y_L_cap))
                        # print('y_L_cap: {}'.format(y_L_cap))
                        y_all_cap = y_U_cap * y_L_cap
                        PICP = np.sum(y_all_cap) / y_L_cap.shape[0]

                        MPIW_array = y_pred_U - y_pred_L                             
                        MPIW = np.mean(MPIW_array)

                        confidence_arr_test = [min(MPIW_train / test_width, 1.0) for test_width in MPIW_array]
                        confidence_arr_train = [min(MPIW_train / train_width, 1.0) for train_width in MPIW_array_train]

                        print('----------- OOD analysis --- confidence scores ----------------')

                        print('--- Train conf_scores MEAN: {}, STD: {}'.format(np.mean(confidence_arr_train), np.std(confidence_arr_train)))
                        print('--- Test: {} rank: {} conf_scores MEAN: {}, STD: {}'.format(test_idx+1, test_idx+1, np.mean(confidence_arr_test), np.std(confidence_arr_test)))                        

                        ''' Calculate the L2 distance to the mean of training data (x-axis), range from 0-30'''
                        dist_arr_train = np.sqrt(np.sum(x_train ** 2.0, axis=1))
                        dist_arr_test = np.sqrt(np.sum(x_test ** 2.0, axis=1))

                        # print('dist_arr_train shape: {}'.format(dist_arr_train.shape))
                        # print('confidence arr train len: {}'.format(len(confidence_arr_train)))

                        # print('dist_arr_test shape: {}'.format(dist_arr_test.shape))
                        # print('confidence arr test len: {}'.format(len(confidence_arr_test)))

                        ''' Save to file and plot the results '''
                        confidence_arr_train = np.array(confidence_arr_train)
                        confidence_arr_test = np.array(confidence_arr_test)

                        DER_OOD_train_np = np.hstack(
                            (dist_arr_train.reshape(-1, 1), confidence_arr_train.reshape(-1, 1)))
                        DER_OOD_test_np = np.hstack(
                            (dist_arr_test.reshape(-1, 1), confidence_arr_test.reshape(-1, 1)))

                        np.savetxt('DER_OOD_flight_delay_'+ str(test_idx+1) +'_train_np.txt', DER_OOD_train_np, delimiter=',')
                        np.savetxt('DER_OOD_flight_delay_'+ str(test_idx+1) +'_test_np.txt', DER_OOD_test_np, delimiter=',')

                        # plt.plot(dist_arr_train, confidence_arr_train, 'r.', label='Training data (in distribution)')
                        # plt.plot(dist_arr_test, confidence_arr_test, 'b.',label='testing data (out of distribution')
                        # plt.xlabel('L2 distance to the mean of training data $\{x_i\}_{i=1}^N$')
                        # plt.ylabel('The Confidence Score')
                        # plt.legend(loc='lower left')
                        # plt.title('DER flight delay test case '+ str(test_idx+1))
                        # # plt.ylim(0, 1.2)
                        # plt.savefig('DER_OOD_flight_delay_'+str(test_idx+1)+'.png')
                        # # plt.show()



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


# # h_params = {
# #     'yacht': {'learning_rate': 5e-4, 'batch_size': 1},
# #     'naval': {'learning_rate': 5e-4, 'batch_size': 1},
# #     'concrete': {'learning_rate': 5e-3, 'batch_size': 1},
# #     'energy': {'learning_rate': 2e-3, 'batch_size': 1},
# #     'kin8nm': {'learning_rate': 1e-3, 'batch_size': 1},
# #     'power': {'learning_rate': 1e-3, 'batch_size': 2},
# #     'boston': {'learning_rate': 1e-3, 'batch_size': 8},
# #     'wine': {'learning_rate': 1e-4, 'batch_size': 32},
# #     'protein': {'learning_rate': 1e-3, 'batch_size': 64},
# #     'MSD': {'learning_rate': 1e-3, 'batch_size': 256} # we added this dataset
# # }