import os
import argparse
import pathlib
import shutil
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from scipy import stats


class CL_Utils:

    def __init__(self):
        pass

    def standardizer(self, input_np):
        input_mean = input_np.mean(axis=0, keepdims=1)
        input_std = input_np.std(axis=0, keepdims=1)
        input_std[input_std < 1e-10] = 1.0
        standardized_input = (input_np - input_mean) / input_std
        return standardized_input, input_mean, input_std


    def getNumInputsOutputs(self, inputsOutputs_np):
        if len(inputsOutputs_np.shape) == 1:
            numInputsOutputs = 1
        if len(inputsOutputs_np.shape) > 1:
            numInputsOutputs = inputsOutputs_np.shape[1]
        return numInputsOutputs

    ### Convert the dtype from numpy.float64, numpy.int64 to python float, int
    ### after using Hyperopt package
    def convertDtype(self, optim_configs):
        for key, value in optim_configs.items():
            # print(type(value).__name__)
            if type(value).__name__== 'float64':
                optim_configs[key] = float(optim_configs[key])
            if type(value).__name__== 'int64':
                optim_configs[key] = int(optim_configs[key])
            if type(value).__name__=='list':
                for i in range(len(value)):
                    if type(value[i]).__name__ == 'float64':
                        optim_configs[key][i] = float(optim_configs[key][i])
                    if type(value[i]).__name__ == 'int64':
                        optim_configs[key][i] = int(optim_configs[key][i])
        return optim_configs


    #### analyze and plot the R2(s) for all hyper-params tuning optimization trials, based on (1) sorted test R2; (2) sorted train R2
    #### and return the TWO sorted 'tid' list
    def sortOptimConfigs(self, configs, path=None, plot_r2s=False, savefig=False):
        trials = pickle.load(open(path, 'rb'))
        final_train_r2s = []
        final_valid_r2s = []
        final_test_r2s = []
        tid_list = []

        for i in range(len(trials.trials)):
            # print(trials.trials[i]['tid'])
            ## extract final r2s
            tid_list.append(trials.trials[i]['tid'])
            final_train_r2s.append(trials.trials[i]['final_r2s'][0])
            final_valid_r2s.append(trials.trials[i]['final_r2s'][1])
            final_test_r2s.append(trials.trials[i]['final_r2s'][2])

        final_r2s_df = pd.DataFrame({
            'tid': tid_list,
            'final_train_r2s': final_train_r2s,
            'final_valid_r2s': final_valid_r2s,
            'final_test_r2s': final_test_r2s
            })
        ### sort r2s based on test and train
        sorted_train_r2s_df = final_r2s_df.sort_values(by=['final_train_r2s'], ascending=False)
        sorted_test_r2s_df = final_r2s_df.sort_values(by=['final_test_r2s'], ascending=False)
        tid_train_r2_sorted = sorted_train_r2s_df['tid'].values 
        tid_test_r2_sorted = sorted_test_r2s_df['tid'].values

        ### plot
        if plot_r2s:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
            ax1.plot(range(len(sorted_train_r2s_df)), sorted_train_r2s_df['final_train_r2s'], '-o', alpha=0.7, label='sorted train R2s')
            ax1.plot(range(len(sorted_train_r2s_df)), sorted_train_r2s_df['final_test_r2s'], '-o', alpha=0.7, label='corresponding test R2s')
            ax1.set_ylim([0, 1.05])
            ax1.set_xlabel('trials')
            ax1.set_ylabel('R2')
            ax1.set_title('Sorted based on Train R2')
            ax1.legend()

            ax2.plot(range(len(sorted_test_r2s_df)), sorted_test_r2s_df['final_train_r2s'], '-o', alpha=0.7, label='corresponding train R2s')
            ax2.plot(range(len(sorted_test_r2s_df)), sorted_test_r2s_df['final_test_r2s'], '-o', alpha=0.7, label='sorte test R2s')
            ax2.set_ylim([0, 1.05])
            ax2.set_xlabel('trials')
            ax2.set_ylabel('R2')
            ax2.set_title('Sorted based on Test R2')
            ax2.legend()

            plt.suptitle('Sorted R2(s) for \'{}\' with {} inputs, {} trials'.format(configs['data_name'], configs['num_inputs'], len(sorted_test_r2s_df)))
            if savefig:
                plt.savefig('./Results_PI3NN/optim_configs/'+configs['data_name']+'_'+str(configs['num_inputs'])+'_inputs_dump.png')
            plt.show()

        # print('max train r2: {}'.format(final_r2s_df['final_train_r2s'].max()))
        # print('max test r2: {}'.format(final_r2s_df['final_test_r2s'].max()))

        ### remove the cases where test R2 > train R2 based on the sorted_test_r2s_df

        clean_tid_test_r2_sorted = []
        for i in range(len(sorted_test_r2s_df)):
            # print(sorted_test_r2s_df['tid'].iloc[i], sorted_test_r2s_df['final_test_r2s'].iloc[i], sorted_test_r2s_df['final_train_r2s'].iloc[i])
            if sorted_test_r2s_df['final_test_r2s'].iloc[i] <= sorted_test_r2s_df['final_train_r2s'].iloc[i]:
                clean_tid_test_r2_sorted.append(sorted_test_r2s_df['tid'].iloc[i])
        # print(clean_tid_test_r2_sorted)

        tmp_idx = clean_tid_test_r2_sorted[0]
        best_train_r2 = final_r2s_df.loc[final_r2s_df['tid'] == tmp_idx, 'final_train_r2s'].item()
        best_test_r2 = final_r2s_df.loc[final_r2s_df['tid'] == tmp_idx, 'final_test_r2s'].item()

        best_valid_r2 = final_r2s_df.loc[final_r2s_df['tid'] == tmp_idx, 'final_valid_r2s'].item()

        print('max train r2: {}'.format(best_train_r2))
        print('max test r2: {}'.format(best_test_r2))
        print('max valid r2: {}'.format(best_valid_r2))

        return tid_train_r2_sorted, tid_test_r2_sorted, clean_tid_test_r2_sorted

    ### function for analysis and plot the residual from MEAN network training
    def analyze_residuals(self, train_idx, valid_idx, diff_train, diff_valid, diff_test, yTrain, \
        yTrain_up_data, yTrain_down_data, yValid, yValid_up_data, yValid_down_data, yTest, yTest_up_data, yTest_down_data, \
        plotfig=False, savefig=False, saveData=False, title=None):
        train_up_idx = np.where(diff_train>0)[0].tolist()
        train_down_idx = np.where(diff_train<0)[0].tolist()

        test_up_idx = np.where(diff_test>0)[0].tolist()
        test_down_idx = np.where(diff_test<0)[0].tolist()

        valid_up_idx = np.where(diff_valid>0)[0].tolist()
        valid_down_idx = np.where(diff_valid<0)[0].tolist()

        if plotfig:
            # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,10))
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20,10))

            # ax1.plot(train_idx, yTrain, 'o', alpha=0.9, label='yTrain')
            # ax1.plot(valid_idx, yValid, 'o', alpha=0.9, label='yValid')
            tmp_arr = np.zeros(len(train_idx)+len(valid_idx))
            tmp_arr[train_idx] = yTrain.flatten()
            tmp_arr[valid_idx] = yValid.flatten()
            ax1.plot(tmp_arr, label='train/valid Y')

            # ax1.plot(train_idx[train_up_idx], yTrain_up_data+yTrain[train_up_idx], 'o', alpha=0.5, label='train_up')
            # ax1.plot(train_idx[train_down_idx], -1.0*yTrain_down_data + yTrain[train_down_idx], 'o', alpha=0.5, label='train_down')

            # ax1.plot(valid_idx[valid_up_idx], yValid_up_data+yValid[valid_up_idx], '*', alpha=0.5, label='valid_up')
            # ax1.plot(valid_idx[valid_down_idx], -1.0*yValid_down_data+yValid[valid_down_idx], '*', alpha=0.5, label='valid_down')

            # ax2.plot(test_up_idx, yTest_up_data+yTest[test_up_idx], 'o', alpha=0.5, label='test_up')
            # ax2.plot(test_down_idx, -1.0*yTest_down_data+yTest[test_down_idx], 'o', alpha=0.5, label='test_down')
            # ax2.plot(yTest, label='test_Y')

            ax1.plot(train_idx[train_up_idx], yTrain_up_data, 'o', alpha=0.5, label='train_up')
            ax1.plot(train_idx[train_down_idx], yTrain_down_data, 'o', alpha=0.5, label='train_down')

            ax1.plot(valid_idx[valid_up_idx], yValid_up_data, '*', alpha=0.5, label='valid_up')
            ax1.plot(valid_idx[valid_down_idx], yValid_down_data, '*', alpha=0.5, label='valid_down')

            ax2.plot(test_up_idx, yTest_up_data, 'o', alpha=0.5, label='test_up')
            ax2.plot(test_down_idx, yTest_down_data, 'o', alpha=0.5, label='test_down')
            ax2.plot(yTest, label='test_Y')

            ### calculate the R2s
            # r2_train = r2_score()
            # r2_valid = r2_score()
            # r2_test = r2_score()

            ax1.set_title('Train U/D: {}/{}, {}/{}, Valid U/D: {}/{}, {}/{}'.format(len(train_up_idx), len(yTrain), len(train_down_idx), \
                len(yTrain),len(valid_up_idx), len(yValid), len(valid_down_idx), len(yValid)))
            ax2.set_title('Test U/D: {}/{}, {}/{}'.format(len(test_up_idx), len(yTest), len(test_down_idx), len(yTest)))
            ax1.set_xlabel('days')
            ax1.set_ylabel('Normalized Y')
            ax2.set_xlabel('days')
            ax2.set_ylabel('Normalized Y')

            ### Calculate min/max/mean/std and generate PDF plots
            res_trainval = np.concatenate((yTrain_up_data, yTrain_down_data, yValid_up_data, yValid_down_data))
            res_test = np.concatenate((yTest_up_data, yTest_down_data))

            res_trainval_min, res_trainval_max = res_trainval.min(), res_trainval.max()
            res_trainval_mean, res_trainval_std = res_trainval.mean(), res_trainval.std()

            res_test_min, res_test_max = res_test.min(), res_test.max()
            res_test_mean, res_test_std = res_test.min(), res_test.std()

            ### for PDF plots
            kde_res_trainval = stats.gaussian_kde(res_trainval.flatten())
            kde_res_test = stats.gaussian_kde(res_test.flatten())

            x1 = np.linspace(res_trainval.min(), res_trainval.max(), 100)
            p1 = kde_res_trainval(x1)

            x2 = np.linspace(res_test.min(), res_test.max(), 100)
            p2 = kde_res_test(x2)

            ax3.plot(x1, p1, label='train/valid')
            ax4.plot(x2, p2, label='test')
            ax3.set_title('PDF for train/val res, \n min:{:.5f}, max:{:.5f}, mean:{:.5f}, std:{:.5f}'.format(res_trainval_min, res_trainval_max, res_trainval_mean, res_trainval_std))
            ax4.set_title('PDF for test res, \n min:{:.5f}, max:{:.5f}, mean:{:.5f}, std:{:.5f}'.format(res_test_min, res_test_max, res_test_mean, res_test_std))

            ax3.legend()
            ax4.legend()

            ax1.legend()
            ax2.legend()
            if title is not None:
                plt.suptitle(title)

            plt.tight_layout()
            plt.show()

            plt.close()
        if savefig is not None:
            plt.savefig(savefig)
  
    ### Calculate the Kling-Gupta efficiency (KGE)
    # Ref: Knoben, Wouter JM, Jim E. Freer, and Ross A. Woods. "Inherent benchmark or not? Comparing Nash–Sutcliffe and Kling–Gupta efficiency scores." 
    # Hydrology and Earth System Sciences 23, no. 10 (2019): 4323-4331.
    def KGE_score(self, yTrue, yPred):
        alpha = yPred.std() / yTrue.std()
        beta = yPred.mean() / yTrue.mean()
        ### Pearson's Correlation
        # r = np.cov(yPred, yTrue) / (yPred.std() * yTrue.std())
        r, _ = stats.pearsonr(yPred, yTrue)
        KGE = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
        return KGE

    ### argument parser helping functions
    def dir_path_proj(self, path):           # project folders
        if os.path.exists(path):
            print('*** The project folder: {}    <EXIST>'.format(path))
            self.proj_path = path
            return path
        else:
            if input('*** The project folder: {} does NOT exist, do you want to create one? (y/n)'.format(path)) != 'y':
                print('*** The program ended.')
                exit()
            else: ## create new project folder
                pathlib.Path(path).mkdir(parents=True, exist_ok=False)
                print('*** New project folder: {} is created'.format(path))
            self.proj_path = path
            return path

    def dir_path_exp(self, exp_name):    ## experiment name/folder
        if os.path.exists(self.proj_path+exp_name):
            print('*** The exp folder: {}    <EXIST>'.format(self.proj_path+exp_name))
            # if input('--- The experiment folder: {} EXIST, do you want to overwrite it? WARNING: if yes, ALL files will be deleted (y/n) '.format(self.proj_path+exp_name)) != 'y':
            #     print('--- The program ended')
            #     exit()
            # else:
            #     shutil.rmtree(self.proj_path+exp_name)
            #     pathlib.Path(self.proj_path+exp_name).mkdir(parents=True, exist_ok=False)
            #     print('--- New folder: {} created and overwrote the existing one'.format(self.proj_path+exp_name))
            return exp_name
        else:
            if input('*** The exp folder: {} does NOT exist, do you want to create one? (y/n)'.format(exp_name)) != 'y':
                print('*** The program ended.')
                exit()
            else: ## create new experiment folder
                pathlib.Path(self.proj_path+exp_name).mkdir(parents=True, exist_ok=False)
                print('*** New experiment folder: {} is created'.format(self.proj_path+exp_name))
                return exp_name

    def dir_path_configs(self, path):           # configs path
        if os.path.exists(path):
            print('*** The configs file: {} EXIST'.format(path))
            print('*** Loading configs...')
            return path
        else:
            print('*** The configs file: {} does NOT exist, please prepare one'.format(path))
            print('*** The program ended.')
            exit()

    def check_encoder_folder(self, path):
        if os.path.exists(path):
            print('*** The encoder results folder: {} EXIST'.format(path))
            return path
        else:
            print('*** The encoder results folder: {} does NOT exist, creating one...'.format(path))
            pathlib.Path(path).mkdir(parents=True, exist_ok=False)

    def check_PI3NN_folder(self, path):
        if os.path.exists(path):
            print('*** The PI3NN results folder: {} EXIST'.format(path))
            return path
        else:
            print('*** The PI3NN results folder: {} does NOT exist, creating one...'.format(path))
            pathlib.Path(path).mkdir(parents=True, exist_ok=False)  

    def check_encoder_predictions(self, path, data_name=None):
        ### check the existance of the encoder_path folder
        if os.path.exists(path):
            print('*** The encoder results folder: {} EXIST'.format(path))
            ### check each individual files
            tot_encoder_files = 5
            num_encoder_files = 0
            if os.path.exists(path+'/'+data_name+'_encoder_trainValid_X.txt'):
                num_encoder_files += 1
                print('*** ({}/{}) encoder results file \'{}_encoder_trainValid_X.txt\' --- EXIST'.format(num_encoder_files, tot_encoder_files, data_name))
            else:
                print('*** encoder results file \'{}_encoder_trainValid_X.txt\' --- NOT EXIST'.format(data_name))

            if os.path.exists(path+'/'+data_name+'_encoder_test_X.txt'):
                num_encoder_files += 1
                print('*** ({}/{}) encoder results file \'{}_encoder_test_X.txt\' --- EXIST'.format(num_encoder_files, tot_encoder_files, data_name))
            else:
                print('*** encoder results file \'{}_encoder_test_X.txt\' --- NOT EXIST'.format(data_name))

            if os.path.exists(path+'/'+data_name+'_encoder_trainValid_Y.txt'):
                num_encoder_files += 1
                print('*** ({}/{}) encoder results file \'{}_encoder_trainValid_Y.txt\' --- EXIST'.format(num_encoder_files, tot_encoder_files, data_name))
            else:
                print('*** encoder results file \'{}_encoder_trainValid_Y.txt\' --- NOT EXIST'.format(data_name))

            if os.path.exists(path+'/'+data_name+'_encoder_test_Y.txt'):
                num_encoder_files += 1
                print('*** ({}/{}) encoder results file \'{}_encoder_test_Y.txt\' --- EXIST'.format(num_encoder_files, tot_encoder_files, data_name))
            else:
                print('*** encoder results file \'{}_encoder_test_Y.txt\' --- NOT EXIST'.format(data_name))

            if os.path.exists(path+'/'+data_name+'_trainValid_idx.txt'):
                num_encoder_files += 1
                print('*** ({}/{}) encoder results file \'{}_trainValid_idx.txt\' --- EXIST'.format(num_encoder_files, tot_encoder_files, data_name))
            else:
                print('*** encoder results file \'{}_trainValid_idx.txt\' --- NOT EXIST'.format(data_name))

            if num_encoder_files < 5:
                print('*** Missing one or more encoder prediction results, please double check')
                print('*** Program ended.')
                exit()
        else:
            print('*** The encoder results folder: {} does NOT exist.'.format(path))
            print('*** The encoder results are required for PI3NN-MLP training, please specify the correct folder in <encoder_path> keyword within the configs file')
            print('*** program ended.')
            exit()  


    def load_scalers(self, path):
        print('*** Checking scalers from {}'.format(path))
        scalerxy_count = 0
        if os.path.exists(path+'/'+'scalerx.pkl'):
            print('*** scaler file \'scalerx.pkl\' --- EXIST')
            scalerx = pickle.load(open(path+'/'+'scalerx.pkl', 'rb'))
            scalerxy_count += 1
        else:
            print('*** scaler file \'scalerx.pkl\' --- DOES NOT EXIST')

        if os.path.exists(path+'/'+'scalery.pkl'):
            print('*** scaler file \'scalery.pkl\' --- EXIST')
            scalery = pickle.load(open(path+'/'+'scalery.pkl', 'rb'))
            scalerxy_count += 1
        else:
             print('*** scaler file \'scalery.pkl\' --- DOES NOT EXIST')
        
        if scalerxy_count<2:
            print('*** WARNING: missing 1 or 2 scalaers, please check the loading path or files')
            print('*** program ended.')
            exit()
        else:
            return scalerx, scalery

    def load_PI3NN_saved_models(self, path, data_name=None):
        print('*** Checking and loading saved PI3NN models from {}'.format(path))
        PI3NN_models_count = 0
        if os.path.exists(path+'/'+data_name+'_mean_model'):
            print('*** PI3NN mean model \'{}_mean_model\' --- EXIST'.format(data_name))
            PI3NN_models_count+=1
            net_mean = tf.saved_model.load(path+'/'+data_name+'_mean_model')
        else:
            print('*** PI3NN mean model \'{}_mean_model\' --- DOES NOT EXIST'.format(data_name))

        if os.path.exists(path+'/'+data_name+'_up_model'):
            print('*** PI3NN up model \'{}_up_model\' --- EXIST'.format(data_name))
            PI3NN_models_count+=1
            net_up = tf.saved_model.load(path+'/'+data_name+'_up_model')
        else:
            print('*** PI3NN up model \'{}_up_model\' --- DOES NOT EXIST'.format(data_name))

        if os.path.exists(path+'/'+data_name+'_down_model'):
            print('*** PI3NN down model \'{}_down_model\' --- EXIST'.format(data_name))
            PI3NN_models_count+=1
            net_down = tf.saved_model.load(path+'/'+data_name+'_down_model')
        else:
            print('*** PI3NN down model \'{}_down_model\' --- DOES NOT EXIST'.format(data_name))

        if PI3NN_models_count<3:
            print('*** WARNING: missing 1-3 PI3NN models, please check the loading path or files')
            print('*** program ended.')
            exit()
        else:
            return net_mean, net_up, net_down

    def save_PI3NN_models(self, path, mean_model, up_model, down_model, scalerx=None, scalery=None, data_name=None):
        print('*** Saving PI3NN models to {}'.format(path))

        tf.saved_model.save(mean_model, path+'/'+data_name+'_mean_model') 
        print('--- Saved PI3NN_MLP MEAN model to {}_mean_model'.format(path+'/'+data_name))

        tf.saved_model.save(up_model, path+'/'+data_name+'_up_model') 
        print('--- Saved PI3NN_MLP UP model to {}_up_model'.format(path+'/'+data_name))

        tf.saved_model.save(down_model, path+'/'+data_name+'_down_model')
        print('--- Saved PI3NN_MLP DOWN model to {}_down_model'.format(path+'/'+data_name))

        if scalerx is not None:
            pickle.dump(scalerx, open(path+'/'+'scalerx.pkl', 'wb'))
            print('--- Saved PI3NN_MLP Scaler_x to {}'.format(path+'/'+'scalerx.pkl'))
        if scalery is not None:
            pickle.dump(scalery, open(path+'/'+'scalery.pkl', 'wb'))
            print('--- Saved PI3NN_MLP Scaler_y to {}'.format(path+'/'+'scalery.pkl'))


    ### Check existence of the folder, if not, create one (ask for permission)
    def check_create_dir(self, path):
        try:
            pathlib.Path(path).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print('--- The path: {} exist'.format(path))
        else:
            print('--- The project folder: {} has been created'.format(path))


'''Customized loss functions'''
class CL_losses:
    def __init__(self):
        pass
    # def MSE(self, yTrue, yPred):
    #     mse = np.square(np.subtract(yTrue, yPred)).mean()
    #     return mse

    def RMSE(self, yTrue, yPred):
        rmse = np.sqrt(np.square(np.subtract(yTrue, yPred)).mean())

    def MSE(self, yTrue, yPred):
        mse = tf.math.reduce_mean(tf.math.square(tf.math.subtract(yTrue, yPred)))
        return mse

    def wMSE(self, yTrue, yPred, weights):
        wmse = tf.math.reduce_mean(weights * tf.math.square(tf.math.subtract(yTrue, yPred)))
        return wmse


''' The early stopping method is taken and adjusted from Keras/Tensorflow EarlyStopping callback function '''
class CL_EarlyStopping:

    def __init__(self,
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0

        if mode not in ['auto', 'min', 'max']:
            print('--- Warning: EarlyStopping mode {} is unknown'.format(mode), 'auto mode will be used')
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def init_early_stopping(self):
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
            # self.best = np.Inf
        self.best_weights = None


    def earlyStop_onEpochEnd_eval(self, epoch, current_loss):
        if self.monitor_op(current_loss - self.min_delta, self.best):
            self.best = current_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get


    def on_train_begin(self):
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None

