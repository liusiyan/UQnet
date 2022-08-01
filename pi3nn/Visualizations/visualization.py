''' Functions for plotting '''
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# matplotlib.rcParams['text.usetex'] = True
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
# print(plt.get_backend())
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.transforms as transforms
from pi3nn.Utils.Utils import CL_Utils

utils = CL_Utils()

class CL_plotter:
    def __init__(self, configs):
        self.configs = configs

    ### plot the original train/valid/test data 
    def plotLSTM_raw_Y(self, yTrain, yValid, yTest, train_idx, valid_idx, scalery=None, figname=None, savefig=None, ylim_1=None, ylim_2=None):
        print('--- Log transformed: {}'.format(self.configs['ylog_trans']))
        yTrain = scalery.inverse_transform(yTrain)
        yValid = scalery.inverse_transform(yValid)
        yTest = scalery.inverse_transform(yTest)

        if self.configs['ylog_trans']:
            yTrain = np.exp(yTrain)
            yValid = np.exp(yValid)
            yTest = np.exp(yTest)

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,10))

        ax1.scatter(train_idx, yTrain, color='g', s=20, alpha=0.5, label='Train Y')
        ax1.scatter(valid_idx, yValid, color='b', s=20, alpha=0.5, label='Valid Y')

        ax1.set_title('Train/Valid')
        ax1.set_xlabel('Time (d)')
        ax1.set_ylabel('Discharge at {} (m^2/s)'.format(self.configs['data_name']))

        ax2.scatter(np.arange(len(yTest)), yTest, color='r', s=20, alpha=0.5, label='Test Y')
        ax2.set_title('Test')

        plt.suptitle('Original Train/Valid/Test data for {} data'.format(self.configs['data_name']))

        if ylim_1 is not None:
            print(ylim_1)
            ax1.set_ylim(ylim_1)
        if ylim_2 is not None:
            ax2.set_ylim(ylim_2)

        ax1.legend()
        ax2.legend()

        if savefig is not None:
            plt.savefig(savefig)
        # plt.show()
        plt.close()

    ### plot LSTM encoder predictions
    def plotLSTM_encoder(self, train_idx, valid_idx, train_results_np, valid_results_np, test_results_np, figname=None, savefig=None, plot_hline=None, ylim_1=None, ylim_2=None, show_plot=False):

        trainValid_np = np.zeros((len(train_results_np)+len(valid_results_np), 2))
        for i in range(2):
            trainValid_np[train_idx, i] = train_results_np[:, i]
            trainValid_np[valid_idx, i] = valid_results_np[:, i]
        r2_trainValid = r2_score(trainValid_np[:, 0], trainValid_np[:, 1])
        r2_test = r2_score(test_results_np[:, 0], test_results_np[:, 1])

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,10))

        ax1.scatter(train_idx, train_results_np[:, 0], marker='o', color='g', s=20, alpha=0.3, label='Obs Train Y')
        ax1.scatter(train_idx, train_results_np[:, 1], marker='o', color='b', s=20, alpha=0.3, label='Pred Train Y')

        ax1.scatter(valid_idx, valid_results_np[:, 0], marker='*', color='g', s=20, alpha=0.3, label='Obs Valid Y')
        ax1.scatter(valid_idx, valid_results_np[:, 1], marker='*', color='b', s=20, alpha=0.3, label='Pred Valid Y')

        ax1.set_title('TrainValid, R2 = {:.4f}'.format(r2_trainValid))
        ax1.set_xlabel('Time (d)')
        ax1.set_ylabel('Discharge at {} (m^2/s)'.format(self.configs['data_name']))

        ax2.plot(test_results_np[:, 0],'r', label='Observation')
        ax2.plot(test_results_np[:, 1],'b--',label='Test prediction')

        ax2.set_title('Test, R2 = {:.4f}'.format(r2_test))
        ax2.set_xlabel('Time (d)')
        ax2.set_ylabel('Discharge at {} (m^2/s)'.format(self.configs['data_name']))

        ax1.legend()
        ax2.legend()

        if ylim_1 is not None:
            ax1.set_ylim(ylim_1)
        if ylim_2 is not None:
            ax2.set_ylim(ylim_2)

        if figname is not None:
            plt.suptitle(figname)

        if savefig is not None:
            plt.savefig(savefig)

        if show_plot:
            plt.show()

        plt.close()


    ### this function is designed for PI3NN-CNN on Permeability estimation example
    def plot_MLPI3NN_CNN_Perm(self):
        pass


    ### this function was originally designed for PI3NN-LSTM streamflow examples
    def plot_MLPI3NN(self, train_idx, valid_idx, train_np, valid_np, test_np, figname=None, savefig=None, plot_hline=None, ylim_1=None, ylim_2=None, gaussian_filter=False, selected_coverage_region=False, show_plot=False, \
        plot_r2s_PICPs=False, test_r2s=None, test_PICPs=None, test_MPIWs=None, days_year_selected_test=None, save_results=None,
        train_PIW_quantile=None):
        trainValid_np = np.zeros((len(train_np)+len(valid_np), 4))
        for i in range(4):
            trainValid_np[train_idx, i] = train_np[:, i]
            trainValid_np[valid_idx, i] = valid_np[:, i]   

        if save_results is not None:
            np.save(save_results+'trainValid_np.npy', trainValid_np)
            np.save(save_results+'test_np.npy', test_np)    

        r2_trainValid = r2_score(trainValid_np[:, 0], trainValid_np[:, 1])
        r2_test = r2_score(test_np[:, 0], test_np[:, 1])
        KGE_trainValid = utils.KGE_score(trainValid_np[:, 0], trainValid_np[:, 1])
        KGE_test = utils.KGE_score(test_np[:, 0], test_np[:, 1])

        if plot_r2s_PICPs:
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1, figsize=(24,14))

        else:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10,10))

        ax1.scatter(train_idx, train_np[:, 0], marker='o', color='g', s=20, alpha=0.3, label='Obs Train Y')
        ax1.scatter(train_idx, train_np[:, 1], marker='o', color='b', s=20, alpha=0.3, label='Pred Train Y')

        ax1.scatter(valid_idx, valid_np[:, 0], marker='*', color='g', s=20, alpha=0.3, label='Obs Valid Y')
        ax1.scatter(valid_idx, valid_np[:, 1], marker='*', color='b', s=20, alpha=0.3, label='Pred Valid Y')

        ### combine train and valid
        trainValid_up = np.zeros(len(train_np)+len(valid_np))
        trainValid_down = np.zeros(len(trainValid_up))

        trainValid_up[train_idx] = train_np[:, 2]
        trainValid_up[valid_idx] = valid_np[:, 2]
        trainValid_down[train_idx] = train_np[:, 3]
        trainValid_down[valid_idx] = valid_np[:, 3]

        ### calculate coverage percentage
        trainValid_obs = 0
        test_in_obs = 0
        test_out_obs = 0

        trainValid_all = len(trainValid_np)
        test_all = len(test_np)    

        for i in range(trainValid_all):
            if selected_coverage_region:
                if i in range(210, 250) or i in range(560, 620):
                    if (trainValid_np[i, 0] >= trainValid_np[i, 3]) and (trainValid_np[i, 0] <= trainValid_np[i, 2]):
                        trainValid_in_obs+=1
            else:
                if (trainValid_np[i, 0] >= trainValid_np[i, 3]) and (trainValid_np[i, 0] <= trainValid_np[i, 2]):
                    trainValid_obs+=1 

        if train_PIW_quantile is not None:
            np_q_train = np.quantile((trainValid_np[:,2]-trainValid_np[:,3]), train_PIW_quantile)

            test_in_idx = []
            test_out_idx = []
            test_in_up = []
            test_in_down = []
            test_out_up = []
            test_out_down = []

            for i in range(test_all):
                if train_PIW_quantile is not None:
                    tmp_test_width = (test_np[i, 2]-test_np[i, 3])
                    if tmp_test_width < np_q_train:     ### assume InD
                        test_in_idx.append(i)
                        test_in_up.append(test_np[i, 2])
                        test_in_down.append(test_np[i, 3])
                    else:
                        test_out_idx.append(i)          ### assume OOD
                        test_out_up.append(test_np[i, 2])
                        test_out_down.append(test_np[i, 3])

            ### calculate PICP based on values < train_PIW_quantile
            test_in_all = len(test_in_idx)
            for i, idx in enumerate(test_in_idx):
                if (test_np[idx, 0] >= test_in_down[i]) and (test_np[idx, 0] <= test_in_up[i]):
                    test_in_obs += 1

            ### calculate PICP based on values >= train_PIW_quantile
            test_out_all = len(test_out_idx)
            for i, idx in enumerate(test_out_idx):
                if (test_np[idx, 0] >= test_out_down[i]) and (test_np[idx, 0] <= test_out_up[i]):
                    test_out_obs += 1  

        if train_PIW_quantile is not None:
            if self.configs['PICP_evaluation_mode'] == 0: # evaluate PICP based on values < threshold from given train quantile
                test_all = test_in_all
                test_obs = test_in_obs

            elif self.configs['PICP_evaluation_mode'] == 1: # evaluate PICP based on values >= threshold from given train quantile
                test_all = test_out_all
                test_obs = test_out_obs

            # print('--- InD: {}/{}={:.2f}'.format(test_in_obs, test_in_all, test_in_obs/test_in_all*100))
            # print('--- OOD: {}/{}={:.2f}'.format(test_out_obs, test_out_all, test_out_obs/test_out_all*100))

            # print('--- test in idx len: {}'.format(len(test_in_idx)))
            # print('--- test out idx len: {}'.format(len(test_out_idx)))

        if train_PIW_quantile is None:
            test_obs = 0
            for i in range(test_all):
                if (test_np[i, 0] >= test_np[i, 3]) and (test_np[i, 0] <= test_np[i, 2]):
                    test_obs+=1                 
    
        trainValid_perc = trainValid_obs/trainValid_all*100
        test_perc = test_obs/test_all*100

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
        ax1.plot(trainValid_np[:, 0],'r', label='Observation')
        ax1.plot(trainValid_np[:, 1],'b--',label='Train prediction')
        ax2.plot(test_np[:, 0],'r', label='Observation')
        ax2.plot(test_np[:, 1],'b--',label='Test prediction')
        ax1.set_ylim(self.configs['plot_PI3NN_ylims'][0])
        ax2.set_ylim(self.configs['plot_PI3NN_ylims'][1])

        if gaussian_filter:
            trainValid_np[:, 2] = gaussian_filter1d(trainValid_np[:, 2], sigma=2)
            trainValid_np[:, 3] = gaussian_filter1d(trainValid_np[:, 3], sigma=2)
            test_np[:, 2] = gaussian_filter1d(test_np[:, 2], sigma=2)
            test_np[:, 3] = gaussian_filter1d(test_np[:, 3], sigma=2)

        ax1.fill_between(np.arange(len(trainValid_np[:, 2])), trainValid_np[:, 2], trainValid_np[:, 3], color='gray', alpha=0.5, label='Uncertainty')

        if train_PIW_quantile is not None:
            ax2.fill_between(np.arange(len(test_np[:, 2])), test_np[:, 2], test_np[:, 3], where=((test_np[:, 2]-test_np[:, 3])<np_q_train), color='C0', alpha=0.5, label='PI<threshold', interpolate=False)
            ax2.fill_between(np.arange(len(test_np[:, 2])), test_np[:, 2], test_np[:, 3], where=((test_np[:, 2]-test_np[:, 3])>=np_q_train), color='C1', alpha=0.5, label='PI>=threshold', interpolate=False)
        else:
            ax2.fill_between(np.arange(len(test_np[:, 2])), test_np[:, 2], test_np[:, 3], color='gray', alpha=0.5, label='Uncertainty')

        ax1.set_title('TrainValid, $R^2$: {:.4f},  KGE: {:.4f}, trainValid_obs (PICP): {}/{}={:.2f} %'.format(r2_trainValid, KGE_trainValid, trainValid_obs, trainValid_all, trainValid_perc))

        if train_PIW_quantile is not None:
            title_str = 'Test, $R^2$ = {:.4f}, KGE: {:4f}, test_obs (PICP): {}/{}={:.2f} %'
            title_str += '\n'
            title_str += 'target train_PIW_quantile: {}, width: {:.4f}, below/above: {}/{}'
            ax2.set_title(title_str.format(r2_test, KGE_test, test_obs, test_all, test_perc, train_PIW_quantile, np_q_train, len(test_in_idx), len(test_out_idx)))

        else:
            ax2.set_title('Test, $R^2$ = {:.4f}, KGE: {:4f}, test_obs (PICP): {}/{}={:.2f} %'.format(r2_test, KGE_test, test_obs, test_all, test_perc))
        ax1.set_xlabel('Time (d)')
        ax1.set_ylabel('Discharge at {} ($m^2/s$)'.format(self.configs['data_name']))
        ax2.set_xlabel('Time (d)')
        ax2.set_ylabel('Discharge at {} ($m^2/s$)'.format(self.configs['data_name']))

        ax1.legend()
        ax2.legend()

        fig.tight_layout()
        plt.subplots_adjust(top=0.92)

        if figname is not None:
            plt.suptitle(figname)

        if savefig is not None:
            plt.savefig(savefig)       

        # plt.show()
        plt.close()

        return trainValid_perc, test_perc