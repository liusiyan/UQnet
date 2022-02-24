
''' Data loaders '''

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class CL_dataLoader:
    def __init__(self, original_data_path=None, configs=None):
        # current_dir = os.path.dirname(__file__)
        # self.data_dir = os.path.join(current_dir, 'UCI_datasets')
        if original_data_path:
            self.data_dir = original_data_path
        if configs:
            self.configs = configs

    ''' (1) Boston '''
    def load_boston(self, Y_data='default'):
        rawData_boston_np = np.loadtxt(os.path.join(self.data_dir, 'boston-housing/boston_housing.txt'))
        X = rawData_boston_np[:, :-1]
        Y = rawData_boston_np[:, -1]
        return X, Y

    ''' (2) Concrete '''
    def load_concrete(self, Y_data='default'):
        rawData_concrete_df = pd.read_excel(os.path.join(self.data_dir, 'concrete/Concrete_Data.xls'))
        X = rawData_concrete_df.values[:, :-1]
        Y = rawData_concrete_df.values[:, -1]
        return X, Y

    ''' (3) Energy (energy efficiency) '''
    def load_energy_efficiency(self, Y_data='default'):
        rawData_energy_df = pd.read_excel(os.path.join(self.data_dir, 'energy-efficiency/ENB2012_data.xlsx'),
                                          engine='openpyxl')
        rawData_energy_df = rawData_energy_df.dropna(how='all', axis='columns')
        rawData_energy_df = rawData_energy_df.dropna(how='all', axis='rows')

        X = rawData_energy_df.values[:, :-2]
        Y_heating = rawData_energy_df.values[:, -2]
        Y_cooling = rawData_energy_df.values[:, -1]
        Y_all = rawData_energy_df.values[:, -2:]
        if Y_data == 'heating':
            return X, Y_heating
        elif Y_data == 'cooling':
            return X, Y_cooling
        elif Y_data == 'all':
            return X, Y_all
        else:
            return X, Y_heating, Y_cooling

    ''' (4) Kin8nm --- forward kinematics of an 8 link robot arm '''
    def load_kin8nm(self, Y_data='default'):
        rawData_kin8nm_df = pd.read_csv(os.path.join(self.data_dir, 'kin8nm/dataset_2175_kin8nm.csv'), sep=',')
        X = rawData_kin8nm_df.values[:, :-1]
        Y = rawData_kin8nm_df.values[:, -1]
        return X, Y

    ''' (5) Naval '''
    def load_naval(self, Y_data='default'):
        rawData_naval_np = np.loadtxt(os.path.join(self.data_dir, 'naval/data.txt'))
        # print(rawData_naval_np)
        X = rawData_naval_np[:, :-2]
        Y_compressor = rawData_naval_np[:, -2]
        Y_turbine = rawData_naval_np[:, -1]
        Y_all = rawData_naval_np[:, -2:]
        if Y_data == 'compressor':
            return X, Y_compressor
        elif Y_data == 'turbine':
            return X, Y_turbine
        elif Y_data == 'all':
            return X, Y_all
        else:
            return X, Y_compressor, Y_turbine

    ''' (6) Power '''
    def load_powerplant(self, Y_data='default'):
        rawData_powerplant_df = pd.read_excel(os.path.join(self.data_dir, 'power-plant/Folds5x2_pp.xlsx'), engine='openpyxl')
        X = rawData_powerplant_df.values[:, :-1]
        Y = rawData_powerplant_df.values[:, -1]
        return X, Y

    ''' (7) Protein '''
    def load_protein(self, Y_data='default'):
        rawData_protein_df = pd.read_csv(os.path.join(self.data_dir, 'protein/CASP.csv'), sep=',')
        X = rawData_protein_df.values[:, 1:]
        Y = rawData_protein_df.values[:, 0]
        return X, Y

    ''' (8) Wine '''
    def load_wine(self, Y_data='default'):
        rawData_wine_df = pd.read_csv(os.path.join(self.data_dir, 'wine-quality/winequality-red.csv'), sep=';')
        X = rawData_wine_df.values[:, :-1]
        Y = rawData_wine_df.values[:, -1]
        return X, Y

    ''' (9) yacht '''
    def load_yacht(self, Y_data='default'):
        rawData_yacht_np = np.loadtxt(os.path.join(self.data_dir, 'yacht/yacht_hydrodynamics.data'))
        X = rawData_yacht_np[:, :-1]
        Y = rawData_yacht_np[:, -1]
        return X, Y

    ''' (10) Song --- YearPredictionMSD dataset '''
    def load_MSD(self, Y_data='default'):
        # rawData_MSD_np = np.loadtxt(os.path.join(uci_dir, 'song/YearPredictionMSD.txt'), delimiter=',')
        # with open(os.path.join(uci_dir, 'song/YearPredictionMSD.npy'), 'wb') as f:
        #     np.save(f, rawData_MSD_np)
        with open(os.path.join(self.data_dir, 'song/YearPredictionMSD.npy'), 'rb') as f:
            rawData_MSD_np = np.load(f)
        X = rawData_MSD_np[:, :-1]
        Y = rawData_MSD_np[:, -1]
        return X, Y

    def load_single_dataset(self, name, Y_data='default'):
        load_funs = {
            'boston': self.load_boston,
            'concrete': self.load_concrete,
            'energy': self.load_energy_efficiency,
            'kin8nm': self.load_kin8nm,
            'naval': self.load_naval,
            'powerplant': self.load_powerplant,
            'protein': self.load_protein,
            'wine': self.load_wine,
            'yacht': self.load_yacht,
            'MSD': self.load_MSD
        }
        if name == 'energy':
            if Y_data == 'default':
                X, Y_heating, Y_cooling = load_funs[name](Y_data=Y_data)
                return X, Y_heating, Y_cooling
        elif name == 'naval':
            if Y_data == 'default':
                X, Y_compressor, Y_turbine = load_funs[name](Y_data=Y_data)
                return X, Y_compressor, Y_turbine
        else:
            X, Y = load_funs[name](Y_data=Y_data)
            return X, Y

    def load_streamflow_timeseries(self, configs, return_original_ydata=False):

        data_name = configs['data_name']
        num_inputs = configs['num_inputs']
        logtrans = configs['ylog_trans']
        plot_origin = configs['plot_origin']
        inputs_smoothing = configs['inputs_smoothing']
        # Name = ['Bradley','Copper','Gothic','Qui','Rock','RUS','EAQ','ph']
        # Ndata = [716, 848, 776, 774, 1131, 805, 774, 1096]
        # area = [881.6728, 5340.8252, 202.2172, 576.4915, 799.9965, 3340.0724, 1191.1487, 19126.0944]*0.042 
        # column name = ['Preciptation','Tmax','Tmin','Model_simulation','streamflow']
        # streamfow = streamflow/area
        # Use first 3 or 4 inputs to predict streamflow, where the streamflow is calcualted as the 5th column values/area
        # Save the last 365 days as testing data, use the remaining data as training
        rawData_np = np.loadtxt(os.path.join(self.data_dir, data_name+'.dat'))

        print(data_name)
        # print(rawData_np)
        print('-- Raw data shape: {}'.format(rawData_np.shape))
        print('-- Num of inputs: {}'.format(num_inputs))


        if inputs_smoothing:
            from tsmoothie.smoother import LowessSmoother, ExponentialSmoother, ConvolutionSmoother
            # from tsmoothie.utils_func import sim_seasonal_data

            smoother = ConvolutionSmoother(window_len=50, window_type='ones')
            smoother.smooth(rawData_np[:, 0:5].T)

            # generate intervals
            # low, up = smoother.get_intervals('prediction_interval')
            low, up = smoother.get_intervals('sigma_interval')
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=5, ncols=2, figsize=(20,16))
            # ax1.plot(rawData_np[:, 0])
            ax1.plot(smoother.data[0], label='Preciptation')
            ax2.plot(smoother.smooth_data[0], label='Preciptation (smoothed)')

            ax3.plot(rawData_np[:, 1], label='MaxT')
            ax4.plot(smoother.smooth_data[1], label='MaxT (smoothed)')
            ax5.plot(rawData_np[:, 2], label='MinT')
            ax6.plot(smoother.smooth_data[2], label='MinT (smoothed)')

            ax7.plot(rawData_np[:, 3], label='Model_simulation')
            ax9.plot(rawData_np[:, 4], label='Stream flow') 

            ax8.plot(smoother.smooth_data[3], label='Model_simulation (smoothed')
            ax10.plot(smoother.smooth_data[4], label='Stream flow (smoothed)') 

            ax1.legend();ax2.legend();ax3.legend();ax4.legend();ax5.legend();ax6.legend();ax7.legend();ax8.legend();ax9.legend();ax10.legend() 
            plt.suptitle('Original data (left) and the smoothed data (right) for {} \n smoother = ConvolutionSmoother(window_len=100, window_type=\'ones\')'.format(data_name))
            plt.show()

        if plot_origin:
            # import matplotlib.pyplot as plt
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(20,16))

            ax1.plot(rawData_np[:, 0], label='Preciptation')
            ax2.plot(rawData_np[:, 1], label='MaxT')
            ax3.plot(rawData_np[:, 2], label='MinT')
            # ax4.plot(rawData_np[:, 3], label='Model_simulation')
            ax4.plot(rawData_np[:, 4], label='Stream flow') 

            ax1.set_title('Precipitation')
            ax2.set_title('MaxT')
            ax3.set_title('MinT')
            # ax4.set_title('Stream flow (model simulation)')
            ax4.set_title('Stream flow')

            ax1.set_xlabel('Time (days)')
            ax2.set_xlabel('Time (days)')
            ax3.set_xlabel('Time (days)')
            # ax4.set_xlabel('Time (days)')
            ax4.set_xlabel('Time (days)')

            plt.suptitle('Original data for {}'.format(data_name))
            plt.tight_layout()
            plt.savefig(configs['project']+configs['exp']+'/'+configs['save_encoder_folder']+'/'+'origin.png')
            # plt.show()
            plt.close()

        ndays = configs['ndays']     # 60 #60 # 45 # 60 for Rock; 45 for Qui
        nfuture = configs['nfuture'] # 1 
        ninputs = configs['num_inputs'] # num_inputs
        nobs = ndays * ninputs
        Ntest = configs['ntest']   # 365 # 365
        Nvalid = configs['nvalid'] # 60

        Name_list = ['Bradley','Copper','Gothic','Qui','Rock','RUS','EAQ','ph']
        area_list = [x*0.042 for x in [881.6728, 5340.8252, 202.2172, 576.4915, 799.9965, 3340.0724, 1191.1487, 19126.0944]]
        area = area_list[Name_list.index(data_name)]

        xtmp = rawData_np[:,0:ninputs]
        ytmp = rawData_np[:,-nfuture]
        ytmp = ytmp/area

        if logtrans:
            ytmp = np.log(ytmp)

        ytmp = np.reshape(ytmp, (-1, 1))
        # print('max of obs (in/d)', np.max(ytmp))

        noutputs = ytmp.shape[1]
        reframed = self.series_to_supervised(xtmp, ytmp, ndays, nfuture)
        XYdata = reframed.values


        ##### New data splitting: first prepare the test/train data, and SHUFFLE split the train/validation data
        print('-- Reframed XYdata shape: {}'.format(XYdata.shape))
        print('-- pre-frame xtmp shape: {}'.format(xtmp.shape))
        print('-- pre-frame ytmp shape: {}'.format(ytmp.shape))
        print('-- pre-frame ndays: {}'.format(ndays))
        print('-- pre-frame nfuture: {}'.format(nfuture))
        ### split the testing data

        ### shuffle splitting train/valid data
        ### manual shuffle
        ### first choose the Y values and the corresponding index
        ### then slice the X values according to the index

        ### np.random.shuffle(list or numpy array)
        ### first split the train and test in chunk
        XYtrain = XYdata[:-Ntest, :]  # remove the last Ntest rows
        XYtest = XYdata[-Ntest:, :]   # keep the last Ntest rows

        print('-- XYtrain shape: {}'.format(XYtrain.shape))   ### (320, 136)   136 = 45 (lookback days) * 3 (input features) + 1 (output features)
        print('-- XYtest shape: {}'.format(XYtest.shape))     ### (365, 136)
        print('-- nobs: {}'.format(nobs))  ##       nobs = ndays * ninputs     ### 45*3 = 135

        ## train
        Xtrain = XYdata[:-Ntest, :nobs]         ### remove the last Ntest rows, and pick first 'nobs' cols 
        yobs_train = XYdata[:-Ntest, -nfuture]  ### remove the last Ntest rows, and pick last col
        print('-- yobs_train shape: {}'.format(yobs_train.shape))
        yobs_train = yobs_train.reshape(-1, 1)
        print('-- yobs_train shape: {}'.format(yobs_train.shape))

        ### test
        Xtest = XYdata[-Ntest:, :nobs]
        yobs_test = XYdata[-Ntest:, -nfuture]
        print('-- yobs_test shape: {}'.format(yobs_test.shape))
        yobs_test = yobs_test.reshape(-1, 1)
        print('-- yobs_test shape: {}'.format(yobs_test.shape))

        # exit()

        ##### Split into train/valid/test datasets
        ### test manual split for testing
        # xTrain, xValid = Xtrain[:-60, :], Xtrain[-60:, :]
        # yTrain, yValid = yobs_train[:-60, :], yobs_train[-60:, :]

        xTrain, xValid, yTrain, yValid = train_test_split(pd.DataFrame(Xtrain), pd.DataFrame(yobs_train), test_size=60, random_state=0)

        xTrain_idx = xTrain.index
        xValid_idx = xValid.index

        xTest = Xtest
        yTest = yobs_test

        print('--- xTrain, yTrain shape: {}, {}'.format(xTrain.shape, yTrain.shape))
        print('--- xValid, yValid shape: {}, {}'.format(xValid.shape, yValid.shape))
        print('--- xTest, yTest shape: {}, {}'.format(xTest.shape, yTest.shape))
        # print(yTrain)
        # exit()

        #### to float32
        xTrain = xTrain.astype(np.float32)
        yTrain = yTrain.astype(np.float32)
        xValid = xValid.astype(np.float32)
        yValid = yValid.astype(np.float32)
        xTest = xTest.astype(np.float32)
        yTest = yTest.astype(np.float32)

        ### keep the original y data without scaling
        ori_yTrain = yTrain   
        ori_yValid = yValid
        ori_yTest = yTest

        # ---scale training data both X and y
        # scalerx = MinMaxScaler(feature_range=(0, 1))
        scalerx = StandardScaler()
        xTrain = scalerx.fit_transform(xTrain)
        xValid = scalerx.transform(xValid)
        xTest = scalerx.transform(xTest)

        # scalery = MinMaxScaler(feature_range=(0, 1))
        scalery = StandardScaler()
        yTrain = scalery.fit_transform(yTrain)
        yValid = scalery.transform(yValid)
        yTest = scalery.transform(yTest)

        # reshape input to be 3D [samples, timesteps, features]
        xTrain = xTrain.reshape((xTrain.shape[0], ndays, ninputs))
        xValid = xValid.reshape((xValid.shape[0], ndays, ninputs))
        xTest = xTest.reshape((xTest.shape[0], ndays, ninputs))

        print('--- xTrain shape: {}'.format(xTrain.shape))
        print('--- xValid shape: {}'.format(xValid.shape))
        print('--- xTest shape: {}'.format(xTest.shape))

        if return_original_ydata:
            return xTrain, xValid, xTest, yTrain, yValid, yTest, ninputs, noutputs, scalerx, scalery, ori_yTrain, ori_yValid, ori_yTest, xTrain_idx, xValid_idx
        else:
            return xTrain, xValid, xTest, yTrain, yValid, yTest, ninputs, noutputs, scalerx, scalery


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


    # --- convert series to supervised learning
    def series_to_supervised(self, data_in, data_out, n_in=1, n_out=1, dropnan=True):
        n_vars_in = 1 if type(data_in) is list else data_in.shape[1]
        n_vars_out = 1 if type(data_out) is list else data_out.shape[1]
        df_in = pd.DataFrame(data_in)
        df_out = pd.DataFrame(data_out)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df_in.shift(i))
            names += [('var_in%d(t-%d)' % (j+1, i)) for j in range(n_vars_in)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df_out.shift(-i))
            if i == 0:
                names += [('var_out%d(t)' % (j+1)) for j in range(n_vars_out)]
            else:
                names += [('var_out%d(t+%d)' % (j+1, i)) for j in range(n_vars_out)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
