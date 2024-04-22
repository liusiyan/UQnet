
''' Benchmark datasets loading and synthetic data generating '''

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class CL_BaseDataLoader:
    def __init__(self, config):
        self.config = config
        self.data = None

    def load_data(self):
        raise NotImplementedError
    
    def preprocess_data(self):
        if 'preprocessing' in self.config:
            for process in self.config['preprocessing']:
                if process['type'] == 'normalize':
                    self.data = self.normalize_data(self.data, process['columns'])
                elif process['type'] == 'standardize':
                    self.data = self.standardize_data(self.data, process['columns'])
                else:
                    raise ValueError('Unknown preprocessing method, please double check!!!')
    
    def get_data(self):
        if self.data is None:
            self.load_data()
            self.preprocess_data()
        return self.data
    




### test dataloader for Boston housing dataset
### User definition of how the data should be loaded and preprocessed
class CL_BostonDataLoader(CL_BaseDataLoader):
    def __init__(self, config):
        super(CL_BostonDataLoader, self).__init__(config)
    
    def load_data(self):
        self.data = pd.read_csv(self.config['data_path'])
    
    def preprocess_data(self):
        pass

    # def normalize_data(self, data, columns):
    #     data[columns] = MinMaxScaler().fit_transform(data[columns])
    #     return data
    
    def standardize_data(self, data, columns):
        data[columns] = StandardScaler().fit_transform(data[columns])
        return data
    
class CL_CSVDataLoader(CL_BaseDataLoader):
    def __init__(self, config):
        super(CL_CSVDataLoader, self).__init__(config)
    
    def load_data(self):
        self.data = pd.read_csv(self.config['data_path'])
    
    def preprocess_data(self):
        pass





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


    def load_ph_timeseries(self):
        rawData_ph_np = np.loadtxt(os.path.join(self.data_dir, 'ph.dat'))
        # print(rawData_ph_np)
        # print(rawData_ph_np.shape)
        ndays = 60  
        nfuture = 1 
        ninputs = 3
        nobs = ndays * ninputs
        Ntest = 365

        area = 19126.0944*0.042

        xtmp = rawData_ph_np[:,0:ninputs]
        ytmp = rawData_ph_np[:,-nfuture]
        ytmp = ytmp/area
        ytmp = np.reshape(ytmp, (-1, 1))

        reframed = self.series_to_supervised(xtmp, ytmp, ndays, nfuture)
        print('Shape of supervised dataset: ', np.shape(reframed))

        XYdata = reframed.values
        # 1.2 ---split into train and test sets
        # 10/1/14-9/30/16 for training, and 10/1/17-9/30/17 for testing
        XYtrain = XYdata[:-Ntest, :]
        XYtest = XYdata[-Ntest:, :]

        Xtrain,yobs_train = XYdata[:-Ntest, :nobs], XYdata[:-Ntest, -nfuture].reshape(-1, 1)
        Xtest,yobs_test = XYdata[-Ntest:, :nobs], XYdata[-Ntest:, -nfuture].reshape(-1, 1)
        print('shape of yobs_train and yobs_test is ', yobs_train.shape, yobs_test.shape)

        Ntrain = len(yobs_train)

        # 1.3 ---scale training data both X and y
        scalerx = MinMaxScaler(feature_range=(0, 1))
        train_X = scalerx.fit_transform(Xtrain)
        test_X = scalerx.transform(Xtest)

        scalery = MinMaxScaler(feature_range=(0, 1))
        train_y = scalery.fit_transform(yobs_train)
        test_y = scalery.transform(yobs_test)
        print('shape of train_X, train_y, test_X, and test_y: ', train_X.shape, train_y.shape, test_X.shape, test_y.shape)

        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], ndays, ninputs))
        test_X = test_X.reshape((test_X.shape[0], ndays, ninputs))
        print('shape of train_X and test_X in 3D: ', train_X.shape, test_X.shape)

        # train_X_t = torch.Tensor(train_X)
        # test_X_t = torch.Tensor(test_X)
        # train_y_t = torch.Tensor(train_y)
        # test_y_t = torch.Tensor(test_y)





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