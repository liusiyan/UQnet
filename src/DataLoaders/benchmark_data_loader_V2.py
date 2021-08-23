
''' Benchmark datasets loading and synthetic data generating '''

import os
import numpy as np
import pandas as pd

class CL_dataLoader:
    def __init__(self, original_data_path):
        # current_dir = os.path.dirname(__file__)
        # self.data_dir = os.path.join(current_dir, 'UCI_datasets')
        self.data_dir = original_data_path

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
            'energy-efficiency': self.load_energy_efficiency,
            'kin8nm': self.load_kin8nm,
            'naval': self.load_naval,
            'powerplant': self.load_powerplant,
            'protein': self.load_protein,
            'wine': self.load_wine,
            'yacht': self.load_yacht,
            'MSD': self.load_MSD
        }
        if name == 'energy-efficiency':
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



    ''' load the saved UCI train/test data files '''
    def LoadData_Splitted_UCI(self, loadCSVName, splitted_data_path, split_seed, **kwargs):
        xyTrain_load = np.loadtxt(splitted_data_path+'xyTrain_'+loadCSVName+'_seed_'+str(split_seed)+'.csv', delimiter=',')
        xyTest_load = np.loadtxt(splitted_data_path+'/xyTest_'+loadCSVName+'_seed_'+str(split_seed)+'.csv', delimiter=',')
        xyTrain_load = xyTrain_load.astype(np.float32)
        xyTest_load = xyTest_load.astype(np.float32)
        return xyTrain_load, xyTest_load


    ''' load the flight delays data sets for year 2008
    Training data is the first 700k data points
    5 testing sets with 100k data points starting from 700k, 2m, 3m, 4m, and 5m

    '''
    def load_flight_delays_df(self):
        df_2008 = pd.read_hdf(os.path.join(self.data_dir, '2008.h5'), 'df')

        # df_2008 = pd.read_csv(os.path.join(self.data_dir, '2008.csv'))
        # df_train = pd.read_hdf(os.path.join(self.data_dir, 'train.h5'), 'df')
        # df_test_1 = pd.read_hdf(os.path.join(self.data_dir, 'test_1.h5'), 'df')
        # df_test_2 = pd.read_hdf(os.path.join(self.data_dir, 'test_2.h5'), 'df')
        # df_test_3 = pd.read_hdf(os.path.join(self.data_dir, 'test_3.h5'), 'df')
        # df_test_4 = pd.read_hdf(os.path.join(self.data_dir, 'test_4.h5'), 'df')
        # df_test_5 = pd.read_hdf(os.path.join(self.data_dir, 'test_5.h5'), 'df')
        # return df_train, df_test_1, df_test_2, df_test_3, df_test_4, df_test_5

        X_data = df_2008[['Distance','Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'ArrTime', 'AirTime']].values
        Y_data = df_2008[['ArrDelay']].values

        xTrain = X_data[:700000, :]
        yTrain = Y_data[:700000, :]

        xTest_1 = X_data[700000:800000, :]
        yTest_1 = Y_data[700000:800000, :]

        xTest_2 = X_data[2000000:2100000, :] 
        yTest_2 = Y_data[2000000:2100000, :] 

        xTest_3 = X_data[3000000:3100000, :] 
        yTest_3 = Y_data[3000000:3100000, :] 

        xTest_4 = X_data[4000000:4100000, :] 
        yTest_4 = Y_data[4000000:4100000, :] 

        xTest_5 = X_data[5000000:5100000, :] 
        yTest_5 = Y_data[5000000:5100000, :] 


        ### remove the lines with missing data (NaN)
        nonnan_xTrain = ~np.isnan(xTrain).any(axis=1)
        nonnan_yTrain = ~np.isnan(yTrain).any(axis=1)
        nonnan_Train_bool = np.logical_and(nonnan_xTrain, nonnan_yTrain)
        xTrain = xTrain[nonnan_Train_bool, :]
        yTrain = yTrain[nonnan_Train_bool, :]
        yTrain = yTrain.flatten()

        nonnan_xTest_1 = ~np.isnan(xTest_1).any(axis=1)
        nonnan_yTest_1 = ~np.isnan(yTest_1).any(axis=1)
        nonnan_Test_1_bool = np.logical_and(nonnan_xTest_1, nonnan_yTest_1)
        xTest_1 = xTest_1[nonnan_Test_1_bool, :]
        yTest_1 = yTest_1[nonnan_Test_1_bool, :]
        yTest_1 = yTest_1.flatten()

        nonnan_xTest_2 = ~np.isnan(xTest_2).any(axis=1)
        nonnan_yTest_2 = ~np.isnan(yTest_2).any(axis=1)
        nonnan_Test_2_bool = np.logical_and(nonnan_xTest_2, nonnan_yTest_2)
        xTest_2 = xTest_2[nonnan_Test_2_bool, :]
        yTest_2 = yTest_2[nonnan_Test_2_bool, :]
        yTest_2 = yTest_2.flatten()

        nonnan_xTest_3 = ~np.isnan(xTest_3).any(axis=1)
        nonnan_yTest_3 = ~np.isnan(yTest_3).any(axis=1)
        nonnan_Test_3_bool = np.logical_and(nonnan_xTest_3, nonnan_yTest_3)
        xTest_3 = xTest_3[nonnan_Test_3_bool, :]
        yTest_3 = yTest_3[nonnan_Test_3_bool, :]
        yTest_3 = yTest_3.flatten()

        nonnan_xTest_4 = ~np.isnan(xTest_4).any(axis=1)
        nonnan_yTest_4 = ~np.isnan(yTest_4).any(axis=1)
        nonnan_Test_4_bool = np.logical_and(nonnan_xTest_4, nonnan_yTest_4)
        xTest_4 = xTest_4[nonnan_Test_4_bool, :]
        yTest_4 = yTest_4[nonnan_Test_4_bool, :]
        yTest_4 = yTest_4.flatten()

        nonnan_xTest_5 = ~np.isnan(xTest_5).any(axis=1)
        nonnan_yTest_5 = ~np.isnan(yTest_5).any(axis=1)
        nonnan_Test_5_bool = np.logical_and(nonnan_xTest_5, nonnan_yTest_5)
        xTest_5 = xTest_5[nonnan_Test_5_bool, :]
        yTest_5 = yTest_5[nonnan_Test_5_bool, :]
        yTest_5 = yTest_5.flatten()

        ### data normalization
        test_data_normalizaed_list = []
        test_data_list = [(xTest_1, yTest_1), (xTest_2, yTest_2), (xTest_3, yTest_3), (xTest_4, yTest_4), (xTest_5, yTest_5)]

        xTrain_normalized, xTrain_mean, xTrain_std = self.standardizer(xTrain)
        yTrain_normalized, yTrain_mean, yTrain_std = self.standardizer(yTrain)

        for i in range(len(test_data_list)):
            xTest_normalized = (test_data_list[i][0] - xTrain_mean) / xTrain_std
            yTest_normalized = (test_data_list[i][1] - yTrain_mean) / yTrain_std

            test_data_normalizaed_list.append((xTest_normalized, yTest_normalized))

        return xTrain_normalized, yTrain_normalized, test_data_normalizaed_list
 


    def load_flight_delay_EDL(self):
        # Download from here: http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/dataset_mirror/airline_delay/
        data = pd.read_pickle(os.path.join(self.data_dir, "filtered_data2.pickle"))

        # print(type(data))
        # print(data)
        # data = pd.read_pickle("data/flight-delay/filtered_data.pickle")
        y = np.array(data['ArrDelay'])
        data.pop('ArrDelay')
        X = np.array(data[:])
        # print(y)

        def standardize(data):
            data -= data.mean(axis=0, keepdims=1)
            scale = data.std(axis=0, keepdims=1)
            data /= scale
            return data, scale

        print('sssssssssssssssssssssssssssssssssssss')
        print(X)
        print(X.var(axis=0))
        print(np.where(data.var(axis=0) > 0)[0])

        X = X[:, np.where(data.var(axis=0) > 0)[0]]
        X, _ = standardize(X)
        y, y_scale = standardize(y.reshape(-1,1))
        y = np.squeeze(y)
        # y_scale = np.array([[1.0]])

        N = 700000
        S = 100000
        X_train = X[:N,:]
        X_test = X[N:N + S, :]
        y_train = y[:N]
        y_test = y[N:N + S]


        return X_train, y_train, X_test, y_test, y_scale