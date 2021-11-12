
''' Benchmark datasets loading and synthetic data generating '''

import os
import numpy as np
import pandas as pd

class CL_dataLoader:
    def __init__(self, original_data_path):
        # current_dir = os.path.dirname(__file__)
        # self.uci_dir = os.path.join(current_dir, 'UCI_datasets')
        self.uci_dir = original_data_path

    ''' (1) Boston '''
    def load_boston(self, Y_data='default'):
        rawData_boston_np = np.loadtxt(os.path.join(self.uci_dir, 'boston-housing/boston_housing.txt'))
        X = rawData_boston_np[:, :-1]
        Y = rawData_boston_np[:, -1]
        return X, Y

    ''' (2) Concrete '''
    def load_concrete(self, Y_data='default'):
        rawData_concrete_df = pd.read_excel(os.path.join(self.uci_dir, 'concrete/Concrete_Data.xls'))
        X = rawData_concrete_df.values[:, :-1]
        Y = rawData_concrete_df.values[:, -1]
        print(Y)
        return X, Y

    ''' (3) Energy (energy efficiency) '''
    def load_energy_efficiency(self, Y_data='default'):
        rawData_energy_df = pd.read_excel(os.path.join(self.uci_dir, 'energy-efficiency/ENB2012_data.xlsx'),
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
        rawData_kin8nm_df = pd.read_csv(os.path.join(self.uci_dir, 'kin8nm/dataset_2175_kin8nm.csv'), sep=',')
        X = rawData_kin8nm_df.values[:, :-1]
        Y = rawData_kin8nm_df.values[:, -1]
        return X, Y

    ''' (5) Naval '''
    def load_naval(self, Y_data='default'):
        rawData_naval_np = np.loadtxt(os.path.join(self.uci_dir, 'naval/data.txt'))
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
        rawData_powerplant_df = pd.read_excel(os.path.join(self.uci_dir, 'power-plant/Folds5x2_pp.xlsx'), engine='openpyxl')
        X = rawData_powerplant_df.values[:, :-1]
        Y = rawData_powerplant_df.values[:, -1]
        return X, Y

    ''' (7) Protein '''
    def load_protein(self, Y_data='default'):
        rawData_protein_df = pd.read_csv(os.path.join(self.uci_dir, 'protein/CASP.csv'), sep=',')
        X = rawData_protein_df.values[:, 1:]
        Y = rawData_protein_df.values[:, 0]
        return X, Y

    ''' (8) Wine '''
    def load_wine(self, Y_data='default'):
        rawData_wine_df = pd.read_csv(os.path.join(self.uci_dir, 'wine-quality/winequality-red.csv'), sep=';')
        X = rawData_wine_df.values[:, :-1]
        Y = rawData_wine_df.values[:, -1]
        return X, Y

    ''' (9) yacht '''
    def load_yacht(self, Y_data='default'):
        rawData_yacht_np = np.loadtxt(os.path.join(self.uci_dir, 'yacht/yacht_hydrodynamics.data'))
        X = rawData_yacht_np[:, :-1]
        Y = rawData_yacht_np[:, -1]
        return X, Y

    ''' (10) Song --- YearPredictionMSD dataset '''
    def load_MSD(self, Y_data='default'):
        # rawData_MSD_np = np.loadtxt(os.path.join(uci_dir, 'song/YearPredictionMSD.txt'), delimiter=',')
        # with open(os.path.join(uci_dir, 'song/YearPredictionMSD.npy'), 'wb') as f:
        #     np.save(f, rawData_MSD_np)
        with open(os.path.join(self.uci_dir, 'song/YearPredictionMSD.npy'), 'rb') as f:
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

    def standardizer_2(self, input_np):
        input_mean = input_np.mean(axis=0, keepdims=1)
        input_std = input_np.std(axis=0, keepdims=1)
        # input_std[input_std < 1e-10] = 1.0
        # standardized_input = (input_np - input_mean) / input_std
        return input_mean, input_std

    def getNumInputsOutputs(self, inputsOutputs_np):
        if len(inputsOutputs_np.shape) == 1:
            numInputsOutputs = 1
        if len(inputsOutputs_np.shape) > 1:
            numInputsOutputs = inputsOutputs_np.shape[1]
        return numInputsOutputs



    ''' load the saved train/test data files '''
    def LoadData_Splitted_UCI(self, loadCSVName, splitted_data_path, split_seed, **kwargs):

        xyTrain_load = np.loadtxt(splitted_data_path+'xyTrain_'+loadCSVName+'_seed_'+str(split_seed)+'.csv', delimiter=',')
        xyTest_load = np.loadtxt(splitted_data_path+'/xyTest_'+loadCSVName+'_seed_'+str(split_seed)+'.csv', delimiter=',')
        xyTrain_load = xyTrain_load.astype(np.float32)
        xyTest_load = xyTest_load.astype(np.float32)
        return xyTrain_load, xyTest_load


