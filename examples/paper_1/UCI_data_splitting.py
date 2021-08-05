import numpy as np
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

''' 
This code is used to split the original UCI data sets with given splitting random seed.
In our study, five random seed 1, 2, 3, 4, 5 were used for the train/test splitting, and 0.1 as the testing fraction.
The same split data will be further used in all 3 methods for performance evaluation and comparison.

To obtain the split data, simply run this file after the original UCI data sets are ready (./UCI_datasets)
the results will be saved at './UCI_TrainTest_Split_test/' for testing.

We also prepared pre-split datasets for your reference (in ./UCI_TrainTest_Split/)

Have fun!
'''



dataset_list = ['boston', 'concrete', 'energy', 'kin8nm', 'naval', 'power', 'protein', 'wine', 'yacht', 'MSD']
for name in dataset_list:
    data_name = name
    print('--- Splitting data: "{}"'.format(data_name))

    ''' Data selection '''
    # data_name = 'concrete'
    original_data_path = './UCI_datasets/'          ## original UCI data sets
    data_save_path = './UCI_TrainTest_Split_test/'
    test_frac_seed_list = [1, 2, 3, 4, 5]
    test_fraction = 0.1

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

    ''' Prepare the train/test data to files '''
    dataLoader = CL_dataLoader(original_data_path)
    X_boston, Y_boston = dataLoader.load_boston(Y_data='default')
    X_concrete, Y_concrete = dataLoader.load_concrete(Y_data='default')
    X_energy, Y_energy_all = dataLoader.load_energy_efficiency(Y_data='all')
    X_kin8nm, Y_kin8nm = dataLoader.load_kin8nm(Y_data='default')
    X_naval, Y_naval_all = dataLoader.load_naval(Y_data='all')
    X_power, Y_power = dataLoader.load_powerplant(Y_data='default')
    X_protein, Y_protein = dataLoader.load_protein(Y_data='default')
    X_wine, Y_wine = dataLoader.load_wine(Y_data='default')
    X_yacht, Y_yacht = dataLoader.load_yacht(Y_data='default')
    X_MSD, Y_MSD = dataLoader.load_MSD(Y_data='default')

    if data_name == 'boston':
        X_data = X_boston; Y_data = Y_boston
    elif data_name == 'concrete':
        X_data = X_concrete; Y_data = Y_concrete
    elif data_name == 'energy':
        X_data = X_energy; Y_data = Y_energy_all
    elif data_name == 'kin8nm':
        X_data = X_kin8nm; Y_data = Y_kin8nm
    elif data_name == 'naval':
        X_data = X_naval; Y_data = Y_naval_all
    elif data_name == 'power':
        X_data = X_power; Y_data = Y_power
    elif data_name == 'protein':
        X_data = X_protein; Y_data = Y_protein
    elif data_name == 'wine':
        X_data = X_wine; Y_data = Y_wine
    elif data_name == 'yacht':
        X_data = X_yacht; Y_data = Y_yacht
    elif data_name == 'MSD':
        X_data = X_MSD; Y_data = Y_MSD

    X_data = X_data.astype(np.float32)
    Y_data = Y_data.astype(np.float32)
    loadCSVName = data_name

    ''' Hyper parameters '''
    for test_frac_seed in test_frac_seed_list:
        test_fraction_split_random_seed = test_frac_seed # 3
        bool_test_fraction_split_shuffle = True
        # valid_fraction = 0.3 ##
        # valid_fraction_split_random_seed = 1
        # bool_valid_fraction_split_shuffle = True
        seed = 1 # 2
        random.seed(seed)
        np.random.seed(seed)

        ''' Train/test data splitting '''
        xTrain, xTest, yTrain, yTest = train_test_split(X_data, Y_data, test_size=test_fraction,
                                                        random_state=test_fraction_split_random_seed,
                                                        shuffle=bool_test_fraction_split_shuffle)
        # xTrainAll, xTest, yTrainAll, yTest = train_test_split(X_data, Y_data, test_size=test_fraction,
        #                                                 random_state=test_fraction_split_random_seed,
        #                                                 shuffle=bool_test_fraction_split_shuffle)
        # ''' Train/validation splitting  '''
        # xTrain, xValid, yTrain, yValid = train_test_split(xTrainAll, yTrainAll, test_size=valid_fraction,
        #                                                   random_state=valid_fraction_split_random_seed,
        #                                                   shuffle=bool_valid_fraction_split_shuffle)
        # print('--- xTrain shape: {}'.format(xTrain.shape))
        # print('--- yTrain shape: {}'.format(yTrain.shape))
        # print('--- xValid shape: {}'.format(xValid.shape))
        # print('--- yValid shape: {}'.format(yValid.shape))
        # print('--- xTest shape: {}'.format(xTest.shape))
        # print('--- yTest shape: {}'.format(yTest.shape))


        ''' Save the splitted data set --- xTrain, xTest, yTrain, yTest '''
        if data_name == 'energy' or data_name == 'naval':
            xyTrain = np.concatenate((xTrain, yTrain), axis=1)
            xyTest = np.concatenate((xTest, yTest), axis=1)
        else:
            xyTrain = np.concatenate((xTrain, np.reshape(yTrain, (-1, 1))), axis=1)
            xyTest = np.concatenate((xTest, np.reshape(yTest, (-1, 1))), axis=1)
        pd.DataFrame(xyTrain).to_csv(data_save_path+'xyTrain_'+data_name+'_seed_'+str(test_fraction_split_random_seed)+'.csv', index=False, header=False)
        pd.DataFrame(xyTest).to_csv(data_save_path+'/xyTest_'+data_name+'_seed_'+str(test_fraction_split_random_seed)+'.csv', index=False, header=False)

        print('Saved'+data_save_path+'xyTrain_'+data_name+'_seed_'+str(test_fraction_split_random_seed)+'.csv')
        print('Saved'+data_save_path+'/xyTest_'+data_name+'_seed_'+str(test_fraction_split_random_seed)+'.csv')

print('Done !!!')