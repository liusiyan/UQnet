
''' Benchmark datasets loading and synthetic data generating '''

import os
import numpy as np
import pandas as pd

class CL_dataLoader:
    def __init__(self, original_data_path, configs):
        # current_dir = os.path.dirname(__file__)
        # self.data_dir = os.path.join(current_dir, 'UCI_datasets')
        self.data_dir = original_data_path
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



    def load_flight_delays_df_2(self):
        df_2008 = pd.read_hdf(os.path.join(self.data_dir, '2008.h5'), 'df')
        # print(df_2008)
        print(df_2008.head(3).T)

        df_reduced = df_2008[['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime', 'ActualElapsedTime', 'CRSElapsedTime', 
        'AirTime', 'ArrDelay', 'DepDelay', 'Origin', 'Dest', 'Distance', 'TaxiIn', 'TaxiOut']]

        # print(df_reduced)
        # print(df_reduced.head(3).T)

        # ## remove the lines with missing data (NaN) in dataFrame
        # nonnan_df = ~np.isnan(df_reduced).any(axis=1)   ## for Numpy array
        # nonnan_df = df_reduced.dropna()

        # print(nonnan_df.duplicated(['Origin']))

        dudf_origin = df_reduced.drop_duplicates(subset='Origin')
        origin_list = dudf_origin['Origin'].values
        # print(dudf_origin['Origin'])



        ''' Similar to the FAA https://www.faa.gov/airports/planning_capacity/categories/         
        we separate the data into 4 ranks based on the percentage of departure flights:        
        --- Large Hub: Receives 1% or more of the annual departures         
        --- Medium Hub: Receives 0.25% - 1.0% of the annual departures         
        --- Small Hub: 0.05% - 0.25%         
        --- Nonhub, nonprimary: <0.05%         
        '''


        # ### get a rank of origin based on the number of airport departures 
        # dep_num_list = []
        # percentage_list = []
        # rank_list = []
        # total_dep = len(df_reduced)
        # for i in range(len(origin_list)):
        #     df_tmp = df_reduced[df_reduced['Origin'] == origin_list[i]]
        #     dep_num = len(df_tmp)

        #     percentage = dep_num/total_dep
        #     if percentage > 0.01:
        #         rank = 1
        #     elif percentage > 0.0025 and percentage <= 0.01:
        #         rank = 2
        #     elif percentage > 0.0005 and percentage <= 0.0025:
        #         rank = 3
        #     elif percentage <= 0.0005:
        #         rank = 4

        #     rank_list.append(rank)

        #     print('--- Id: {}, Origin: {}, dep_num: {}, percentage: {:.3f} %, rank: {}'.format(i+1, origin_list[i], dep_num, dep_num/total_dep*100, rank))

        #     dep_num_list.append(dep_num)
        #     percentage_list.append(dep_num/total_dep)

        # df_dep_num = pd.DataFrame({
        #     'Origin': origin_list,
        #     'dep_num': dep_num_list,
        #     'percentage': percentage_list,
        #     'rank': rank_list
        #     }) 

        # df_dep_num = df_dep_num.sort_values(by=['dep_num'], ascending=False) ### sorting data 
        # df_dep_num.to_csv('Airport_rank.csv')


        df_dep_num = pd.read_csv(os.path.join(self.data_dir,'Airport_rank.csv'))

        print(len(df_dep_num[df_dep_num['rank'] == 1]))   ## 28  airports,  4292050 departures
        print(len(df_dep_num[df_dep_num['rank'] == 2]))   ## 44  airports,  1741620 departures
        print(len(df_dep_num[df_dep_num['rank'] == 3]))   ## 93  airports,  787416 departures
        print(len(df_dep_num[df_dep_num['rank'] == 4]))   ## 138 airports,  188642 departures



        rank_list = []
        for i in range(4):
            rank_list.append(df_dep_num[df_dep_num['rank'] == (i+1)]['Origin'].values)


        test_1_labels = df_dep_num[df_dep_num['rank'] == 1]['Origin'].values
        test_2_labels = df_dep_num[df_dep_num['rank'] == 2]['Origin'].values
        test_3_labels = df_dep_num[df_dep_num['rank'] == 3]['Origin'].values
        test_4_labels = df_dep_num[df_dep_num['rank'] == 4]['Origin'].values



        ### (Optional) remove some features
        # [['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime', 'ActualElapsedTime', 'CRSElapsedTime', 
        # 'AirTime', 'ArrDelay', 'DepDelay', 'Origin', 'Dest', 'Distance', 'TaxiIn', 'TaxiOut']]
        df_reduced = df_reduced[['DayofMonth', 'DayOfWeek', 'AirTime', 'ArrDelay', 'Origin', 'Dest', 'Distance', 'TaxiIn', 'TaxiOut', 'DepDelay']]


        ### separate data based on the labels
        # df_train = df_reduced[df_reduced['Origin'].isin(train_labels)]
        df_test_1 = df_reduced[df_reduced['Origin'].isin(test_1_labels)]
        df_test_2 = df_reduced[df_reduced['Origin'].isin(test_2_labels)]
        df_test_3 = df_reduced[df_reduced['Origin'].isin(test_3_labels)]
        df_test_4 = df_reduced[df_reduced['Origin'].isin(test_4_labels)]

        ### drop NaN in the data
        # df_train = df_train.dropna()
        df_test_1 = df_test_1.dropna()  # len = 4,196,851
        df_test_2 = df_test_2.dropna()  # len = 1,710,108
        df_test_3 = df_test_3.dropna()  # len = 765,398
        df_test_4 = df_test_4.dropna()  # len = 182,672


        input_features = ['DayofMonth', 'DayOfWeek', 'AirTime', 'Distance', 'DepDelay', 'TaxiOut']  # 'TaxiIn'
        output_features = ['ArrDelay']


        #######################################
        ## Customize the training/testing data sets

        print('TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT')
        print('-- rank 1 max: {}, min: {}, mean: {}'.format(df_test_1['TaxiOut'].max(), df_test_1['TaxiOut'].min(), df_test_1['TaxiOut'].mean()))
        print('-- rank 2 max: {}, min: {}, mean: {}'.format(df_test_2['TaxiOut'].max(), df_test_2['TaxiOut'].min(), df_test_2['TaxiOut'].mean()))
        print('-- rank 3 max: {}, min: {}, mean: {}'.format(df_test_3['TaxiOut'].max(), df_test_3['TaxiOut'].min(), df_test_3['TaxiOut'].mean()))
        print('-- rank 4 max: {}, min: {}, mean: {}'.format(df_test_4['TaxiOut'].max(), df_test_4['TaxiOut'].min(), df_test_4['TaxiOut'].mean()))

        # (1) find smaller TaxiIn data in rank 4 as the training data
        df_test_4_sorted = df_test_4.sort_values(by=['TaxiOut'], ascending=True) # len = 182,672

        df_test_4_for_train = df_test_4_sorted.iloc[0:int(len(df_test_4_sorted)*0.05)] # len = 18,267, 10% of test 4
        # df_test_4_for_train = df_test_4_sorted.iloc[0:10000] # len = 18,267, 10% of test 4
        df_test_4_for_train = df_test_4_for_train.sample(frac=1, random_state=1).reset_index(drop=True)   # shuffle
        # print(df_test_4_for_train)
        print('-- len of rank 4 for train: {}'.format(len(df_test_4_for_train)))
        print('-- rank 4 for train max: {}, min: {}, mean: {}'.format(df_test_4_for_train['TaxiOut'].max(), df_test_4_for_train['TaxiOut'].min(), df_test_4_for_train['TaxiOut'].mean()))

        # print(len(df_test_4_for_train))

        df_test_4_new = df_test_4_sorted.iloc[int(len(df_test_4_sorted)*0.5):int(len(df_test_4_sorted)*0.55)]  # len = 18,267, 10% of test 4
        # df_test_4_new = df_test_4_sorted.iloc[90000:100000]  # len = 18,267, 10% of test 4
        df_test_4_new = df_test_4_new.sample(frac=1, random_state=1).reset_index(drop=True)   # shuffle
        print('-- len of new rank 4: {}'.format(len(df_test_4_new)))
        print('-- rank 4 new max: {}, min: {}, mean: {}'.format(df_test_4_new['TaxiOut'].max(), df_test_4_new['TaxiOut'].min(), df_test_4_new['TaxiOut'].mean()))


        # (2) find relatively larger TaxiIn data in rank 1,2,3 as the testing data
        df_test_1_sorted = df_test_1.sort_values(by=['TaxiOut'], ascending=False)
        df_test_1_new = df_test_1_sorted.iloc[int(len(df_test_1_sorted)*0.1):int(len(df_test_1_sorted)*0.15)]
        # df_test_1_new = df_test_1_sorted.iloc[400000:410000]
        df_test_1_new = df_test_1_new.sample(frac=1, random_state=1).reset_index(drop=True)   # shuffle
        print('-- len of new rank 1: {}'.format(len(df_test_1_new)))
        print('-- rank 1 new max: {}, min: {}, mean: {}'.format(df_test_1_new['TaxiOut'].max(), df_test_1_new['TaxiOut'].min(), df_test_1_new['TaxiOut'].mean()))

        df_test_2_sorted = df_test_2.sort_values(by=['TaxiOut'], ascending=False)
        df_test_2_new = df_test_2_sorted.iloc[0:int(len(df_test_2_sorted)*0.05)]
        # df_test_2_new = df_test_2_sorted.iloc[0:10000]
        df_test_2_new = df_test_2_new.sample(frac=1, random_state=1).reset_index(drop=True)   # shuffle
        print('-- len of new rank 2: {}'.format(len(df_test_2_new)))
        print('-- rank 2 new max: {}, min: {}, mean: {}'.format(df_test_2_new['TaxiOut'].max(), df_test_2_new['TaxiOut'].min(), df_test_2_new['TaxiOut'].mean()))

        df_test_3_sorted = df_test_3.sort_values(by=['TaxiOut'], ascending=False)
        df_test_3_new = df_test_3_sorted.iloc[0:int(len(df_test_3_sorted)*0.05)]
        # df_test_3_new = df_test_3_sorted.iloc[0:10000]
        df_test_3_new = df_test_3_new.sample(frac=1, random_state=1).reset_index(drop=True)   # shuffle
        print('-- len of new rank 3: {}'.format(len(df_test_3_new)))
        print('-- rank 3 new max: {}, min: {}, mean: {}'.format(df_test_3_new['TaxiOut'].max(), df_test_3_new['TaxiOut'].min(), df_test_3_new['TaxiOut'].mean()))


        # (3) re-define the train and 4 testing data
        df_train = df_test_4_for_train
        df_test_1 = df_test_4_new   # rank 4
        df_test_2 = df_test_1_new   # rank 3
        df_test_3 = df_test_2_new   # rank 2
        df_test_4 = df_test_3_new   # rank 1


        xTrain = df_train[input_features].values
        yTrain = df_train[output_features].values.flatten()
        xTest_list = [df_test_1[input_features].values,  df_test_2[input_features].values,  df_test_3[input_features].values,  df_test_4[input_features].values]
        yTest_list = [df_test_1[output_features].values.flatten(), df_test_2[output_features].values.flatten(), df_test_3[output_features].values.flatten(), df_test_4[output_features].values.flatten()]


        ### data normalization

        test_data_normalizaed_list = []
        xTrain_normalized, xTrain_mean, xTrain_std = self.standardizer(xTrain)
        yTrain_normalized, yTrain_mean, yTrain_std = self.standardizer(yTrain)

        for i in range(len(xTest_list)):
            xTest_normalized = (xTest_list[i] - xTrain_mean) / xTrain_std
            yTest_normalized = (yTest_list[i] - yTrain_mean) / yTrain_std
            test_data_normalizaed_list.append((xTest_normalized, yTest_normalized))

        return xTrain_normalized, yTrain_normalized, test_data_normalizaed_list


