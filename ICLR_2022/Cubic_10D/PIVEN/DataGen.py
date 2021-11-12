"""
Data creation:
Load the data, normalize it, and split into train and test.
"""



'''
Added the capability of loading pre-separated UCI train/test data
function LoadData_Splitted_UCI

'''


import numpy as np
import os
import pandas as pd
import tensorflow as tf

DATA_PATH = "../UCI_Datasets"


class DataGenerator:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        # used for metrics calculation
        self.scale_c = None  # std
        self.shift_c = None  # mean

    def create_cubic_10D_data(self):

        Npar = 10
        Ntrain = 5000
        Nout = 1
        Ntest = 1000

        # x_train = tf.random.uniform(shape=(Ntrain, Npar))*4.0-2.0
        x_train = tf.random.normal(shape=(Ntrain, Npar))
        y_train = x_train ** 3
        y_train = tf.reduce_sum(y_train, axis=1, keepdims=True)/10.0 + 1.0*tf.random.normal([x_train.shape[0], 1])

        # x_test = tf.random.uniform(shape=(Ntest, Npar))
        # x_test[:,1] = x_test[:,1] + 4.0
        # x_test = np.random.uniform(size=(Ntest,Npar))
        # x_test[:,1] = x_test[:,1] + 4.0
        x_test = np.random.normal(size=(Ntest,Npar)) + 2.0
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        scale_c = np.std(x_test.eval(session=tf.compat.v1.Session()))
        y_test = x_test ** 3
        y_test = tf.reduce_sum(y_test, axis=1, keepdims=True)/10.0 + 1.0*tf.random.normal([x_test.shape[0], 1])


        ### to Numpy array in TF1 compat environment using TF2
        x_train = x_train.eval(session=tf.compat.v1.Session())
        y_train = y_train.eval(session=tf.compat.v1.Session())
        x_test = x_test.eval(session=tf.compat.v1.Session())
        y_test = y_test.eval(session=tf.compat.v1.Session())

        ### normalization
        x_mean = np.mean(x_train, axis=0)
        x_std = np.std(x_train,axis=0)
        xtrain_normal = (x_train - x_mean)/x_std

        y_mean = np.mean(y_train,axis=0)
        y_std = np.std(y_train,axis=0)
        ytrain_normal = (y_train - y_mean)/y_std

        xvalid_normal = (x_test - x_mean) / x_std
        yvalid_normal = (y_test - y_mean) / y_std

        X_train = xtrain_normal
        y_train = ytrain_normal         
        X_val = xvalid_normal
        y_val = yvalid_normal

        self.scale_c = scale_c

        return X_train, y_train, X_val, y_val

    def create_data(self, seed_in=5, train_prop=0.9):
        """
        @param seed_in: seed for numpy random seed
        @param train_prop: train proportion
        """
        np.random.seed(seed_in)

        # load UCI data
        dataset = self.dataset_name
        dataset_path = f"{DATA_PATH}/{dataset}.txt"

        if dataset == 'YearPredictionMSD':
            data = np.loadtxt(dataset_path, delimiter=',')
        elif dataset == 'naval':
            data = np.loadtxt(dataset_path)
            data = data[:, :-1]  # have 2 y as GT, ignore last
        else:
            data = np.loadtxt(dataset_path)

        # save normalization constants (used for calculating results)
        if dataset == 'YearPredictionMSD':
            scale_c = np.std(data[:, 0])  # in YearPredictionMSD, label's index = 0
            shift_c = np.mean(data[:, 0])
        else:
            scale_c = np.std(data[:, -1])
            shift_c = np.mean(data[:, -1])

        # normalize data
        for i in range(data.shape[1]):
            sdev_norm = np.std(data[:, i])
            sdev_norm = 0.001 if sdev_norm == 0 else sdev_norm  # avoid zero variance features
            data[:, i] = (data[:, i] - np.mean(data[:, i])) / sdev_norm

        # split train test
        if dataset == 'YearPredictionMSD':
            # train: first 463,715 examples
            # test: last 51,630 examples
            train = data[:463715, :]
            test = data[-51630:, :]

        else:
            # split into train/test in random
            perm = np.random.permutation(data.shape[0])
            train_size = int(round(train_prop * data.shape[0]))
            train = data[perm[:train_size], :]
            test = data[perm[train_size:], :]

        # split to target and data
        if dataset == 'YearPredictionMSD':
            y_train = train[:, 0].reshape(-1, 1)
            X_train = train[:, 1:]
            y_val = test[:, 0].reshape(-1, 1)
            X_val = test[:, 1:]

        else:
            y_train = train[:, -1].reshape(-1, 1)
            X_train = train[:, :-1]
            y_val = test[:, -1].reshape(-1, 1)
            X_val = test[:, :-1]

        self.scale_c = scale_c
        self.shift_c = shift_c

        return X_train, y_train, X_val, y_val


    def LoadData_Splitted_UCI(self, loadCSVName, original_data_path, splitted_data_path, split_seed, **kwargs):

        ## (1) Load the original data for the normalization purpose
        # current_dir = os.path.dirname(__file__)
        # uci_dir = os.path.join(current_dir, 'UCI_datasets')
        uci_dir = original_data_path
        if loadCSVName == 'boston':
            data = np.loadtxt(os.path.join(uci_dir, 'boston-housing/boston_housing.txt'))

        if loadCSVName == 'concrete':
            data_df = pd.read_excel(os.path.join(uci_dir, 'concrete/Concrete_Data.xls'))
            data = data_df.values

        if loadCSVName == 'energy':
            data_df = pd.read_excel(os.path.join(uci_dir, 'energy-efficiency/ENB2012_data.xlsx'), engine='openpyxl')
            data_df = data_df.dropna(how='all', axis='columns')
            data_df = data_df.dropna(how='all', axis='rows')
            data = data_df.values


        if loadCSVName == 'kin8nm':
            data_df = pd.read_csv(os.path.join(uci_dir, 'kin8nm/dataset_2175_kin8nm.csv'), sep=',')
            data = data_df.values

        if loadCSVName == 'naval':
            data = np.loadtxt(os.path.join(uci_dir, 'naval/data.txt'))

        if loadCSVName == 'power':
            data_df = pd.read_excel(os.path.join(uci_dir, 'power-plant/Folds5x2_pp.xlsx'), engine='openpyxl')
            data = data_df.values

        if loadCSVName == 'protein':
            data_df = pd.read_csv(os.path.join(uci_dir, 'protein/CASP.csv'), sep=',')
            # print(data_df)
            '''Move the Y data (originally located at the first column) to last column in order to keep consistency
            with the normalization process'''
            col_names = data_df.columns.tolist()
            col_names.append(col_names[0])
            del col_names[col_names.index(col_names[0])]
            # print(col_names)
            data_df = data_df[col_names]
            # print(data_df)
            data = data_df.values

        if loadCSVName == 'wine':
            data_df = pd.read_csv(os.path.join(uci_dir, 'wine-quality/winequality-red.csv'), sep=';')
            data = data_df.values

        if loadCSVName == 'yacht':
            data = np.loadtxt(os.path.join(uci_dir, 'yacht/yacht_hydrodynamics.data'))

        if loadCSVName == 'MSD':
            with open(os.path.join(uci_dir, 'song/YearPredictionMSD.npy'), 'rb') as f:
                data = np.load(f)


        ## (2) Load the pre-splitted train/test data
        ## 
        xyTrain_load = np.loadtxt(splitted_data_path+'xyTrain_'+loadCSVName+'_seed_'+str(split_seed)+'.csv', delimiter=',')
        xyTest_load = np.loadtxt(splitted_data_path+'xyTest_'+loadCSVName+'_seed_'+str(split_seed)+'.csv', delimiter=',')
        xyTrain_load = xyTrain_load.astype(np.float32)
        # xyValid_load = xyValid_load.astype(np.float32)
        xyTest_load = xyTest_load.astype(np.float32)

        # original normalization functions 
        # work out normalisation constants (need when unnormalising later)
        scale_c = np.std(data[:, -1])
        shift_c = np.mean(data[:, -1])

        # normalise data
        num_cols = xyTrain_load.shape[1]
        print('num cols: {}'.format(num_cols))

        for i in range(0, num_cols):
            # get the sdev_norm from original data
            sdev_norm = np.std(data[:, i])
            sdev_norm = 0.001 if sdev_norm == 0 else sdev_norm
            # apply on the pre-splitted data
            xyTrain_load[:, i] = (xyTrain_load[:, i] - np.mean(data[:, i]) )/sdev_norm
            xyTest_load[:, i]  = (xyTest_load[:, i] - np.mean(data[:, i]) )/sdev_norm
            # xyValid_load[:, i] = (xyValid_load[:, i] - np.mean(data[:, i]) )/sdev_norm

        if loadCSVName == 'energy' or loadCSVName == 'naval':
            xTrain = xyTrain_load[:, :-2]  ## all columns except last two columns as inputs
            yTrain = xyTrain_load[:, -1]  ## last column as output
            xTest = xyTest_load[:, :-2]
            yTest = xyTest_load[:, -1]
        else:
            xTrain = xyTrain_load[:, :-1]
            yTrain = xyTrain_load[:, -1]
            xTest = xyTest_load[:, :-1]
            yTest = xyTest_load[:, -1]

        self.scale_c = scale_c
        self.shift_c = shift_c

        return xTrain, yTrain, xTest, yTest



