
''' Data loaders '''

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import io, transform


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


    def list_all_files_in_dir(self, dir_path, **kwargs):  # Return a list with all file names
        if "BOOL_include_dir_names" in kwargs:
            include_dir_names_bool = kwargs.get("BOOL_include_dir_names")
            if include_dir_names_bool is False:
                name_list = []
                for fname in os.listdir(dir_path):
                    test_path = os.path.join(dir_path, fname)
                    if os.path.isdir(test_path):  # skip directories
                        continue
                    else:
                        name_list.append(fname)
            else:  # include_dir_names_bool is True:
                name_list = os.listdir(dir_path)
        else:  # When there is NO "BOOL_include_dir_names" input, ONLY return file names and include folders
            name_list = os.listdir(dir_path)
        return name_list

    def list_all_file_paths_in_dir(self, dir_path, **kwargs):  # Return a list with full paths of all files inside a folder
        if "BOOL_include_dir_names" in kwargs:
            include_dir_names_bool = kwargs.get("BOOL_include_dir_names")
            if include_dir_names_bool is False:
                path_list = []
                for fname in os.listdir(dir_path):
                    test_path = os.path.join(dir_path, fname)
                    if os.path.isdir(test_path):  # skip directories
                        continue
                    else:
                        path_list.append(test_path)
            else:  # include_dir_names_bool is True:
                path_list = [os.path.abspath(x) for x in os.listdir(dir_path)]
        else:  # When there is NO "BOOL_include_dir_names" input, ONLY return paths for all files including folders
            path_list = [os.path.abspath(x) for x in os.listdir(dir_path)]

        return path_list

    # --- Image loading for CNN training
    # (1) Sub-sampled images (training and testing)
    # (2) Simulated permeability values

    ### note: if the images are not large, we can load them at once and form the tf.Dataset like tf.data.Dataset.from_tensor_slices((images, labels)) 
    ### if the images are too large for RAM, to use tf.Dataset efficiently, we can code the label (perm) in the file name (or path)
    ### OR we can try to use wrap the 'get_label' function with a tf.py_function which may affect the performance of the data loader

    ### use conventional train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ###                  test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ### Ref: https://keras.io/examples/vision/3D_image_classification/



    ### trial (1) using direct loading for all images to tf.Dataset with tf.data.Dataset.from_tensor_slices((images, labels)) 
    ### Results: failed because of the large RAM consumptions 
    ### funcion tested: 'load_rock_imgs'
    def load_rock_imgs(self):
        print(self.data_dir)
        name_list = self.list_all_files_in_dir(self.data_dir+'/4_sub_samples/')
        name_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        # print(name_list)  ### sorted name list
        ##### loop load images and concatenate them together (may use a lot of memory)
        print('--- Loop loading all images and concatenate them together...')
        tmp_image = io.imread(self.data_dir+'/4_sub_samples/'+name_list[0])
        xTrain = np.zeros(((len(name_list),)+tmp_image.shape))
        print(xTrain.shape)
        print('--- Loading images')
        for i in range(len(name_list)):  ### too large, cannot fit into memory, need to install 128GB RAM
                                         ### use sklearn train/valid split for only file name and perm
            # print(name_list[i])
            # print(self.data_dir)
            tmp_image = io.imread(self.data_dir+'/4_sub_samples/'+name_list[i])
            tmp_image = tmp_image / 255
            xTrain[i, :] = tmp_image
            print(i)
        print(xTrain)
        # exit()

    ### trial (2) using tf.py_function, which may affect data loader performance, but added more flexibility
    ### Ref: https://stackoverflow.com/questions/65321954/tensorflow-custom-preprocessing-with-tf-py-function-losing-shape

    ### 'get_label'and 'get_perm' function, need to be wrapped by tf.py_function  tf.py_function(func, inp, Tout, name=None)
    ### the big python function will take a single file_path and generate image and label, and the tf.py_function will output image and label TENSORS


    ### extract perm and load image
    def get_img_perm(self, file_path):
        file_name = str(file_path.numpy()).split('/')[-1].split('.')[0]
        # look up for the perm value
        tmp_df = self.df_lbm_perm.loc[self.df_lbm_perm['file_name'] == file_name]
        tmp_perm = tmp_df['perm'].values[0]
        # load and preprocess image
        load_img_path = file_path.numpy().decode('UTF-8')
        image = io.imread(load_img_path)
        image = image / 255
        ## add one channel (black and white image), from (x,x,x) to (x,x,x,1)
        image = np.expand_dims(image, axis=3)
        # image[:,:,:,0] = 1.  
        return file_name, image, tmp_perm

    def tf_get_img_perm(self, file_path):
        [fname, image, tmp_perm] = tf.py_function(self.get_img_perm, [file_path], [tf.string, tf.float32, tf.float32])
        return fname, image, tmp_perm

    # def configure_for_performance(self, ds):
    #     AUTOTUNE = tf.data.AUTOTUNE
    #     ds = ds.cache()
    #     ds = ds.shuffle(buffer_size=1000)
    #     ds = ds.batch(128)
    #     ds = ds.prefetch(buffer_size=AUTOTUNE)
    #     return ds

    def load_rock_imgs_C(self):
        import pathlib
        data_dir = pathlib.Path(self.data_dir+'/4_sub_samples/')
        img_count = len(list(data_dir.glob('*.tif')))
        img_path_list = list(data_dir.glob('*.tif'))
        list_ds = tf.data.Dataset.list_files(self.data_dir+'4_sub_samples/*.tif', shuffle=False)

        #### For fast testing

        # train_size = int(img_count * 0.1)
        # valid_size = int(img_count * 0.1)
        # test_size = int(img_count * 0.1)

        # train_ds = list_ds.take(train_size)
        # test_ds = list_ds.skip(train_size)  ## rest of 90% data

        # valid_ds = test_ds.take(valid_size) ## take the 10% for validation

        # test_ds = test_ds.skip(valid_size) ## rest of 80% data
        # test_ds = test_ds.take(test_size)  ## take another 10% for testing

        ## For 80% train, 10% valid, 10% test
        train_size = int(img_count * 0.05)
        valid_size = int(img_count * 0.05)
        test_size = int(img_count * 0.9)

        train_ds = list_ds.take(train_size)
        test_ds = list_ds.skip(train_size)  ## rest of 20% data
        valid_ds = test_ds.take(valid_size) ## take the 10% for validation
        test_ds = test_ds.skip(test_size)   ## rest of 10% for testing

        train_ds = list_ds.skip(valid_size)
        valid_ds = list_ds.take(valid_size)

        ## use 100% for testing (for PI3NN)
        test_ds = list_ds.take(img_count)

        print(tf.data.experimental.cardinality(train_ds).numpy())
        print(tf.data.experimental.cardinality(valid_ds).numpy())
        print(tf.data.experimental.cardinality(test_ds).numpy())

        self.df_lbm_perm = pd.read_csv(self.data_dir+'8_validation_dataset/all.csv')
        print(self.df_lbm_perm)
        # # Use Dataset.map to create a dataset of image, label pairs:   https://www.tensorflow.org/tutorials/load_data/images
        AUTOTUNE = tf.data.AUTOTUNE
        # train_ds = self.configure_for_performance(train_ds)
        train_ds = train_ds.map(self.tf_get_img_perm, num_parallel_calls=AUTOTUNE)
        valid_ds = valid_ds.map(self.tf_get_img_perm, num_parallel_calls=AUTOTUNE)
        test_ds = test_ds.map(self.tf_get_img_perm, num_parallel_calls=AUTOTUNE)

        train_ds = train_ds.batch(self.configs['batch_size'])
        # train_ds = train_ds.shuffle(buffer_size=1000).batch(self.configs['batch_size'])
        valid_ds = valid_ds.batch(self.configs['batch_size'])
        test_ds = test_ds.batch(self.configs['batch_size'])


        # iii = 0
        # for fname, img, label in train_ds: #.take(10):
        #     print('--- Fname: {}'.format(fname.numpy()))
        #     print('--- Image shape: {}'.format(img.numpy().shape))
        #     print('--- Label: {}'.format(label.numpy()))
        #     iii+=1
        #     print(iii)

        # exit()

        # print('YYYYYYYYYYYYYYYYYYYYYYY')

        # exit()

        return train_ds, valid_ds, test_ds

        ## Ref: https://stackoverflow.com/questions/51125266/how-do-i-split-tensorflow-datasets
        ## .take() Creates a Dataset with at most count elements from this dataset
        ## .skip() Creates a Dataset that skips count elements from this dataset
        ## .shard() Creates a Dataset that includes only 1/num_shards of this dataset








        # ### to take a batch of data for training
        # for fname, img, label in valid_ds.take(1): # .take(2):
        #     print('--- Fname: {}'.format(fname.numpy()))
        #     print('--- Image shape: {}'.format(img.numpy().shape))
        #     print('--- Label: {}'.format(label.numpy())) 


        # ### Training consists of nested loop, inner loop (1) screen all images, and run CNN training; (2) run validation for all validation imgs
        # ### Out loop consists of         

        # exit()


        # # self.train_dataset = self.train_dataset.shuffle(buffer_size=self.configs['batch_shuffle_buffer']).batch(self.configs['batch_size'])
        # # self.train_dataset = self.train_dataset.batch(self.configs['batch_size'])

        # # iii = 0
        # # for fname, img, label in train_ds.take(2000):
        # #     print('--- Fname: {}'.format(fname.numpy()))
        # #     print('--- Image shape: {}'.format(img.numpy().shape))
        # #     print('--- Label: {}'.format(label.numpy()))
        # #     iii+=1
        # #     print(iii)


        # iii = 0
        # for fname, img, label in valid_ds.take(2000):
        #     print('--- Fname: {}'.format(fname.numpy()))
        #     print('--- Image shape: {}'.format(img.numpy().shape))
        #     print('--- Label: {}'.format(label.numpy()))
        #     iii+=1
        #     print(iii)



        


    ### trail (3) code the perm in the file (or path) name, and use ALL TF ops to get maximum performance (try later for performane comparison)





    # exit()


    # ##### load the pandas dataframe for perm and sort the them

    # ##### extract perm to numpy array

    # print('Test end here')
    # print(name_list[0])



    # # ##### train/valid split
    # # xTrain, xValid, yTrain, yValid = train_test_split(pd.DataFrame(Xtrain), pd.DataFrame(yobs_train), test_size=60, random_state=0)


    # # ##### and construct TF dataset
    # # train_dataset = tf.data.Dataset.from_tensor_slices((xTrain_imgs, yTrain_imgs))

    # exit()


    # import pathlib
    # data_dir = pathlib.Path(self.data_dir+'/4_sub_samples/')

    # print(data_dir)

    # img_count = len(list(data_dir.glob('*.tif')))
    # print(img_count)
    # img_path_list = list(data_dir.glob('*.tif'))
    # print(img_path_list[0])

    # print(img_path_list)

    # ### test batch loading
    # batch_size = 32
    # img_length = 128
    # img_height = 128
    # img_width = 128

    # self.df_lbm_perm = pd.read_csv(self.data_dir+'8_validation_dataset/all.csv')


    # exit()








    # def get_perm(self, file_path):
    #     # find the corresponding perm value

    #     # parts = tf.strings.split(file_path, os.path.sep)

    #     # print(parts)
    #     # print(parts[-1])
 
    #     # exit()
    
    #     file_name = str(file_path.numpy()).split('/')[-1].split('.')[0]
    #     print(file_name)
    #     # look up for the perm value
    #     tmp_df = self.df_lbm_perm.loc[self.df_lbm_perm['file_name'] == file_name]
    #     tmp_perm = tmp_df['perm'].values[0]
    #     return file_name, tmp_perm

    # def process_img(self, file_path):
    #     # load and preprocess image
    #     load_img_path = file_path.numpy().decode('UTF-8')
    #     image = io.imread(load_img_path)
    #     image = image / 255
    #     return image

    # def process_path(self, file_path):
    #     ### Get file name and label (Perm)
    #     file_name, tmp_perm = self.get_perm(file_path)

    #     ### Load the corresponding image
    #     img = self.process_img(file_path)
    #     return img, tmp_perm

    # def load_rock_imgs_BBBBBBBBBBBB(self):
    #     print(self.data_dir)
    #     print('SSSSSSSSSSSSSSSSs')

    #     import pathlib
    #     data_dir = pathlib.Path(self.data_dir+'/4_sub_samples/')

    #     print(data_dir)

    #     img_count = len(list(data_dir.glob('*.tif')))
    #     print(img_count)
    #     img_path_list = list(data_dir.glob('*.tif'))
    #     print(img_path_list[0])

    #     ### test batch loading
    #     batch_size = 32
    #     img_length = 128
    #     img_height = 128
    #     img_width = 128

    #     print(self.data_dir)

    #     self.df_lbm_perm = pd.read_csv(self.data_dir+'8_validation_dataset/all.csv')
    #     list_ds = tf.data.Dataset.list_files(self.data_dir+'4_sub_samples/*.tif', shuffle=False)

    #     for f in list_ds.take(10):
    #         print(f.numpy())
    #         fname, tmp_perm = self.get_perm(f)

    #         # print(fname)
    #         # print(tmp_perm)

    #         # self.process_img(f)

    #     exit()






    #     test_size = int(img_count * 0.1)
    #     train_ds = list_ds.skip(test_size)
    #     test_ds = list_ds.take(test_size)

    #     print(tf.data.experimental.cardinality(train_ds).numpy())
    #     print(tf.data.experimental.cardinality(test_ds).numpy())

    #     self.df_lbm_perm = pd.read_csv(self.data_dir+'8_validation_dataset/all.csv')
    #     print(self.df_lbm_perm)


    #     exit()

    #     # Use Dataset.map to create a dataset of image, label pairs:   https://www.tensorflow.org/tutorials/load_data/images
    #     AUTOTUNE = tf.data.AUTOTUNE
    #     train_ds = train_ds.map(self.process_path, num_parallel_calls=AUTOTUNE)












    #     exit()

    #     print(df_lbm_perm.iloc[0, 0])
    #     print(type(df_lbm_perm.iloc[0, 0]))


    #     for f in train_ds.take(5):
    #         print(f.numpy())
    #         fname, tmp_perm = self.get_perm(df_lbm_perm, f)

    #         print(fname)
    #         print(tmp_perm)

    #         self.process_img(f)

    #         # tmp_df = df_lbm_perm.loc[df_lbm_perm['file_name'] == fname]
    #         # perm = tmp_df['perm'].values

    #         # print(tmp_df)
    #         # print(perm[0])
    #         # print(type(perm))



    #     # for f in test_ds.take(5):
    #     #     print(f.numpy())





    #     exit()


    #     test_ds = tf.data.Dataset.list_files(self.data_dir+'8_validation_dataset/validation_images/*.tif', shuffle=False)
    #     # list_ds = list_ds.shuffle(img_count, reshuffle_each_iteration=False)
    #     print(train_ds)
    #     print(test_ds)

    #     for f in train_ds.take(5):
    #         print(f.numpy())

    #     for f in test_ds.take(5):
    #         print(f.numpy())


    #     print('SSSSSSSSSSSSSSSSSSSSSSSSSS')

    #     exit()


    #     for f in train_ds.take(5):
    #         print(f.numpy())
    #         print(type(f.numpy()))
    #         print(type(str(f.numpy())))

    #         print(str(f.numpy()).split('/')[-1].split('.')[0])



    #     exit()
    #     # sortt = sorted(list_ds)
    #     # print(sortt)


    #     ### load perm values
    #     print(self.data_dir+'8_validation_dataset/')

    #     df_lbm_perm = pd.read_csv(self.data_dir+'8_validation_dataset/train.csv')

    #     print(df_lbm_perm)
    #     print(df_lbm_perm.iloc[0, 0])


    #     ### test and verify the loaded image and associated perm values

    #     # for image, label in train_ds.take(1):
    #     #     print('--- Image path: ')
    #     #     print('--- Image shape: {}'.format(image.numpy().shape))
    #     #     print('--- Label: {}'.format(label.numpy()))

        




