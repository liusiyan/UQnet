"""
IO module for train/test regression datasets
"""
import numpy as np
import pandas as pd
import os
import h5py
import tensorflow as tf

def generate_cubic(x, noise=False):
    x = x.astype(np.float32)
    y = x**3

    if noise:
        sigma = 3 * np.ones_like(x)
    else:
        sigma = np.zeros_like(x)
    r = np.random.normal(0, sigma).astype(np.float32)
    return y+r, sigma


#####################################
# individual data files             #
#####################################
vb_dir   = os.path.dirname(__file__)
data_dir = os.path.join(vb_dir, "data/uci")

def _load_boston():
    """
    Attribute Information:
    1. CRIM: per capita crime rate by town
    2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
    3. INDUS: proportion of non-retail business acres per town
    4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    5. NOX: nitric oxides concentration (parts per 10 million)
    6. RM: average number of rooms per dwelling
    7. AGE: proportion of owner-occupied units built prior to 1940
    8. DIS: weighted distances to five Boston employment centres
    9. RAD: index of accessibility to radial highways
    10. TAX: full-value property-tax rate per $10,000
    11. PTRATIO: pupil-teacher ratio by town
    12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    13. LSTAT: % lower status of the population
    14. MEDV: Median value of owner-occupied homes in $1000's
    """
    data = np.loadtxt(os.path.join(data_dir,
                                   'boston-housing/boston_housing.txt'))
    X    = data[:, :-1]
    y    = data[:,  -1]
    return X, y


def _load_powerplant():
    """
    attribute information:
    features consist of hourly average ambient variables
    - temperature (t) in the range 1.81 c and 37.11 c,
    - ambient pressure (ap) in the range 992.89-1033.30 millibar,
    - relative humidity (rh) in the range 25.56% to 100.16%
    - exhaust vacuum (v) in teh range 25.36-81.56 cm hg
    - net hourly electrical energy output (ep) 420.26-495.76 mw
    the averages are taken from various sensors located around the
    plant that record the ambient variables every second.
    the variables are given without normalization.
    """
    data_file = os.path.join(data_dir, 'power-plant/Folds5x2_pp.xlsx')
    data = pd.read_excel(data_file)
    x    = data.values[:, :-1]
    y    = data.values[:,  -1]
    return x, y


def _load_concrete():
    """
    Summary Statistics:
    Number of instances (observations): 1030
    Number of Attributes: 9
    Attribute breakdown: 8 quantitative input variables, and 1 quantitative output variable
    Missing Attribute Values: None
    Name -- Data Type -- Measurement -- Description
    Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable
    Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable
    Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable
    Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable
    Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable
    Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable
    Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable
    Age -- quantitative -- Day (1~365) -- Input Variable
    Concrete compressive strength -- quantitative -- MPa -- Output Variable
    ---------------------------------
    """
    data_file = os.path.join(data_dir, 'concrete/Concrete_Data.xls')
    data = pd.read_excel(data_file)
    X    = data.values[:, :-1]
    y    = data.values[:,  -1]
    return X, y


def _load_yacht():
    """
    Attribute Information:
    Variations concern hull geometry coefficients and the Froude number:
    1. Longitudinal position of the center of buoyancy, adimensional.
    2. Prismatic coefficient, adimensional.
    3. Length-displacement ratio, adimensional.
    4. Beam-draught ratio, adimensional.
    5. Length-beam ratio, adimensional.
    6. Froude number, adimensional.
    The measured variable is the residuary resistance per unit weight of displacement:
    7. Residuary resistance per unit weight of displacement, adimensional.
    """
    data_file = os.path.join(data_dir, 'yacht/yacht_hydrodynamics.data')
    data = pd.read_csv(data_file, delim_whitespace=True)
    X    = data.values[:, :-1]
    y    = data.values[:,  -1]
    return X, y


def _load_energy_efficiency():
    """
    Data Set Information:
    We perform energy analysis using 12 different building shapes simulated in
    Ecotect. The buildings differ with respect to the glazing area, the
    glazing area distribution, and the orientation, amongst other parameters.
    We simulate various settings as functions of the afore-mentioned
    characteristics to obtain 768 building shapes. The dataset comprises
    768 samples and 8 features, aiming to predict two real valued responses.
    It can also be used as a multi-class classification problem if the
    response is rounded to the nearest integer.
    Attribute Information:
    The dataset contains eight attributes (or features, denoted by X1...X8) and two responses (or outcomes, denoted by y1 and y2). The aim is to use the eight features to predict each of the two responses.
    Specifically:
    X1    Relative Compactness
    X2    Surface Area
    X3    Wall Area
    X4    Roof Area
    X5    Overall Height
    X6    Orientation
    X7    Glazing Area
    X8    Glazing Area Distribution
    y1    Heating Load
    y2    Cooling Load
    """
    data_file = os.path.join(data_dir, 'energy-efficiency/ENB2012_data.xlsx')
    data      = pd.read_excel(data_file)
    X         = data.values[:, :-2]
    y_heating = data.values[:, -2]
    y_cooling = data.values[:, -1]
    return X, y_cooling


def _load_wine():
    """
    Attribute Information:
    For more information, read [Cortez et al., 2009].
    Input variables (based on physicochemical tests):
    1 - fixed acidity
    2 - volatile acidity
    3 - citric acid
    4 - residual sugar
    5 - chlorides
    6 - free sulfur dioxide
    7 - total sulfur dioxide
    8 - density
    9 - pH
    10 - sulphates
    11 - alcohol
    Output variable (based on sensory data):
    12 - quality (score between 0 and 10)
    """
    # data_file = os.path.join(data_dir, 'wine-quality/winequality-red.csv')
    data_file = os.path.join(data_dir, 'wine-quality/wine_data_new.txt')
    data     = pd.read_csv(data_file, sep=' ', header=None)
    X = data.values[:, :-1]
    y = data.values[:,  -1]
    return X, y

def _load_kin8nm():
    """
    This is data set is concerned with the forward kinematics of an 8 link robot arm. Among the existing variants of
     this data set we have used the variant 8nm, which is known to be highly non-linear and medium noisy.

    Original source: DELVE repository of data. Source: collection of regression datasets by Luis Torgo
    (ltorgo@ncc.up.pt) at http://www.ncc.up.pt/~ltorgo/Regression/DataSets.html Characteristics: 8192 cases,
    9 attributes (0 nominal, 9 continuous).

    Input variables:
    1 - theta1
    2 - theta2
    ...
    8 - theta8
    Output variable:
    9 - target
    """
    data_file = os.path.join(data_dir, 'kin8nm/dataset_2175_kin8nm.csv')
    data     = pd.read_csv(data_file, sep=',')
    X = data.values[:, :-1]
    y = data.values[:,  -1]
    return X, y


def _load_naval():
    """
    http://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants

    Input variables:
    1 - Lever position(lp)[]
    2 - Ship speed(v)[knots]
    3 - Gas Turbine shaft torque(GTT)[kNm]
    4 - Gas Turbine rate of revolutions(GTn)[rpm]
    5 - Gas Generator rate of revolutions(GGn)[rpm]
    6 - Starboard Propeller Torque(Ts)[kN]
    7 - Port Propeller Torque(Tp)[kN]
    8 - HP Turbine exit temperature(T48)[C]
    9 - GT Compressor inlet air temperature(T1)[C]
    10 - GT Compressor outlet air temperature(T2)[C]
    11 - HP Turbine exit pressure(P48)[bar]
    12 - GT Compressor inlet air pressure(P1)[bar]
    13 - GT Compressor outlet air pressure(P2)[bar]
    14 - Gas Turbine exhaust gas pressure(Pexh)[bar]
    15 - Turbine Injecton Control(TIC)[ %]
    16 - Fuel flow(mf)[kg / s]
    Output variables:
    17 - GT Compressor decay state coefficient.
    18 - GT Turbine decay state coefficient.
    """
    data = np.loadtxt(os.path.join(data_dir, 'naval/data.txt'))
    X = data[:, :-2]
    y_compressor = data[:, -2]
    y_turbine = data[:, -1]
    return X, y_turbine

def _load_protein():
    """
    Physicochemical Properties of Protein Tertiary Structure Data Set
    Abstract: This is a data set of Physicochemical Properties of Protein Tertiary Structure.
    The data set is taken from CASP 5-9. There are 45730 decoys and size varying from 0 to 21 armstrong.

    TODO: Check that the output is correct

    Input variables:
        RMSD-Size of the residue.
        F1 - Total surface area.
        F2 - Non polar exposed area.
        F3 - Fractional area of exposed non polar residue.
        F4 - Fractional area of exposed non polar part of residue.
        F5 - Molecular mass weighted exposed area.
        F6 - Average deviation from standard exposed area of residue.
        F7 - Euclidian distance.
        F8 - Secondary structure penalty.
    Output variable:
        F9 - Spacial Distribution constraints (N,K Value).
    """
    data_file = os.path.join(data_dir, 'protein/CASP.csv')
    data     = pd.read_csv(data_file, sep=',')
    X = data.values[:, 1:]
    y = data.values[:, 0]
    return X, y

def _load_song():
    """
    INSTRUCTIONS:
    1) Download from http://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
    2) Place YearPredictionMSD.txt in data/uci/song/

    Dataloader is slow since file is large.

    YearPredictionMSD Data Set
    Prediction of the release year of a song from audio features. Songs are mostly western, commercial tracks ranging
    from 1922 to 2011, with a peak in the year 2000s.

    90 attributes, 12 = timbre average, 78 = timbre covariance
    The first value is the year (target), ranging from 1922 to 2011.
    Features extracted from the 'timbre' features from The Echo Nest API.
    We take the average and covariance over all 'segments', each segment
    being described by a 12-dimensional timbre vector.

    """
    data = np.loadtxt(os.path.join(data_dir,
                                   'song/YearPredictionMSD.txt'), delimiter=',')
    X    = data[:, :-1]
    y    = data[:,  -1]
    return X, y


def _load_depth():
    train = h5py.File("data/depth_train.h5", "r")
    test = h5py.File("data/depth_test.h5", "r")
    return (train["image"], train["depth"]), (test["image"], test["depth"])

def load_depth():
    return _load_depth()

def load_apollo():
    test = h5py.File("data/apolloscape_test.h5", "r")
    return (None, None), (test["image"], test["depth"])


def LoadData_Splitted_UCI(loadCSVName, original_data_path, splitted_data_path, split_seed, **kwargs):
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
    xyTrain_load = np.loadtxt(splitted_data_path+'/xyTrain_' + loadCSVName + '_seed_' + str(split_seed) + '.csv', delimiter=',')
    xyTest_load = np.loadtxt(splitted_data_path+'./xyTest_' + loadCSVName + '_seed_' + str(split_seed) + '.csv', delimiter=',')
    xyTrain_load = xyTrain_load.astype(np.float32)
    xyTest_load = xyTest_load.astype(np.float32)

    def standardize(data):
        mu = data.mean(axis=0, keepdims=1)
        scale = data.std(axis=0, keepdims=1)
        scale[scale<1e-10] = 1.0

        data = (data - mu) / scale
        return data, mu, scale

    ## (3) select the train/test data
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

    yTrain = np.reshape(yTrain, (-1, 1))
    yTest = np.reshape(yTest, (-1, 1))

    ## (4) standardize the loaded train/test data
    xTrain, xTrain_mu, xTrain_scale = standardize(xTrain)
    xTest = (xTest - xTrain_mu) / xTrain_scale

    yTrain, yTrain_mu, yTrain_scale = standardize(yTrain)
    yTest = (yTest - yTrain_mu) / yTrain_scale

    return xTrain, yTrain, xTest, yTest, yTrain_scale


def load_dataset(name, split_seed=0, test_fraction=.1, return_as_tensor=False):
    # load full dataset
    load_funs = { "wine"              : _load_wine,
                  "boston"            : _load_boston,
                  "concrete"          : _load_concrete,
                  "power-plant"       : _load_powerplant,
                  "yacht"             : _load_yacht,
                  "energy-efficiency" : _load_energy_efficiency,
                  "kin8nm"            : _load_kin8nm,
                  "naval"             : _load_naval,
                  "protein"           : _load_protein,
                  "depth"              : _load_depth,
                  "song"              : _load_song}

    print("Loading dataset {}....".format(name))
    if name == "depth":
        (X_train, y_train), (X_test, y_test) = load_funs[name]()
        y_scale = np.array([[1.0]])
        return (X_train, y_train), (X_test, y_test), y_scale

    X, y = load_funs[name]()
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    def standardize(data):
        mu = data.mean(axis=0, keepdims=1)
        scale = data.std(axis=0, keepdims=1)
        scale[scale<1e-10] = 1.0

        data = (data - mu) / scale
        return data, mu, scale



    # We create the train and test sets with 90% and 10% of the data

    if split_seed == -1:  # Do not shuffle!
        permutation = range(X.shape[0])
    else:
        rs = np.random.RandomState(split_seed)
        permutation = rs.permutation(X.shape[0])

    if name == "boston" or name == "wine":
        test_fraction = 0.2
    size_train  = int(np.round(X.shape[ 0 ] * (1 - test_fraction)))
    index_train = permutation[ 0 : size_train ]
    index_test  = permutation[ size_train : ]

    X_train = X[ index_train, : ]
    X_test  = X[ index_test, : ]

    if name == "depth":
        y_train = y[index_train]
        y_test = y[index_test]
    else:
        y_train = y[index_train, None]
        y_test = y[index_test, None]


    X_train, x_train_mu, x_train_scale = standardize(X_train)
    X_test = (X_test - x_train_mu) / x_train_scale

    y_train, y_train_mu, y_train_scale = standardize(y_train)
    y_test = (y_test - y_train_mu) / y_train_scale

    if return_as_tensor:
        X_train = tf.convert_to_tensor(X_train, tf.float32)
        X_test = tf.convert_to_tensor(X_test, tf.float32)
        y_train = tf.convert_to_tensor(y_train, tf.float32)
        y_test = tf.convert_to_tensor(y_test, tf.float32)

    print("Done loading dataset {}".format(name))
    return (X_train, y_train), (X_test, y_test), y_train_scale



def standardizer(input_np):
    input_mean = input_np.mean(axis=0, keepdims=1)
    input_std = input_np.std(axis=0, keepdims=1)
    input_std[input_std < 1e-10] = 1.0
    standardized_input = (input_np - input_mean) / input_std
    return standardized_input, input_mean, input_std

def getNumInputsOutputs(inputsOutputs_np):
    if len(inputsOutputs_np.shape) == 1:
        numInputsOutputs = 1
    if len(inputsOutputs_np.shape) > 1:
        numInputsOutputs = inputsOutputs_np.shape[1]
    return numInputsOutputs



def load_flight_delays(file_path):

    df_2008 = pd.read_hdf(file_path+'2008.h5', 'df')
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


    df_dep_num = pd.read_csv(file_path+'Airport_rank.csv')

    print(len(df_dep_num[df_dep_num['rank'] == 1]))   ## 28  airports,  4292050 departures
    print(len(df_dep_num[df_dep_num['rank'] == 2]))   ## 44  airports,  1741620 departures
    print(len(df_dep_num[df_dep_num['rank'] == 3]))   ## 93  airports,  787416 departures
    print(len(df_dep_num[df_dep_num['rank'] == 4]))   ## 138 airports,  188642 departures


    rank_list = []
    for i in range(4):
        rank_list.append(df_dep_num[df_dep_num['rank'] == (i+1)]['Origin'].values)


    # train_labels = df_dep_num[df_dep_num['rank'] == 1].iloc[idx_list[0:10]]['Origin'].values
    # test_1_labels = df_dep_num[df_dep_num['rank'] == 1].iloc[idx_list[10:]]['Origin'].values
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

    df_test_4_new = df_test_4_sorted.iloc[int(len(df_test_4_sorted)*0.25):int(len(df_test_4_sorted)*0.3)]  # len = 18,267, 10% of test 4
    # df_test_4_new = df_test_4_sorted.iloc[90000:100000]  # len = 18,267, 10% of test 4
    df_test_4_new = df_test_4_new.sample(frac=1, random_state=1).reset_index(drop=True)   # shuffle
    print('-- len of new rank 4: {}'.format(len(df_test_4_new)))
    print('-- rank 4 new max: {}, min: {}, mean: {}'.format(df_test_4_new['TaxiOut'].max(), df_test_4_new['TaxiOut'].min(), df_test_4_new['TaxiOut'].mean()))

    df_test_3_sorted = df_test_3.sort_values(by=['TaxiOut'], ascending=True)
    # df_test_3_new = df_test_3_sorted.iloc[0:int(len(df_test_3_sorted)*0.05)]

    df_test_3_new = df_test_3_sorted.iloc[int(len(df_test_3_sorted)*0.55):int(len(df_test_3_sorted)*0.6)]

    # df_test_3_new = df_test_3_sorted.iloc[0:10000]
    df_test_3_new = df_test_3_new.sample(frac=1, random_state=1).reset_index(drop=True)   # shuffle
    print('-- len of new rank 3: {}'.format(len(df_test_3_new)))
    print('-- rank 3 new max: {}, min: {}, mean: {}'.format(df_test_3_new['TaxiOut'].max(), df_test_3_new['TaxiOut'].min(), df_test_3_new['TaxiOut'].mean()))



    df_test_2_sorted = df_test_2.sort_values(by=['TaxiOut'], ascending=True)
    # df_test_2_new = df_test_2_sorted.iloc[0:int(len(df_test_2_sorted)*0.05)]
    df_test_2_new = df_test_2_sorted.iloc[int(len(df_test_2_sorted)*0.85):int(len(df_test_2_sorted)*0.9)]
    # df_test_2_new = df_test_2_sorted.iloc[0:10000]
    df_test_2_new = df_test_2_new.sample(frac=1, random_state=1).reset_index(drop=True)   # shuffle
    print('-- len of new rank 2: {}'.format(len(df_test_2_new)))
    print('-- rank 2 new max: {}, min: {}, mean: {}'.format(df_test_2_new['TaxiOut'].max(), df_test_2_new['TaxiOut'].min(), df_test_2_new['TaxiOut'].mean()))


    # (2) find relatively larger TaxiIn data in rank 1,2,3 as the testing data
    df_test_1_sorted = df_test_1.sort_values(by=['TaxiOut'], ascending=True)
    # df_test_1_new = df_test_1_sorted.iloc[int(len(df_test_1_sorted)*0.1):int(len(df_test_1_sorted)*0.15)]
    df_test_1_new = df_test_1_sorted.iloc[int(len(df_test_1_sorted)*0.75):int(len(df_test_1_sorted)*0.8)]


    # df_test_1_new = df_test_1_sorted.iloc[400000:410000]
    df_test_1_new = df_test_1_new.sample(frac=1, random_state=1).reset_index(drop=True)   # shuffle
    print('-- len of new rank 1: {}'.format(len(df_test_1_new)))
    print('-- rank 1 new max: {}, min: {}, mean: {}'.format(df_test_1_new['TaxiOut'].max(), df_test_1_new['TaxiOut'].min(), df_test_1_new['TaxiOut'].mean()))


    # (3) re-define the train and 4 testing data
    df_train = df_test_4_for_train
    df_test_4 = df_test_4_new   # rank 4
    df_test_3 = df_test_3_new   # rank 3
    df_test_2 = df_test_2_new   # rank 2
    df_test_1 = df_test_1_new   # rank 1

    xTrain = df_train[input_features].values
    yTrain = df_train[output_features].values.flatten()
    xTest_list = [df_test_1[input_features].values,  df_test_2[input_features].values,  df_test_3[input_features].values,  df_test_4[input_features].values]
    yTest_list = [df_test_1[output_features].values.flatten(), df_test_2[output_features].values.flatten(), df_test_3[output_features].values.flatten(), df_test_4[output_features].values.flatten()]


    test_data_normalizaed_list = []
    xTrain_normalized, xTrain_mean, xTrain_std = standardizer(xTrain)
    yTrain_normalized, yTrain_mean, yTrain_std = standardizer(yTrain)

    for i in range(len(xTest_list)):
        xTest_normalized = (xTest_list[i] - xTrain_mean) / xTrain_std
        yTest_normalized = (yTest_list[i] - yTrain_mean) / yTrain_std
        test_data_normalizaed_list.append((xTest_normalized, yTest_normalized))

    return xTrain_normalized, yTrain_normalized, yTrain_std, test_data_normalizaed_list

