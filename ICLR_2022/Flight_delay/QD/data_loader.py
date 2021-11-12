"""
IO module for train/test regression datasets
"""
import numpy as np
import pandas as pd
import os
import h5py
import tensorflow as tf


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

    return xTrain_normalized, yTrain_normalized, test_data_normalizaed_list