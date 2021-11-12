import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

''' Compare the results between PI3NN, QD, DER, PIVEN and SQR '''

use_pre_generated_results = True

PI3NN_results_path = './PI3NN_UCI/Results/'                          # xxx_PI_results.txt
QD_results_path = './QD_UCI/Results/'                                # xxx_QD_results.txt
DER_results_path = './DER_UCI/Results_DER/'                          # xxx_DER_results.txt
PIVEN_results_path = './PIVEN_UCI/Results/piven/'                   # xxx_PIVEN_UCI.txt
# SQR_results_path = './SQR_UCI/aleatoric/regression/pre_split_UCI_add_valid_results/'   # CQ_alpha_0.5_xxx.txt
SQR_results_path = './SQR_UCI/aleatoric/regression/SQR_tmp_results/'   # CQ_alpha_0.5_xxx.txt

dataset_list = ['boston', 'concrete', 'energy', 'kin8nm', 'naval', 'power', 'protein', 'wine', 'yacht'] # MSD

processing_list = ['PI3NN', 'QD', 'DER', 'PIVEN', 'SQR']
# processing_list = ['PI3NN']

bool_save_csv = False
bool_compare_by_split_seed = False
split_seed = 1


### load the 'y_range' (normalized y_all.max() - y_all.min()) for interval normalization 
### apply to PI3NN, QD, DER and PIVEN method (SQR implemented this itself)
df_yr = pd.read_csv('y_range.txt', sep=' ')

''' (1) PI3NN'''
if 'PI3NN' in processing_list:
    print('--- Processing PI3NN results...')
    PICP_good = []; PICP_bad = []; PICP_mean = []; PICP_std = [];\
    MPIW_good = []; MPIW_bad = []; MPIW_mean = []; MPIW_std = [];\
    RMSE_good = []; RMSE_bad = []; RMSE_mean = []; RMSE_std = [];\
    R2_good = []; R2_bad = []; R2_mean = []; R2_std = []; MPIW_4GoodPICP = []


    data_length_list = []
    data_list = []
    for idx, dataset in enumerate(dataset_list):
        if idx >= 0:
            data_list.append(dataset)

            ## PI3NN
            # print('-- PI3NN: {}'.format(dataset))
            df_PI3NN = pd.read_csv(PI3NN_results_path+dataset+'_PI_results.txt', sep=' ')
            data_length_list.append(len(df_PI3NN))

            if bool_compare_by_split_seed:
                df_PI3NN = df_PI3NN[df_PI3NN['split_seed'] == split_seed]
                df_PI3NN.reset_index(drop=True, inplace=True) 

                ## apply the y_range
                tmp_y_range = df_yr[(df_yr['data_name']==dataset) & (df_yr['split_seed']==split_seed)]['y_range'].values[0]
                df_PI3NN['MPIW_test'] = df_PI3NN['MPIW_test'] / tmp_y_range

            # print(df_PI3NN)
            good_PICP_idx = (df_PI3NN['PICP_test'] - 0.95).abs().idxmin()
            bad_PICP_idx = (df_PI3NN['PICP_test'] - 0.95).abs().idxmax()
            PICP_good.append(df_PI3NN['PICP_test'].iloc[good_PICP_idx])
            PICP_bad.append(df_PI3NN['PICP_test'].iloc[bad_PICP_idx])
            PICP_mean.append(df_PI3NN['PICP_test'].mean())
            PICP_std.append(df_PI3NN['PICP_test'].std())

            good_MPIW_idx = df_PI3NN['MPIW_test'][df_PI3NN['MPIW_test']>0].abs().idxmin()
            bad_MPIW_idx = df_PI3NN['MPIW_test'][df_PI3NN['MPIW_test']>0].abs().idxmax()
            MPIW_good.append(df_PI3NN['MPIW_test'].iloc[good_MPIW_idx])
            MPIW_bad.append(df_PI3NN['MPIW_test'].iloc[bad_MPIW_idx])        
            MPIW_mean.append(df_PI3NN['MPIW_test'].mean())
            MPIW_std.append(df_PI3NN['MPIW_test'].std())

            ## find the corresponding MPIW for best PICP, calculate average value if duplicates encountered
            good_PICP = df_PI3NN['PICP_test'].iloc[good_PICP_idx]
            tmp_df = df_PI3NN.where(df_PI3NN['PICP_test'] == good_PICP).dropna()
            tmp_MPIW = tmp_df['MPIW_test'].mean()
            MPIW_4GoodPICP.append(tmp_MPIW)

            good_RMSE_idx = df_PI3NN['RMSE'][df_PI3NN['RMSE']>0].abs().idxmin()
            bad_RMSE_idx = df_PI3NN['RMSE'][df_PI3NN['RMSE']>0].abs().idxmax()        
            RMSE_good.append(df_PI3NN['RMSE'].iloc[good_RMSE_idx])
            RMSE_bad.append(df_PI3NN['RMSE'].iloc[bad_RMSE_idx])
            RMSE_mean.append(df_PI3NN['RMSE'].mean())
            RMSE_std.append(df_PI3NN['RMSE'].std())

            good_R2_idx = (df_PI3NN['R2'] - 1.0).abs().idxmin()
            bad_R2_idx = (df_PI3NN['R2'] - 1.0).abs().idxmax() 
            R2_good.append(df_PI3NN['R2'].iloc[good_R2_idx])
            R2_bad.append(df_PI3NN['R2'].iloc[bad_R2_idx])
            R2_mean.append(df_PI3NN['R2'].mean())
            R2_std.append(df_PI3NN['R2'].std())


    df_PI3NN_results = pd.DataFrame({
        'data': data_list,
        'PICP_good': PICP_good,
        'PICP_bad': PICP_bad,
        'PICP_mean': PICP_mean,
        'PICP_std': PICP_std,
        'MPIW_good': MPIW_good,
        'MPIW_bad': MPIW_bad,
        'MPIW_mean': MPIW_mean,
        'MPIW_std': MPIW_std,
        'RMSE_good': RMSE_good,
        'RMSE_bad': RMSE_bad,
        'RMSE_mean': RMSE_mean,
        'RMSE_std': RMSE_std,
        'R2_good': R2_good,
        'R2_bad': R2_bad,
        'R2_mean': R2_mean,
        'R2_std': R2_std,
        'MPIW_4GoodPICP': MPIW_4GoodPICP
        })
    print('--- PI3NN results for split seed: --- {} ---'.format(split_seed))
    print(df_PI3NN_results)
    if bool_save_csv:
        if bool_compare_by_split_seed:
            df_PI3NN_results.to_csv('PI3NN_summary_split_seed_'+str(split_seed)+'.csv')
        else:
            df_PI3NN_results.to_csv('PI3NN_summary.csv')


''' (2) QD'''
if 'QD' in processing_list:
    print('--- Processing QD results...')
    PICP_good = []; PICP_bad = []; PICP_mean = []; PICP_std = [];\
    MPIW_good = []; MPIW_bad = []; MPIW_mean = []; MPIW_std = [];\
    RMSE_good = []; RMSE_bad = []; RMSE_mean = []; RMSE_std = [];\
    R2_good = []; R2_bad = []; R2_mean = []; R2_std = []; MPIW_4GoodPICP = []

    data_length_list = []
    data_list = []
    for idx, dataset in enumerate(dataset_list):
        if idx >= 0:
            data_list.append(dataset)
            # print('-- QD: {}'.format(dataset))

            df_QD = pd.read_csv(QD_results_path+dataset+'_QD_results.txt', sep=' ')
            data_length_list.append(len(df_QD))
            # print(df_QD)

            if bool_compare_by_split_seed:
                df_QD = df_QD[df_QD['split_seed'] == split_seed]
                df_QD.reset_index(drop=True, inplace=True) 
                ## apply the y_range
                tmp_y_range = df_yr[(df_yr['data_name']==dataset) & (df_yr['split_seed']==split_seed)]['y_range'].values[0]
                df_QD['MPIW_test'] = df_QD['MPIW_test'] / tmp_y_range

            good_PICP_idx = (df_QD['PICP_test'] - 0.95).abs().idxmin()
            bad_PICP_idx = (df_QD['PICP_test'] - 0.95).abs().idxmax()
            PICP_good.append(df_QD['PICP_test'].iloc[good_PICP_idx])
            PICP_bad.append(df_QD['PICP_test'].iloc[bad_PICP_idx])
            PICP_mean.append(df_QD['PICP_test'].mean())
            PICP_std.append(df_QD['PICP_test'].std())

            good_MPIW_idx = df_QD['MPIW_test'][df_QD['MPIW_test']>0].abs().idxmin()
            bad_MPIW_idx = df_QD['MPIW_test'][df_QD['MPIW_test']>0].abs().idxmax()
            MPIW_good.append(df_QD['MPIW_test'].iloc[good_MPIW_idx])
            MPIW_bad.append(df_QD['MPIW_test'].iloc[bad_MPIW_idx])        
            MPIW_mean.append(df_QD['MPIW_test'].mean())
            MPIW_std.append(df_QD['MPIW_test'].std())

            good_PICP = df_QD['PICP_test'].iloc[good_PICP_idx]
            tmp_df = df_QD.where(df_QD['PICP_test'] == good_PICP).dropna()
            tmp_MPIW = tmp_df['MPIW_test'].mean()            
            MPIW_4GoodPICP.append(tmp_MPIW)

            good_RMSE_idx = df_QD['RMSE'][df_QD['RMSE']>0].abs().idxmin()
            bad_RMSE_idx = df_QD['RMSE'][df_QD['RMSE']>0].abs().idxmax()        
            RMSE_good.append(df_QD['RMSE'].iloc[good_RMSE_idx])
            RMSE_bad.append(df_QD['RMSE'].iloc[bad_RMSE_idx])
            RMSE_mean.append(df_QD['RMSE'].mean())
            RMSE_std.append(df_QD['RMSE'].std())

            good_R2_idx = (df_QD['R2'] - 1.0).abs().idxmin()
            bad_R2_idx = (df_QD['R2'] - 1.0).abs().idxmax() 
            R2_good.append(df_QD['R2'].iloc[good_R2_idx])
            R2_bad.append(df_QD['R2'].iloc[bad_R2_idx])
            R2_mean.append(df_QD['R2'].mean())
            R2_std.append(df_QD['R2'].std())


    df_QD_results = pd.DataFrame({
        'data': data_list,
        'PICP_good': PICP_good,
        'PICP_bad': PICP_bad,
        'PICP_mean': PICP_mean,
        'PICP_std': PICP_std,
        'MPIW_good': MPIW_good,
        'MPIW_bad': MPIW_bad,
        'MPIW_mean': MPIW_mean,
        'MPIW_std': MPIW_std,
        'RMSE_good': RMSE_good,
        'RMSE_bad': RMSE_bad,
        'RMSE_mean': RMSE_mean,
        'RMSE_std': RMSE_std,
        'R2_good': R2_good,
        'R2_bad': R2_bad,
        'R2_mean': R2_mean,
        'R2_std': R2_std,
        'MPIW_4GoodPICP': MPIW_4GoodPICP
        })
    print('--- QD results for split seed: --- {} ---'.format(split_seed))
    print(df_QD_results)
    if bool_save_csv:
        if bool_compare_by_split_seed:
            df_QD_results.to_csv('QD_summary_split_seed_'+str(split_seed)+'.csv')
        else:
            df_QD_results.to_csv('QD_summary.csv')



''' (3) DER '''
if 'DER' in processing_list:
    print('--- Processing DER results...')
    PICP_good = []; PICP_bad = []; PICP_mean = []; PICP_std = [];\
    MPIW_good = []; MPIW_bad = []; MPIW_mean = []; MPIW_std = [];\
    RMSE_good = []; RMSE_bad = []; RMSE_mean = []; RMSE_std = [];\
    R2_good = []; R2_bad = []; R2_mean = []; R2_std = []; MPIW_4GoodPICP = []

    data_length_list = []
    data_list = []
    for idx, dataset in enumerate(dataset_list):
        if idx >= 0:
            data_list.append(dataset)
            # print('-- QD: {}'.format(dataset))

            df_DER = pd.read_csv(DER_results_path+dataset+'_DER_results.txt', sep=' ')
            df_DER = df_DER.dropna(axis=1)
            data_length_list.append(len(df_DER))

            if bool_compare_by_split_seed:
                df_DER = df_DER[df_DER['split_seed'] == split_seed]
                df_DER.reset_index(drop=True, inplace=True) 

                ## apply the y_range
                tmp_y_range = df_yr[(df_yr['data_name']==dataset) & (df_yr['split_seed']==split_seed)]['y_range'].values[0]
                df_DER['MPIW_test'] = df_DER['MPIW_test'] / tmp_y_range


            good_PICP_idx = (df_DER['PICP_test'] - 0.95).abs().idxmin()
            bad_PICP_idx = (df_DER['PICP_test'] - 0.95).abs().idxmax()
            PICP_good.append(df_DER['PICP_test'].iloc[good_PICP_idx])
            PICP_bad.append(df_DER['PICP_test'].iloc[bad_PICP_idx])
            PICP_mean.append(df_DER['PICP_test'].mean())
            PICP_std.append(df_DER['PICP_test'].std())

            good_MPIW_idx = df_DER['MPIW_test'][df_DER['MPIW_test']>0].abs().idxmin()
            bad_MPIW_idx = df_DER['MPIW_test'][df_DER['MPIW_test']>0].abs().idxmax()
            MPIW_good.append(df_DER['MPIW_test'].iloc[good_MPIW_idx])
            MPIW_bad.append(df_DER['MPIW_test'].iloc[bad_MPIW_idx])        
            MPIW_mean.append(df_DER['MPIW_test'].mean())
            MPIW_std.append(df_DER['MPIW_test'].std())

            good_PICP = df_DER['PICP_test'].iloc[good_PICP_idx]
            tmp_df = df_DER.where(df_DER['PICP_test'] == good_PICP).dropna()
            tmp_MPIW = tmp_df['MPIW_test'].mean()  
            MPIW_4GoodPICP.append(tmp_MPIW)           

            good_RMSE_idx = df_DER['RMSE'][df_DER['RMSE']>0].abs().idxmin()
            bad_RMSE_idx = df_DER['RMSE'][df_DER['RMSE']>0].abs().idxmax()        
            RMSE_good.append(df_DER['RMSE'].iloc[good_RMSE_idx])
            RMSE_bad.append(df_DER['RMSE'].iloc[bad_RMSE_idx])
            RMSE_mean.append(df_DER['RMSE'].mean())
            RMSE_std.append(df_DER['RMSE'].std())

            good_R2_idx = (df_DER['R2'] - 1.0).abs().idxmin()
            bad_R2_idx = (df_DER['R2'] - 1.0).abs().idxmax() 
            R2_good.append(df_DER['R2'].iloc[good_R2_idx])
            R2_bad.append(df_DER['R2'].iloc[bad_R2_idx])
            R2_mean.append(df_DER['R2'].mean())
            R2_std.append(df_DER['R2'].std())

    df_DER_results = pd.DataFrame({
        'data': data_list,
        'PICP_good': PICP_good,
        'PICP_bad': PICP_bad,
        'PICP_mean': PICP_mean,
        'PICP_std': PICP_std,
        'MPIW_good': MPIW_good,
        'MPIW_bad': MPIW_bad,
        'MPIW_mean': MPIW_mean,
        'MPIW_std': MPIW_std,
        'RMSE_good': RMSE_good,
        'RMSE_bad': RMSE_bad,
        'RMSE_mean': RMSE_mean,
        'RMSE_std': RMSE_std,
        'R2_good': R2_good,
        'R2_bad': R2_bad,
        'R2_mean': R2_mean,
        'R2_std': R2_std,
        'MPIW_4GoodPICP': MPIW_4GoodPICP
        })
    print('--- DER results for split seed: --- {} ---'.format(split_seed))
    print(df_DER_results)
    if bool_save_csv:
        if bool_compare_by_split_seed:
            df_DER_results.to_csv('DER_summary_split_seed_'+str(split_seed)+'.csv')
        else:
            df_DER_results.to_csv('DER_summary.csv')


''' (4) PIVEN '''
if 'PIVEN' in processing_list:
    print('--- Processing PIVEN results...')
    num_cases = 243

    PICP_good = []; PICP_bad = []; PICP_mean = []; PICP_std = [];\
    MPIW_good = []; MPIW_bad = []; MPIW_mean = []; MPIW_std = [];\
    RMSE_good = []; RMSE_bad = []; RMSE_mean = []; RMSE_std = []; MPIW_4GoodPICP = []
    # R2_good = []; R2_bad = []; R2_mean = []; R2_std = []

    data_length_list = []
    data_list = []
    for idx, dataset in enumerate(dataset_list):
        if idx >= 0:
            data_list.append(dataset)
            # print('-- PIVEN: {}'.format(dataset))
            df_PIVEN = pd.read_csv(PIVEN_results_path+dataset+'_PIVEN_UCI.txt', sep=' ')
            data_length_list.append(len(df_PIVEN))
            # print('--- Training finished {}: {}/{}'.format(dataset, len(df_PIVEN), num_cases))
            # print(df_PIVEN)

            if bool_compare_by_split_seed:
                df_PIVEN = df_PIVEN[df_PIVEN['split_seed'] == split_seed]
                df_PIVEN.reset_index(drop=True, inplace=True) 
                ## apply the y_range
                tmp_y_range = df_yr[(df_yr['data_name']==dataset) & (df_yr['split_seed']==split_seed)]['y_range'].values[0]
                df_PIVEN['MPIW_test'] = df_PIVEN['MPIW_test'] / tmp_y_range

            good_PICP_idx = (df_PIVEN['PICP_test'] - 0.95).abs().idxmin()
            bad_PICP_idx = (df_PIVEN['PICP_test'] - 0.95).abs().idxmax()
            PICP_good.append(df_PIVEN['PICP_test'].iloc[good_PICP_idx])
            PICP_bad.append(df_PIVEN['PICP_test'].iloc[bad_PICP_idx])
            PICP_mean.append(df_PIVEN['PICP_test'].mean())
            PICP_std.append(df_PIVEN['PICP_test'].std())

            good_MPIW_idx = df_PIVEN['MPIW_test'][df_PIVEN['MPIW_test']>0].abs().idxmin()
            bad_MPIW_idx = df_PIVEN['MPIW_test'][df_PIVEN['MPIW_test']>0].abs().idxmax()
            MPIW_good.append(df_PIVEN['MPIW_test'].iloc[good_MPIW_idx])
            MPIW_bad.append(df_PIVEN['MPIW_test'].iloc[bad_MPIW_idx])        
            MPIW_mean.append(df_PIVEN['MPIW_test'].mean())
            MPIW_std.append(df_PIVEN['MPIW_test'].std())

            good_PICP = df_PIVEN['PICP_test'].iloc[good_PICP_idx]
            tmp_df = df_PIVEN.where(df_PIVEN['PICP_test'] == good_PICP).dropna()
            tmp_MPIW = tmp_df['MPIW_test'].mean()  
            MPIW_4GoodPICP.append(tmp_MPIW)  

            good_RMSE_idx = df_PIVEN['RMSE'][df_PIVEN['RMSE']>0].abs().idxmin()
            bad_RMSE_idx = df_PIVEN['RMSE'][df_PIVEN['RMSE']>0].abs().idxmax()        
            RMSE_good.append(df_PIVEN['RMSE'].iloc[good_RMSE_idx])
            RMSE_bad.append(df_PIVEN['RMSE'].iloc[bad_RMSE_idx])
            RMSE_mean.append(df_PIVEN['RMSE'].mean())
            RMSE_std.append(df_PIVEN['RMSE'].std())


    df_PIVEN_results = pd.DataFrame({
        'data': data_list,
        'PICP_good': PICP_good,
        'PICP_bad': PICP_bad,
        'PICP_mean': PICP_mean,
        'PICP_std': PICP_std,
        'MPIW_good': MPIW_good,
        'MPIW_bad': MPIW_bad,
        'MPIW_mean': MPIW_mean,
        'MPIW_std': MPIW_std,
        'RMSE_good': RMSE_good,
        'RMSE_bad': RMSE_bad,
        'RMSE_mean': RMSE_mean,
        'RMSE_std': RMSE_std,
        'MPIW_4GoodPICP': MPIW_4GoodPICP
        })
    print('--- PIVEN results for split seed: --- {} ---'.format(split_seed))
    print(df_PIVEN_results)
    if bool_save_csv:
        if bool_compare_by_split_seed:
            df_PIVEN_results.to_csv('PIVEN_summary_split_seed_'+str(split_seed)+'.csv')
        else:
            df_PIVEN_results.to_csv('PIVEN_summary.csv')


'''(5) SQR '''
if 'SQR' in processing_list:
    print('--- Processing SQR results...')
    num_cases = 1620

    PICP_good = []; PICP_bad = []; PICP_mean = []; PICP_std = [];\
    MPIW_good = []; MPIW_bad = []; MPIW_mean = []; MPIW_std = [];\
    RMSE_good = []; RMSE_bad = []; RMSE_mean = []; RMSE_std = [];\
    R2_good = []; R2_bad = []; R2_mean = []; R2_std = []; MPIW_4GoodPICP = []

    data_length_list = []
    data_list = []
    finished = 0

    lr=1e-2
    dropout=0.1 
    wd=0.1

    data_length_list = []
    data_list = []
    finished = 0

    for idx, dataset in enumerate(dataset_list):
        if idx >= 0:
            data_list.append(dataset)
            # print('-- SQR: {}'.format(dataset))
            df_SQR = pd.read_csv(SQR_results_path+'CQ_alpha_0.05_'+dataset+'.txt', sep=' ')
            data_length_list.append(len(df_SQR))
            # print('--- Training finished {}: {}/{}  {:.2f} %'.format(dataset, len(df_SQR), num_cases, len(df_SQR)/num_cases*100))
            finished+=len(df_SQR)

            ### Group by the same lr, dropout and wd parameter combination lr=1e-3, dropout=0.1 wd=0
            df_SQR = df_SQR[(df_SQR['lr'] == lr) & (df_SQR['dropout'] == dropout) & (df_SQR['wd'] == wd)]
            df_SQR.reset_index(drop=True, inplace=True) 

            if bool_compare_by_split_seed:
                df_PI3NN = df_PI3NN[df_PI3NN['split_seed'] == split_seed]
                df_PI3NN.reset_index(drop=True, inplace=True) 
            # exit()

            good_PICP_idx = (df_SQR['capture_te'] - 0.95).abs().idxmin()
            bad_PICP_idx = (df_SQR['capture_te'] - 0.95).abs().idxmax()

            PICP_good.append(df_SQR['capture_te'].iloc[good_PICP_idx])
            PICP_bad.append(df_SQR['capture_te'].iloc[bad_PICP_idx])
            PICP_mean.append(df_SQR['capture_te'].mean())
            PICP_std.append(df_SQR['capture_te'].std())

            good_MPIW_idx = df_SQR['width_te'][df_SQR['width_te']>0].abs().idxmin()
            bad_MPIW_idx = df_SQR['width_te'][df_SQR['width_te']>0].abs().idxmax()
            MPIW_good.append(df_SQR['width_te'].iloc[good_MPIW_idx])
            MPIW_bad.append(df_SQR['width_te'].iloc[bad_MPIW_idx])        
            MPIW_mean.append(df_SQR['width_te'].mean())
            MPIW_std.append(df_SQR['width_te'].std())

            good_PICP = df_SQR['width_te'].iloc[good_PICP_idx]
            tmp_df = df_SQR.where(df_SQR['width_te'] == good_PICP).dropna()
            tmp_MPIW = tmp_df['width_te'].mean()  
            MPIW_4GoodPICP.append(tmp_MPIW)

            df_SQR['rmse_te'] = df_SQR['mse_te']**(1/2)
            good_RMSE_idx = df_SQR['rmse_te'][df_SQR['rmse_te']>0].abs().idxmin()
            bad_RMSE_idx = df_SQR['rmse_te'][df_SQR['rmse_te']>0].abs().idxmax()        
            RMSE_good.append(df_SQR['rmse_te'].iloc[good_RMSE_idx])
            RMSE_bad.append(df_SQR['rmse_te'].iloc[bad_RMSE_idx])
            RMSE_mean.append(df_SQR['rmse_te'].mean())
            RMSE_std.append(df_SQR['rmse_te'].std())


    print('--- Total finished: {}/{}, {:.2f} %'.format(finished, num_cases*9, finished/(num_cases*9)*100))


    df_SQR_results = pd.DataFrame({
        'data': data_list,
        'PICP_good': PICP_good,
        'PICP_bad': PICP_bad,
        'PICP_mean': PICP_mean,
        'PICP_std': PICP_std,
        'MPIW_good': MPIW_good,
        'MPIW_bad': MPIW_bad,
        'MPIW_mean': MPIW_mean,
        'MPIW_std': MPIW_std,
        'RMSE_good': RMSE_good,
        'RMSE_bad': RMSE_bad,
        'RMSE_mean': RMSE_mean,
        'RMSE_std': RMSE_std,
        'MPIW_4GoodPICP': MPIW_4GoodPICP
        })
    print('--- SQR results for split seed: --- {} ---'.format(split_seed))
    print(df_SQR_results)

    if bool_save_csv:
        df_SQR_results.to_csv('SQR_summary_split_seed_'+str(split_seed)+'.csv')







