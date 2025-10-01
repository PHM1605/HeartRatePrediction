import os, glob
import numpy as np
import pandas as pd

import datetime
from expt_settings.configs import ExperimentConfig
chosen_columns = range(0,300)

def download_function(config):
    data_folder = config.data_folder
    hr_paths = glob.glob( os.path.join(data_folder, '*_hr.csv') )
    xt_paths = glob.glob( os.path.join(data_folder, '*_xethru.csv') )
    
    df_list = []
    for i in range(len(hr_paths)):
        df_hr = pd.read_csv(hr_paths[i], header=None)
        start_time = datetime.datetime.strptime( (df_hr.iloc[1,2] + ' ' + df_hr.iloc[1,3]), "%d-%m-%Y %I:%M:%S %p" )  
        df_hr = df_hr.iloc[3:, [1,2]].reset_index(drop=True)
        df_xt = pd.read_csv(xt_paths[i], header=None)
        df_xt = df_xt.iloc[ [i for i in range(0, df_xt.shape[0],10)], 1: ].reset_index(drop=True)
        
        # shorten the longer signal
        if df_hr.shape[0] > df_xt.shape[0]:
            df_hr = df_hr.iloc[:df_xt.shape[0], :]
        elif df_xt.shape[0] > df_hr.shape[0]:
            df_xt = df_xt.iloc[:df_hr.shape[0],:]
        
        # put two together into a data frame
        output_df = pd.DataFrame( {'xt{}'.format(channel) : df_xt[channel+1] for channel in range(df_xt.shape[1])} )
        for j in range(df_hr.shape[0]):
            tmp = datetime.datetime.strptime(df_hr.iloc[j,0], '%H:%M:%S')
            df_hr.iloc[j, 0] = start_time + datetime.timedelta(seconds = tmp.minute*60+tmp.second)
        output_df['time'] = df_hr[1]
        output_df['hr'] = df_hr[2]
        output_df['id'] = hr_paths[i][54:-16]
        
        df_list.append(output_df)
    
    output = pd.concat(df_list).reset_index(drop=True)
    output = output[['id','time','hr']+['xt{}'.format(i) for i in chosen_columns]] # only take the first 300 xethru signals
    output.to_csv(config.data_csv_path, index=False)

def main():
    print('### Running download script ###')
    expt_config = ExperimentConfig()
    download_function(expt_config)
    print('Download completed')

main()