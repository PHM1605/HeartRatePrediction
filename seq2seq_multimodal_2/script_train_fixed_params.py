import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import hickle as hkl
import sklearn.preprocessing

import libs.seq2seq_model
ModelClass = libs.seq2seq_model.Sequence2Sequence

def split_data(df, train_boundary=15): # 15 people as train people
    index = df['id']
    df_list_train, df_list_test = [], []
    for one_id in index.unique():
        if len(df_list_train) < train_boundary:
            df_list_train.append( df[index==one_id] )
        else:
            df_list_test.append( df[index==one_id] )
    train = pd.concat(df_list_train).reset_index(drop=True)
    test = pd.concat(df_list_test).reset_index(drop=True)        
    return [train, test]

def batch_data(data, val_columns):
    piece_len = 30 
    num_encoder_steps = 20

    def _batch_single_entity(input_data):
        time_steps = len(input_data)
        x = input_data.values 
        if time_steps >= piece_len:
            return np.stack( [x[i:time_steps-(piece_len-1)+i] for i in range(piece_len)], axis=1 )
        else:
            return None
        
    target_col = ['hr']
    data_map = {}
    for _, sliced in data.groupby('id'):
        col_mappings = {'outputs': target_col, 'inputs': val_columns}
        for k in col_mappings:
            cols = col_mappings[k]
            arr = _batch_single_entity(sliced[cols].copy())
            if arr is not None:
                if k not in data_map:
                    data_map[k] = [arr]
                else:
                    data_map[k].append(arr)
    for k in data_map:
        data_map[k] = np.concatenate(data_map[k], axis=0)
    data_map['full_outputs'] = data_map['outputs']
    data_map['outputs'] = data_map['outputs'][:, num_encoder_steps:, :]
    return data_map # dict of 4 entries 'identifier' [n_samples,30,1], 'inputs' [n_samples,30,301], 'outputs' [n_samples,10,1], 'time' [n_samples,30,1], each is 3D np array
    

experiments = ['rest', 'normal', 'abnormal']
all_experiments_losses = []
for experiment_name in experiments:
    # data_folder = os.path.join('/uio/kant/ifi-ansatt-u04/hoangmph/hoangminh/data/mecs', experiment_name)
    data_folder = os.path.join('../data/mecs', experiment_name)
    print('Start experiment ' + experiment_name)
    all_losses = {}
    for q in [0.1, 0.5, 0.9]:
        tmp_loss = []
        for col in range(0,300,10):
            chosen_columns = ['xt{}'.format(i) for i in range(col,col+10)]
            target_column = ['hr']
            val_columns = ['hr']+chosen_columns
            
            if experiment_name=='normal':
                file_name = 'uwb_hr_normal.csv'
            elif experiment_name=='abnormal':
                file_name = 'uwb_hr_abnormal.csv'
            else:
                file_name = 'uwb_hr_rest.csv'
        
            raw_data = pd.read_csv(os.path.join(data_folder, file_name), dtype={'hr': float})
            data = raw_data[['id', 'time', 'hr'] + chosen_columns]     
            
            # split into train and test 
            train, test = split_data(data)
            
            # scaled the heart rate and uwb signal
            scaled_train, scaled_test = train.copy(), test.copy()
            input_scaler = sklearn.preprocessing.StandardScaler().fit(train[chosen_columns])
            target_scaler = sklearn.preprocessing.StandardScaler().fit(train[['hr']])
            scaled_train[chosen_columns] = input_scaler.transform(train[chosen_columns])
            scaled_train[target_column] = target_scaler.transform(train[target_column])
            scaled_test[chosen_columns] = input_scaler.transform(test[chosen_columns])
            scaled_test[target_column] = target_scaler.transform(test[target_column])
           
            model = ModelClass(quantile = q)
            train_data = batch_data(scaled_train, val_columns)
            test_data = batch_data(scaled_test, val_columns)
        
            model.fit(train_data)
            output_map = model.predict(test_data)
            targets = target_scaler.inverse_transform( output_map['targets'] )
            all_targets = target_scaler.inverse_transform( output_map['all_targets'] )
            predictions = target_scaler.inverse_transform( output_map['predictions'] )
                
            test_loss = model.evaluate(targets, predictions)
            tmp_loss.append(test_loss)
            # t=13
            # y1 = all_targets[t,:20]
            # x1 = np.array(range(len(y1)))
            # y2 = targets[t,:]
            # x2 = np.array(range(len(y1), len(y1)+len(y2)))
            # y3 = predictions[t,:]
            # x3 = np.array(range(len(y1), len(y1)+len(y3)))
            # plt.plot(x1, y1)
            # plt.plot(x2, y2)
            # plt.plot(x3, y3)
            # plt.ylim(0,170)
        all_losses['p{}'.format(int(q*100))] = np.array(tmp_loss)
    all_experiments_losses.append(all_losses)

# hkl.dump(all_experiments_losses, 'all_experiments_losses.hkl', mode='w')
all_result = hkl.load('all_experiments_losses.hkl')

dist = np.linspace(0, 195, num=30)

expr = 'rest'
p10_curve = all_result[expr]['p10']
p50_curve = all_result[expr]['p50']
p90_curve = all_result[expr]['p90']
plt.plot(dist, p10_curve, label='p10 quantile loss')
plt.plot(dist, p50_curve, label='p50 quantile loss')
plt.plot(dist, p90_curve, label='p90 quantile loss')
plt.legend()
plt.xlabel('XeThru distance [cm]')
plt.ylabel('Loss')

expr = 'abnormal'
p10_curve = all_result[expr]['p10']
p50_curve = all_result[expr]['p50']
p90_curve = all_result[expr]['p90']
plt.plot(dist, p10_curve, label='p10 quantile loss')
plt.plot(dist, p50_curve, label='p50 quantile loss')
plt.plot(dist, p90_curve, label='p90 quantile loss')
plt.legend()
plt.xlabel('XeThru distance [cm]')
plt.ylabel('Loss')

expr = 'normal'
p10_curve = all_result[expr]['p10']
p50_curve = all_result[expr]['p50']
p90_curve = all_result[expr]['p90']
plt.plot(dist, p10_curve, label='p10 quantile loss')
plt.plot(dist, p50_curve, label='p50 quantile loss')
plt.plot(dist, p90_curve, label='p90 quantile loss')
plt.legend()
plt.xlabel('XeThru distance [cm]')
plt.ylabel('Loss')

a = 7.49
b = 6.33
print( (a-b)/b*100 )