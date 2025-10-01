import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import libs.utils as utils
import hickle as hkl

import expt_settings.configs

import libs.tft_model
ModelClass = libs.tft_model.TemporalFusionTransformer
ExperimentConfig = expt_settings.configs.ExperimentConfig

all_test_loss = []
all_result = []

for i_column in range(0, 300, 10):
#for i_column in [20]:
    chosen_columns = np.array( range(i_column, i_column+10) )   
    config = ExperimentConfig('rest')
    data_formatter=config.make_data_formatter(chosen_columns)
    use_testing_mode=True
    raw_data = pd.read_csv(config.data_csv_path, dtype={'hr': float})
    data = raw_data[['id', 'time', 'hr'] + [ 'xt{}'.format(i_column) for i_column in chosen_columns ]]
    train, test = data_formatter.split_data(data)
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    if use_testing_mode:
        fixed_params['num_epochs'] = 100
        fixed_params['hidden_layer_size'] = 5
        
    params.update(fixed_params)
        
    model = ModelClass(params)
    if not model.training_data_cached():
        model.cache_batched_data(train, 'train')
        model.cache_batched_data(test, 'test')
    model.fit(data_formatter=data_formatter)
    test_loss = model.evaluate(data_formatter=data_formatter)
    output_map = model.predict()
    targets = output_map['targets']
    all_targets = output_map['all_targets']
    p10_forecast = data_formatter.format_predictions(output_map['p10'])
    p50_forecast = data_formatter.format_predictions(output_map['p50'])
    p90_forecast = data_formatter.format_predictions(output_map['p90'])
    
    test_loss = []
    y_true = targets.values[:, 2:]
    y_pred = p10_forecast.values[:,2:]
    test_loss.append(utils.numpy_quantile_loss(y_true, y_pred, 0.1))
    
    y_pred = p50_forecast.values[:,2:]
    test_loss.append(utils.numpy_quantile_loss(y_true, y_pred, 0.5))
    
    y_pred = p90_forecast.values[:,2:]
    test_loss.append(utils.numpy_quantile_loss(y_true, y_pred, 0.9))
    
    result = {'targets' : targets, 'all_targets': all_targets, 'p10': p10_forecast, 'p50': p50_forecast, 'p90': p90_forecast, 'loss': np.array(test_loss)}

    all_result.append(result)

hkl.dump(all_result, 'result_3.hkl', mode='w')
testt = hkl.load('result_3.hkl')

all_loss = []
for i in range(len(all_result)):
    all_loss.append( all_result[i]['loss'] )
all_loss = np.stack(all_loss)

# plot loss graph
plt.plot(all_loss[:,0])
plt.plot(all_loss[:,1])
plt.plot(all_loss[:,2])
plt.legend(['10% quantile loss', '50% quantile loss', '90% quantile loss'])
plt.xlabel('Xethru distance')
plt.ylabel('Quantile loss')
plt.xticks([])

col = 100
res = all_result[27]
plt.plot( np.array(range(20)), res['all_targets'][col, :20, 0] )
plt.plot( np.array(range(20, 30)), res['targets'].values[col, 2:] )
plt.plot( np.array(range(20, 30)), res['p90'].values[col, 2:] )
plt.legend(['Historical values', 'Target values', 'Predicted values'])
plt.ylim(0, 170)
plt.xlabel('Time [s]')
plt.ylabel('Heart rate [ppm]')
plt.show()