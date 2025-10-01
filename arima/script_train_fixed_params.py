import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl

import expt_settings.configs

import libs.arima_model
ModelClass = libs.arima_model.ArimaModel

ExperimentConfig = expt_settings.configs.ExperimentConfig

config = ExperimentConfig('normal')
data_formatter=config.make_data_formatter()
use_testing_mode=True
raw_data = pd.read_csv(config.data_csv_path, dtype={'hr': float})
data = raw_data[['id', 'time', 'hr']]
left_data, used_data = data_formatter.split_data(data)
fixed_params = data_formatter.get_experiment_params()
params = data_formatter.get_default_model_params()

params.update(fixed_params)
    
model = ModelClass(params)
if not model.training_data_cached():
    model.cache_batched_data(used_data, 'batched_data')
    
model.fit()
#test_loss = model.evaluate(data_formatter=data_formatter)
output_map = model.predict()

inputs = output_map['inputs']
outputs = output_map['outputs']
predictions = output_map['predictions']



test_loss = model.evaluate(outputs, predictions)
#hkl.dump(test_loss, 'result_arima.hkl', mode='w')

t=4
y1 = inputs[t,:]
x1 = np.array(range(len(y1)))
y2 = outputs[t,:]
x2 = np.array(range(len(y1), len(y1)+len(y2)))
y3 = predictions[t,:]
x3 = np.array(range(len(y1), len(y1)+len(y3)))

plt.plot(x1, y1)
plt.plot(x2, y2)
plt.plot(x3, y3)

plt.ylim(0,170)