import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import libs.utils as utils
import hickle as hkl

import expt_settings.configs

import libs.seq2seq_model
ModelClass = libs.seq2seq_model.Sequence2Sequence
ExperimentConfig = expt_settings.configs.ExperimentConfig

all_test_loss = []
all_result = []

config = ExperimentConfig('normal')
data_formatter=config.make_data_formatter()
raw_data = pd.read_csv(config.data_csv_path, dtype={'hr': float})
data = raw_data[['id', 'time', 'hr']]
train, test = data_formatter.split_data(data)
fixed_params = data_formatter.get_experiment_params()
params = data_formatter.get_default_model_params()
fixed_params['num_epochs'] = 20000
fixed_params['hidden_layer_size'] = 5
        
params.update(fixed_params)
        
model = ModelClass(params)
if not model.training_data_cached():
    model.cache_batched_data(train, 'train')
    model.cache_batched_data(test, 'test')
model.fit(data_formatter=data_formatter)
output_map = model.predict()
targets = output_map['targets']
all_targets = output_map['all_targets']
predictions = output_map['predictions']
    
test_loss = model.evaluate(targets, predictions)
  
t=2
y1 = all_targets[t,:20]
x1 = np.array(range(len(y1)))
y2 = targets[t,:]
x2 = np.array(range(len(y1), len(y1)+len(y2)))
y3 = predictions[t,:]
x3 = np.array(range(len(y1), len(y1)+len(y3)))
plt.plot(x1, y1)
plt.plot(x2, y2)
plt.plot(x3, y3)
plt.ylim(0,170)

#     all_result.append(result)

# hkl.dump(all_result, 'result_3.hkl', mode='w')
# testt = hkl.load('result_3.hkl')

all_loss = []
for i in range(len(testt)):
    all_loss.append( testt[i]['loss'] )
all_loss = np.stack(all_loss)

# # plot loss graph
# plt.plot(all_loss[:,0])
# plt.plot(all_loss[:,1])
# plt.plot(all_loss[:,2])
# plt.legend(['10% quantile loss', '50% quantile loss', '90% quantile loss'])
# plt.xlabel('Xethru distance')
# plt.ylabel('Quantile loss')
# plt.xticks([])

# col = 100
# res = all_result[27]
# plt.plot( np.array(range(20)), res['all_targets'][col, :20, 0] )
# plt.plot( np.array(range(20, 30)), res['targets'].values[col, 2:] )
# plt.plot( np.array(range(20, 30)), res['p90'].values[col, 2:] )
# plt.legend(['Historical values', 'Target values', 'Predicted values'])
# plt.ylim(0, 170)
# plt.xlabel('Time [s]')
# plt.ylabel('Heart rate [ppm]')
# plt.show()