from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import libs.utils as utils
import data_formatters.heart_rate

InputTypes = data_formatters.heart_rate.InputTypes

class TFTDataCache():
    _data_cache = {}
    
    @classmethod
    def update(cls, data, key):
        cls._data_cache[key] = data
    
    @classmethod
    def get(cls, key):
        return cls._data_cache[key].copy()
    
    @classmethod
    def contains(cls, key):
        return key in cls._data_cache    

class ArimaModel():
    def __init__(self, raw_params):
        params = dict(raw_params)
        self.time_steps = int(params['total_time_steps']) # 30s        
        self._input_obs_loc = params['input_obs_loc']
        self.column_definition = params['column_definition']        
        self.minibatch_size = int(params['minibatch_size'])
        self.num_epochs = int(params['num_epochs'])
        self.early_stopping_patience = int(params['early_stopping_patience'])
        self.num_encoder_steps = int(params['num_encoder_steps'])       
        self.order = params['order']
        self.quantiles = [0.1, 0.5, 0.9]
                
    def _get_single_col_by_type(self, input_type):
        return utils.get_single_col_by_input_type(input_type, self.column_definition)
    
    def training_data_cached(self):
        return TFTDataCache.contains('train_test')
    
    def _batch_data(self, data):
        
        def _batch_single_entity(input_data):
            time_steps = len(input_data)
            lags = self.time_steps # duration that we are going to trim down to
            x = input_data.values 
            if time_steps >= lags:
                return np.stack( [x[i:time_steps-(lags-1)+i] for i in range(lags)], axis=1 )
            else:
                return None
        
        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        
        data_map = {}
        for _, sliced in data.groupby(id_col):
            col_mappings = {'identifier': [id_col], 'time': [time_col], 'targets': [target_col]} # for ARIMA input is first 20s, target is next 10s
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
        data_map['inputs'] = data_map['targets'][:, :self.num_encoder_steps, 0]
        data_map['outputs'] = data_map['targets'][:, self.num_encoder_steps:, 0]
        return data_map 
    
    def cache_batched_data(self, data, cache_key):
        TFTDataCache.update(self._batch_data(data), cache_key)
    
    def fit(self):
        batched_data = TFTDataCache.get('batched_data')
        
        # fit 1 ARIMA for each sample
        self.model_list = {}
        for i in range( batched_data['targets'].shape[0] ):
            model = ARIMA(batched_data['inputs'][i,:], order=self.order)        
            model_fit = model.fit()
            self.model_list[ 'model_{}'.format(i) ] = model_fit
        self.n_models = batched_data['targets'].shape[0]
        
    def predict(self):
        batched_data = TFTDataCache.get('batched_data')
        fc_res = []
        for i in range(self.n_models):
            fc = self.model_list['model_{}'.format(i)].forecast(steps=self.time_steps-self.num_encoder_steps)
            fc_res.append(fc)
        
        res = {'inputs' : batched_data['inputs'],
               'outputs' : batched_data['outputs'],
               'predictions' : np.stack(fc_res)
               }
        
        return res
    
    def evaluate(self, y_true, y_pred):
        quantiles = self.quantiles
        loss = []
        for quantile in quantiles:
            loss.append( utils.tensorflow_quantile_loss( y_true, y_pred, quantile) )
        return loss