import enum
import pandas as pd
import sklearn.preprocessing
import libs.utils as utils

class InputTypes(enum.IntEnum):
    TARGET = 0
    KNOWN_INPUT = 1
    ID = 2
    TIME = 3
        
class HeartRateFormatter():                            
        
    def __init__(self):
        self._time_steps = self.get_fixed_params()['total_time_steps']
        self._column_definition = [('id', InputTypes.ID), ('time', InputTypes.TIME), ('hr', InputTypes.TARGET)]

    def get_column_definition(self):
        column_definition = self._column_definition
        identifier = [ tup for tup in column_definition if tup[1] == InputTypes.ID ]
        time = [ tup for tup in column_definition if tup[1] == InputTypes.TIME]
        inputs = [ tup for tup in column_definition if tup[1] not in {InputTypes.ID, InputTypes.TIME} ]
        return identifier + time + inputs
        
    def _get_input_columns(self):
        return [ tup[0] for tup in self.get_column_definition() if tup[1] not in {InputTypes.ID, InputTypes.TIME} ]
    
    def split_data(self, df, train_boundary=15): # 15 people as train people
        index = df['id']
        df_list_train, df_list_test = [], []
        for one_id in index.unique():
            if len(df_list_train) < train_boundary:
                df_list_train.append( df[index==one_id] )
            else:
                df_list_test.append( df[index==one_id] )
                
        train = pd.concat(df_list_train).reset_index(drop=True)
        test = pd.concat(df_list_test).reset_index(drop=True)
        
        return (train, test)
         
    def _get_input_indices(self):            
        def _get_locations(input_types, defn):
            return [i for i, tup in enumerate(defn) if tup[1] in input_types]
            
        column_definition = [ tup for tup in self.get_column_definition() if tup[1] not in {InputTypes.ID, InputTypes.TIME} ] # remove ID and TIME columns
            
        # location excludes the first two columns ID and TIME (i.e. 'heart_rate' is column 0,...)
        locations = { 'input_size' : len(self._get_input_columns()),
                     'output_size' : len(_get_locations({InputTypes.TARGET}, column_definition)),
                     'input_obs_loc' : _get_locations({InputTypes.TARGET}, column_definition),
                     'known_inputs' : _get_locations({InputTypes.KNOWN_INPUT}, column_definition)
                     }
        return locations
        
    def get_fixed_params(self):
        fixed_params = {
            'total_time_steps' : 30, # use 20s to predict next 10s -> totally 30s
            'num_encoder_steps': 20,
            'num_epochs' : 100,
            'early_stopping_patience' : 5 }
        return fixed_params
        
    def get_experiment_params(self):
        params = self.get_fixed_params()
        params['column_definition'] = self.get_column_definition()
        params.update(self._get_input_indices()) # which column is known, target...
        return params
        
    def get_default_model_params(self):
        model_params = { 'dropout_rate' : 0.1, 'hidden_layer_size' : 160, 
                        'learning_rate' : 0.001, 'minibatch_size' :64, 
                        'max_gradient_norm' : 0.01, 'num_heads' : 4,
                        'stack_size' : 1 }
        return model_params