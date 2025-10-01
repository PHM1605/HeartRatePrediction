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
        
    def __init__(self, chosen_columns):
        self._time_steps = self.get_fixed_params()['total_time_steps']
        self._column_definition = [('id', InputTypes.ID), ('time', InputTypes.TIME), ('hr', InputTypes.TARGET)] + \
                         [('xt{}'.format(i), InputTypes.KNOWN_INPUT) for i in chosen_columns]
    def set_scalers(self, df):
        column_definitions = self.get_column_definition()
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definitions)
        inputs = utils.get_cols_by_input_type(InputTypes.KNOWN_INPUT, column_definitions)
            
        # set scaler for each participant
        self._input_scalers = {}
        self._target_scaler = {}
            
        data = df[inputs].values # 2D array of float 
        target = df[[target_column]].values # 2D array of float (1 column)
        self._input_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(target)
        
    # apply scalers to input data
    def transform_inputs(self, df): 
        column_definitions = self.get_column_definition()
        inputs = utils.get_cols_by_input_type(InputTypes.KNOWN_INPUT, column_definitions)
            
        output = df.copy()
        output[inputs] = self._input_scalers.transform(output[inputs].values)

        return output
        
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
        self.set_scalers(train)
            
        return (self.transform_inputs(data) for data in [train, test])
        
    def format_predictions(self, predictions):
        column_names = predictions.columns
        output = predictions.copy()
        for col in column_names:
            if col not in {'forecast_time', 'identifier'}:
                output[col] = self._target_scaler.inverse_transform(output[col])        
        return output