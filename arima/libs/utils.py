import numpy as np

def get_single_col_by_input_type(input_type, column_definition):
    l = [tup[0] for tup in column_definition if tup[1]==input_type]
    return l[0] # return the first string of column name

def get_cols_by_input_type(input_type, column_definition):
    return [ tup[0] for tup in column_definition if tup[1]==input_type ]

def tensorflow_quantile_loss(y, y_pred, quantile):
    prediction_underflow = y - y_pred
    q_loss = quantile * np.maximum(prediction_underflow, 0) + (1.-quantile) * np.maximum(-prediction_underflow, 0.)
    q_loss_arr = np.sum(q_loss, axis=-1)
    return np.mean(q_loss_arr)