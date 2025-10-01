import pandas as pd
import numpy as np

for i_column in range(0, 300, 10):
    chosen_columns = np.array( range(i_column, i_column+10) )   
    config = ExperimentConfig('rest')
    data_formatter=config.make_data_formatter(chosen_columns)
    raw_data = pd.read_csv(config.data_csv_path, dtype={'hr': float})
    data = raw_data[['id', 'time', 'hr'] + [ 'xt{}'.format(i_column) for i_column in chosen_columns ]]
    train, test = data_formatter.split_data(data)