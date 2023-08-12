""" 
           .__.__                
      ____ |__|  |  __ _____  ___
     /    \|  |  | |  |  \  \/  /
    |   |  \  |  |_|  |  />    < 
    |___|  /__|____/____//__/\_ \
         \/                    \/


"""
import pandas as pd
import numpy as np
from pandas import DataFrame   

def column_builder(dataframe: DataFrame, column_config: dict, producers: list, drop_others: bool = False):
    """ 
    An easy way to work with producer columns and remove/add suffixes... 
    Various functions to unify, summarise, rename, manipulate or do other calculations on producer columns.

    :param dataframe: The input DataFrame.
    :param column_config: Dict with column names, mode and suffix.
    :param producers: List with producer names
    :param drop_others: Drop the remaining producer columns
    :return: Modified DataFrame with new columns.
    """

    created_columns = set()

    for unified_col, config in column_config.items():
        columns = config['columns']
        mode = config['mode']
        config_keep_suffix = config.get('keep_suffix', False)
        config_rolling = config.get('rolling', 1)

        for producer in producers:
            possible_columns = [f"{col}_{producer}" for col in columns]
            valid_columns = [col for col in possible_columns if col in dataframe.columns]

            if valid_columns:
                new_col_name = f"{unified_col}_{producer}" if config_keep_suffix else f"{unified_col}"
                new_col_name = new_col_name.rstrip('_')

                if mode in ['max', 'min', 'mean', 'median', 'sum', 'std', 'var', 'skew', 'kurt']:
                    rolled = dataframe[valid_columns].rolling(window=config_rolling)
                    result = getattr(rolled, mode)()
                    result = result.max(axis=1)

                elif mode == 'mean_max':
                    result = dataframe[valid_columns].rolling(window=config_rolling).max()
                    result = np.mean(result.values, axis=1)

                elif mode == 'mean_min':
                    result = dataframe[valid_columns].rolling(window=config_rolling).min()
                    result = np.mean(result.values, axis=1)

                elif mode == 'quantile':
                    quantile_val = config.get('quantile_value', 0.5)
                    result = dataframe[valid_columns].rolling(window=config_rolling).quantile(quantile_val)
                    result = result.max(axis=1)

                elif mode == 'dir':
                    rolling_diff = dataframe[valid_columns].diff().rolling(window=config_rolling).sum().sum(axis=1)
                    result = ((rolling_diff > 0) * 1) + ((rolling_diff < 0) * -1)

                elif mode in ['above', 'below']:
                    rolling_means = dataframe[valid_columns].rolling(window=config_rolling).mean()
                    overall_mean = rolling_means.mean(axis=1)
                    
                    limit = config.get('limit', float(0.0))
                    if mode == 'above':
                        result = (overall_mean > limit).astype(int)
                    else:
                        result = (overall_mean < limit).astype(int)

                dataframe[new_col_name] = result
                created_columns.add(new_col_name)
                
    if drop_others:
        columns_to_drop = [col for col in dataframe.columns 
                        if any(col.endswith(f"_{producer}") for producer in producers)]
        
        dataframe.drop(columns=columns_to_drop, inplace=True)

    return dataframe


""" 
Column Builder Example:

producers = ['ft_2', 'ft_5', 'ft_6', 'ft_7','ft_8']

column_config = {
    'enter_long_combined': {
        'columns': ['buy_signal', 'enter_long', 'prediction_up'],
        'mode': 'max',
        'rolling': 2
    },
    'enter_short_combined': {
        'columns': ['sell_signal', 'enter_short', 'prediction_down'],
        'mode': 'max',
        'rolling': 2
    },
    'model_predict': {
        'columns': ['target_close', 'predicted_price'],
        'mode': 'above',
        'limit': dataframe['close'],
        'rolling': 3
    },
    'avg_pred_high': {
        'columns': ['stop_high', 'pred_high'],
        'mode': 'mean',
        'rolling': 3
    },
    'avg_pred_low': {
        'columns': ['stop_low', 'pred_low'],
        'mode': 'mean',
        'rolling': 3
    }
}

dataframe = column_builder(dataframe, column_config, producers)

"""
