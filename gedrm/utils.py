# gedrm/utils.py
import pandas as pd
import numpy as np
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_timeseries(data_config):
    """
    Loads and prepares the stressor time series from config.
    """
    # Get the read_csv options from the config.
    # Default to an empty dictionary if 'read_csv_options' is not provided.
    csv_options = data_config.get('read_csv_options', {})

    # Use dictionary unpacking to pass the options (e.g., comment='#')
    # to pandas.read_csv
    df = pd.read_csv(data_config['filepath'], **csv_options)
    df[data_config['datetime_col']] = pd.to_datetime(df[data_config['datetime_col']])
    df.sort_values(data_config['datetime_col'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    time_index = df[data_config['datetime_col']]
    stressor = df[data_config['stressor_col']].values.astype(np.float64)
    
    # Handle zeros
    stressor = np.where(stressor == 0, 0.025, stressor)
    
    discharge = None
    if data_config.get('discharge_col') in df.columns:
        discharge = df[data_config['discharge_col']].values.astype(np.float64)
        
    dt = (time_index.iloc[1] - time_index.iloc[0]).total_seconds() / 3600
    if dt <= 0:
        raise ValueError('Non-positive or invalid time step (dt)')
        
    return time_index, stressor, dt, discharge

def apply_smoothing(stressor_values, window_samples):
    """Applies a moving average."""
    return np.convolve(
        stressor_values, 
        np.ones(window_samples) / window_samples, 
        mode='same'
    )