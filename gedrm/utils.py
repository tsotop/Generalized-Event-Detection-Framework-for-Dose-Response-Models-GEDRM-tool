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
    csv_options = data_config.get('read_csv_options', {})
    df = pd.read_csv(data_config['filepath'], **csv_options)
    
    dt_col = data_config['datetime_col']
    dt_format = data_config.get('datetime_format', None) # Get new option
    
    try:
        if dt_format:
            # Option 1: User provided the *exact* format
            print(f"Using user-specified datetime format: {dt_format}")
            df[dt_col] = pd.to_datetime(df[dt_col], format=dt_format)
        else:
            # Option 2: Auto-detect (fast for standard formats)
            df[dt_col] = pd.to_datetime(df[dt_col], infer_datetime_format=True)
            
    except (ValueError, TypeError) as e:
        print(f"\n--- DATETIME PARSING ERROR ---")
        print(f"Error: Could not parse the datetime column '{dt_col}'.")
        print(f"Pandas Error: {e}")
        print("\nIf your datetimes are not in a standard format (like YYYY-MM-DD),")
        print("please specify the *exact* format in your 'config.yaml'.")
        print("Example:")
        print("  data:")
        print("    datetime_format: '%d/%m/%Y %H:%M'")
        print("\nSee https://strftime.org/ for all format codes.")
        raise e

    df.sort_values(dt_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    time_index = df[dt_col]
    stressor = df[data_config['stressor_col']].values.astype(np.float64)
    
    # Get the replacement value. Default to 'None' (disabled) if not found.
    replacement_val = data_config.get('replace_zeros_with', None)
    
    # Conditionally apply the replacement
    if replacement_val is not None:
        print(f"Replacing 0 values with {replacement_val}")
        stressor = np.where(stressor == 0, replacement_val, stressor)
    
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