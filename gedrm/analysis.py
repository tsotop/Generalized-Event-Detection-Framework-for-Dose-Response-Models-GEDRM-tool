# gedrm/analysis.py
import numpy as np
import pandas as pd
from .model_interface import DoseResponseModel # Import the contract

# build_thresholds
def build_thresholds(model: DoseResponseModel, response_targets, max_hours):
    """
    Builds the threshold matrix by calling the provided model.
    """
    durations = np.arange(1, max_hours + 1)
    thresholds = np.empty((len(response_targets), max_hours), dtype=np.float64)
    
    # Call the model's method
    for i, target in enumerate(response_targets):
        thresholds[i] = model.compute_stressor_threshold(target, durations)
        
    return thresholds, durations

# build_full_masks
def build_full_masks(start_flags, match_len, response_targets):
    # ... (logic from v0.py, but rename 'sev' -> 'target') ...
    n_targets, N = start_flags.shape
    masks = {}
    for i, target in enumerate(response_targets): # Renamed
        diff = np.zeros(N+1, dtype=np.int32)
        hits = np.nonzero(start_flags[i])[0]
        for t in hits:
            w = match_len[i, t]
            diff[t]     += 1
            diff[min(N, t+w)] -= 1
        masks[target] = np.cumsum(diff)[:N] > 0 # Use target as key
    return masks
    
# generate_event_summary
def generate_event_summary(time_index, masks):
    events = []
    for target_level, mask in masks.items(): # Renamed
        active = False
        for idx, val in enumerate(mask):
            if val and not active:
                t0, active = time_index.iloc[idx], True
            elif not val and active:
                t1 = time_index.iloc[idx-1]
                dur = (t1 - t0).total_seconds() / 3600
                events.append([t0, t1, target_level, dur]); active = False # Renamed
        if active:
            t1 = time_index.iloc[-1]
            dur = (t1 - t0).total_seconds() / 3600
            events.append([t0, t1, target_level, dur]) # Renamed
    return pd.DataFrame(events, columns=["Start","End","Response Level","Duration (h)"]) # Renamed

def _calculate_ucut_stepped_data(durations, T):

    if durations.size == 0:
        x_plot = np.array([0, 100]) # Use 100 for 100%
        y_plot = np.array([0, 0])
        df = pd.DataFrame({'Cumulative Duration (%)': x_plot, 'Time Above Response (h)': y_plot})
        return (x_plot, y_plot), df

    # Step 1: Sort durations descending
    sorted_durs = np.sort(durations)[::-1]

    # Step 2: Compute % contribution and cumulative
    percents = sorted_durs / T
    cumulative = np.cumsum(percents) * 100 # Convert to percentage

    # Step 3: Build horizontal-step curve (your exact logic)
    x = [0]
    y = [sorted_durs[0]]
    for i in range(len(sorted_durs)):
        x.append(cumulative[i])
        x.append(cumulative[i])
        y.append(y[-1])  # horizontal step
        y.append(sorted_durs[i + 1] if i + 1 < len(sorted_durs) else 0)
    
    x_plot = np.array(x)
    y_plot = np.array(y)

    # Create the DataFrame for saving
    df = pd.DataFrame({
        'Cumulative Duration (%)': x_plot,
        'Time Above Response (h)': y_plot
    })
    
    return (x_plot, y_plot), df


def compute_ucut_curve(df, time_index, response_targets):
    """
    Computes UCUT curves for all dynamic response targets.
    """
    T = (time_index.iloc[-1] - time_index.iloc[0]).total_seconds() / 3600.0
    ucut_dict = {}
    all_dfs = []
    
    for target in response_targets: 
        durs = df[df['Response Level'] == target]['Duration (h)'].values 
        
        (x, y), temp_df = _calculate_ucut_stepped_data(durs, T)
        
        ucut_dict[target] = (x, y)
        temp_df['Response Level'] = target
        all_dfs.append(temp_df)
        
    ucut_df = pd.concat(all_dfs, ignore_index=True)
    return ucut_dict, ucut_df

def compute_static_ucut(stressor_values, time_index, threshold):
    """
    Compute UCUT curve for a fixed threshold.
    """
    mask = stressor_values >= threshold
    # Use generic "Response Level" column name
    df = generate_event_summary(time_index, {threshold: mask}) 
    
    T = (time_index.iloc[-1] - time_index.iloc[0]).total_seconds() / 3600.0
    durs = df['Duration (h)'].values
    
    # Use the new helper function
    (x, y), static_df = _calculate_ucut_stepped_data(durs, T)
    
    return (x, y), static_df