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
    # ... (logic from v0.py, but rename 'sev' -> 'target_level') ...
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

# compute_ucut_curve
def compute_ucut_curve(df, time_index, response_targets):
    T = (time_index.iloc[-1] - time_index.iloc[0]).total_seconds() / 3600.0
    ucut = {}
    for target in response_targets:
        durs = df[df['Response Level'] == target]['Duration (h)'].values
        
        if durs.size == 0:
            ucut[target] = (np.array([0, 1]), np.array([0, 0]))
            continue
            
        sorted_durs = np.sort(durs)[::-1]
        percents = sorted_durs / T
        cumulative = np.cumsum(percents)
        
        x = [0]
        y = [sorted_durs[0]]
        for i in range(len(sorted_durs)):
            x.append(cumulative[i])
            x.append(cumulative[i])
            y.append(y[-1])  # horizontal
            if i + 1 < len(sorted_durs):
                y.append(sorted_durs[i + 1])  # step down
            else:
                y.append(0)  # final drop
                
        ucut[target] = (np.array(x), np.array(y))
        
    return ucut

# compute_static_ucut 
def compute_static_ucut(stressor_values, time_index, threshold):
    """
    Compute UCUT curve for a fixed threshold.
    """
    mask = stressor_values >= threshold 

    df = generate_event_summary(time_index, {threshold: mask})
    df.columns = ["Start", "End", "SEV", "Duration (h)"]

    # Extract durations
    durs = df['Duration (h)'].values
    if durs.size == 0:
        return np.array([0, 1]), np.array([0, 0])

    # Total time span of dataset
    T = (time_index.iloc[-1] - time_index.iloc[0]).total_seconds() / 3600.0

    # Step 1: Sort durations descending
    sorted_durs = np.sort(durs)[::-1]

    # Step 2: Compute % contribution of each event
    percents = sorted_durs / T
    cumulative = np.cumsum(percents)

    # Step 3: Build horizontal-step curve
    x = [0]
    y = [sorted_durs[0]]
    for i in range(len(sorted_durs)):
        x.append(cumulative[i])
        x.append(cumulative[i])
        y.append(y[-1])  # horizontal step
        y.append(sorted_durs[i + 1] if i + 1 < len(sorted_durs) else 0)

    return np.array(x), np.array(y)
