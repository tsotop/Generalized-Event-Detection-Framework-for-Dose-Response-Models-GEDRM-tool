# main.py
import time
import importlib
import numpy as np
import os

# Import gedrm library functions
import gedrm.utils
import gedrm.core
import gedrm.analysis
import gedrm.plotting

def main():
    start_time = time.time()
    
    # 1. Load Configuration
    config_path = 'config.yaml'
    cfg = gedrm.utils.load_config(config_path)

    # 2. Load and Prepare Data
    print(f"Loading data from {cfg['data']['filepath']}...")
    time_index, stressor, dt, discharge = gedrm.utils.load_timeseries(cfg['data'])

    # Optional smoothing (if specified in config)
    if cfg['analysis'].get('moving_avg_window', 1) > 1:
        stressor = gedrm.utils.apply_smoothing(
            stressor, cfg['analysis']['moving_avg_window']
        )

    # 3. Dynamically Load the Model Plugin
    print(f"Loading model: {cfg['model']['module']}.{cfg['model']['class']}...")
    try:
        ModelModule = importlib.import_module(cfg['model']['module'])
        ModelClass = getattr(ModelModule, cfg['model']['class'])
        # Pass model-specific params (e.g., {'group': 1}) to its constructor
        model = ModelClass(**cfg['model']['params']) 
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Set Up Analysis Parameters
    response_targets = cfg['analysis']['response_targets']
    
    # Compute baseline (or disable)
    baseline_pct = cfg['analysis'].get('baseline_percentile')
    baseline = -1.0 if baseline_pct is None else np.percentile(stressor, baseline_pct)

    # Determine max duration to scan
    full_T = int(np.floor((time_index.iloc[-1] - time_index.iloc[0]).total_seconds() / 3600))
    cap_T_config = cfg['analysis'].get('max_duration_hours')
    max_hours = full_T if cap_T_config is None else min(full_T, int(cap_T_config))

    # 5. Build Thresholds (using the model)
    print("Building duration-threshold matrix...")
    thresholds, durations = gedrm.analysis.build_thresholds(
        model, response_targets, max_hours
    )
    # Convert duration hours to sample window sizes
    window_sizes = np.round(durations / dt).astype(np.int32)
    
    # 6. Run Detection Engine
    print("Building sparse table (RMQ)...")
    st, log2 = gedrm.core.build_sparse_table(stressor)

    print("Scanning time series for events...")
    start_flags, match_len = gedrm.core.detect_exceedance_events(
        stressor, thresholds, window_sizes, st, log2, baseline
    )

    # 7. Post-Process and Summarize
    print("Summarizing events...")
    full_masks = gedrm.analysis.build_full_masks(start_flags, match_len, response_targets)
    summary_df = gedrm.analysis.generate_event_summary(time_index, full_masks)
    
    # Save summary
    output_dir = cfg['plotting']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'event_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved event summary to {summary_path}")

    # 8. Compute UCUT Curves
    ucut, ucut_df = gedrm.analysis.compute_ucut_curve(summary_df, time_index, response_targets)
    
    # Compute static threshold UCUT (if specified)
    static_uc = None
    static_label = None
    if cfg['analysis'].get('static_threshold') is not None:
        static_thresh = cfg['analysis']['static_threshold']

        # Capture the new DataFrame
        static_uc, static_uc_df = gedrm.analysis.compute_static_ucut(stressor, time_index, static_thresh)

        static_label = f"≥{static_thresh} {cfg['plotting'].get('stressor_units', 'mg/L')}"

        # Add the static data to the main DataFrame
        static_uc_df['Response Level'] = static_label
        ucut_df = pd.concat([ucut_df, static_uc_df], ignore_index=True)
        
    # Save the combined UCUT data
    ucut_path = os.path.join(output_dir, 'UCUT_data.csv')
    ucut_df.to_csv(ucut_path, index=False, float_format='%.2f')
    print(f"Saved UCUT data to {ucut_path}")

    # 9. Plot Results
    print("Generating plots...")
    plot_cfg = cfg['plotting']
    
    # Plot 1: Stressor Timeseries
    gedrm.plotting.plot_stressor_timeseries(
        time_index, 
        stressor, 
        discharge=discharge,
        stressor_label=f"{plot_cfg['stressor_label']} ({plot_cfg.get('stressor_units', '')})",
        discharge_label=f"Discharge ({plot_cfg.get('discharge_units', 'm³/s')})",
        output_path=os.path.join(output_dir, 'stressor_timeseries.png')
    )
    
    # Plot 2: Combined Exceedance and UCUT
    gedrm.plotting.plot_exceedance_and_ucut(
        time_index, 
        full_masks, 
        ucut, 
        response_targets,
        static_uc=static_uc,
        static_label=static_label,
        colormap=plot_cfg['colormap'],
        response_label=plot_cfg['response_label'],
        output_path=os.path.join(output_dir, 'exceedance_ucut.png')
    )
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Analysis complete. Elapsed time: {elapsed:.2f} seconds")

if __name__ == '__main__':
    main()