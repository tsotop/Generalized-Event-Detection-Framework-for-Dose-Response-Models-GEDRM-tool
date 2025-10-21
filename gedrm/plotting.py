# gedrm/plotting.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from .analysis import generate_event_summary 

# plot_ssc
def plot_stressor_timeseries(time_index, stressor, discharge=None, 
                             stressor_label='Stressor', discharge_label='Discharge',
                             output_path=None):
    fig, ax1 = plt.subplots(figsize=(7, 3), dpi=300)

    if discharge is not None:
        ax2 = ax1.twinx()
        ax2.plot(time_index, discharge, color='gray', lw=0.8, label='Discharge', zorder=1, alpha=0.8)
        ax2.set_ylabel(discharge_label, color='gray') # Use label
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.set_zorder(1)  
        ax2.patch.set_alpha(0)  

    ax1.plot(time_index, stressor, color='black', lw=0.8, label='Stressor', zorder=2) # Renamed
    ax1.set_ylabel(stressor_label, color='black') # Use label
    ax1.tick_params(axis='y', labelcolor='black')
    
    # X-axis formatting
    total_months = (time_index.iloc[-1] - time_index.iloc[0]).days / 30.0
    for iv in [1, 2, 3, 4, 6, 12]:
        if total_months / iv <= 7:
            interval_months = iv
            break
    locator = mdates.MonthLocator(interval=interval_months)
    formatter = mdates.DateFormatter('%b-%Y')
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

# plot_exceedance_and_ucut 
def plot_exceedance_and_ucut(time_index, masks, ucut, response_targets, colormap,
                             static_uc=None, static_label=None,
                             response_label='Response Level',
                             output_path=None):
    """
    Creates the side-by-side plot for exceedance and UCUT curves.
    """
    # This is your second, more advanced plot function from v0.py
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3), dpi=300)

    cmap = plt.get_cmap(colormap, len(response_targets) + 1)

    # --- Left: Exceedance ---
    for i, target in enumerate(response_targets): # Renamed
        events = generate_event_summary(time_index, {target: masks[target]})
        for _, row in events.iterrows():
            ax1.hlines(y=target, # Renamed
                       xmin=row['Start'],
                       xmax=row['End'],
                       color=cmap(i + 1),
                       linewidth=10)

    ax1.set_ylim(min(response_targets) - 0.5, max(response_targets) + 0.5)
    ax1.set_yticks(response_targets)
    ax1.set_ylabel(response_label) # Use label

    total_days = (time_index.iloc[-1] - time_index.iloc[0]).days
    total_months = total_days / 30.0
    for iv in [1, 2, 3, 4, 6, 12]:
        if total_months / iv <= 4:
            interval_months = iv
            break
    locator = mdates.MonthLocator(interval=interval_months)
    formatter = mdates.DateFormatter('%b-%Y')
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # --- Right: UCUT curves ---
    for i, target in enumerate(response_targets): # Renamed
        x, y = ucut.get(target, (None, None))
        if x is None or x.size == 0 or np.all(y == 0):
            continue
        ax2.step(x * 100, y,
                 where='post',
                 color=cmap(i + 1),
                 linewidth=1.5,
                 label=f'{response_label} {target}') # Use label

    if static_uc is not None:
        xs, ys = static_uc
        ax2.step(xs * 100, ys,
                 where='post',
                 color='blue',
                 linewidth=1.5,
                 label=static_label)
    
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_ylabel('Time Above Response (h)') # Generalized
    ax2.set_xlabel('Cumulative Duration (%)')
    
    # Legend: shared across bottom
    handles, labels = ax2.get_legend_handles_labels()
    ncol = min(len(labels), 5)
    ax2.legend(handles, labels,
               ncol=ncol,
               loc='lower center',
               bbox_to_anchor=(-0.20, -0.45),
               frameon=False)

    fig.subplots_adjust(wspace=0.35)
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
