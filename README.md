# Generalized Event Detection Framework for Dose–Response Models (GEDRM)

This repository contains the Python implementation of the Generalized Event Detection Framework for Dose–Response Models (GEDRM). GEDRM is a computational approach designed to identify, quantify, and visualize biologically relevant exposure events in environmental stressor time series.

## Description

The Generalized Event Detection Framework for Dose–Response Models (GEDRM) is a computational approach designed to identify, quantify, and visualize biologically relevant exposure events in environmental stressor time series. Originally implemented for turbidity and suspended sediment concentration (SSC) using the Severity of Ill Effect (SEV) model by Newcombe and Jensen (1996), GEDRM generalizes the concept of dose–response modeling by combining dynamic, duration-dependent thresholds with efficient event detection algorithms.

The framework inverts empirical dose–response relationships to derive stressor thresholds as a function of exposure duration, allowing detection of periods where the stressor remains continuously above these biologically meaningful limits. Using a Range Minimum Query (RMQ) structure, GEDRM efficiently scans long time series to extract all valid exceedance events, which are then summarized in terms of duration, magnitude, and cumulative dose. These results are synthesized through diagnostic visualizations, including exceedance timelines, UCAT (Uniform Continuous Above Threshold) and UCUT (Uniform Continuous Under Threshold) curves, and stressor–duration threshold plots.

Altogether, GEDRM provides a transparent, **model-agnostic**, and **scalable** framework for analyzing chronic and episodic exposure dynamics in aquatic ecosystems.

## Features

* **Model-Agnostic:** Decouples the detection engine from the dose-response model. Plug in any model (e.g., SEV, toxicity curves) by implementing a simple Python class.
* **High Performance:** Uses a `numba`-accelerated Range Minimum Query (RMQ) algorithm to scan large time series (millions of data points) in seconds.
* **Flexible Configuration:** All parameters—data inputs, model selection, analysis settings, and plot styling—are controlled via a single `config.yaml` file.
* **Rich Visualizations:** Automatically generates:
    * Stressor time series plots (with optional discharge).
    * Exceedance event timelines.
    * UCAT/UCUT (Uniform Continuous Above/Under Threshold) curves to summarize event frequency and duration.

## Project Structure
<pre>
gedrm_project/
│
├── gedrm/                      # The core framework library (PACKAGE)
│   ├── __init__.py
│   ├── analysis.py             # Event summarization, UCUT logic
│   ├── core.py                 # RMQ and Numba-accelerated event detection
│   ├── model_interface.py      # Abstract class for all models
│   ├── plotting.py             # All plotting functions
│   └── utils.py                # Data and config loading helpers
│
├── models/                     # User-defined model plugins (PACKAGE)
│   ├── __init__.py
│   └── sev.py                  # Example: The SEV model
│
├── data/                       # Folder for input time series data
│   └── your_data.csv
│
├── results/                    # Default folder for output plots and CSVs
├── main.py                     # The main script to run the analysis
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
└── README.md                   # This file
</pre>

---

## Installation

This framework was developed and tested with **Python 3.8** and newer. Using a virtual environment is strongly recommended.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/tsotop/gedrm-framework.git](https://github.com/tsotop/gedrm-framework.git)
    cd gedrm-framework
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment (use your specific python version)
    python3.8 -m venv venv
    
    # Activate on macOS/Linux
    source venv/bin/activate
    
    # Or activate on Windows
    # venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

All analysis is run from the `main.py` script, which is configured by `config.yaml`.

1.  **Edit `config.yaml`:** Open the file and edit the sections to point to your data, select your model, and set your analysis parameters. See the "Configuration" section below for details.

2.  **Run the analysis:**
    ```bash
    python main.py
    ```

3.  **Check your results:** All outputs (plots and `Event_Summary.csv`) will be saved to the `output_dir` specified in your config (default: `results/`).

---

## Configuration (`config.yaml`)

This file controls the entire analysis.

```yaml
# 1. DATA INPUT
data:
  filepath: "data/your_data.csv"
  datetime_col: "datetime"    # Column name for datetime
  stressor_col: "ssc"         # Column name for stressor data
  discharge_col: "q"          # (Optional) Column for discharge
  datetime_format: null       # (Optional) e.g., "%d/%m/%Y %H:%M"
  replace_zeros_with: 0.025   # (Optional) Set to null to disable
  read_csv_options:           # (Optional) Any pd.read_csv options
    comment: '#'              # e.g., if your CSV has comments

# 2. MODEL SELECTION
model:
  module: "models.sev"        # Python import path to the model file
  class: "SEVModel"           # Class name inside the file
  params:                     # Parameters for the model's __init__
    group: 1                  # e.g., fish group for SEV

# 3. ANALYSIS PARAMETERS
analysis:
  response_targets: [1, 2, 3, 4, 5] # List of response levels
  max_duration_hours: null        # (Optional) Max duration to scan
  baseline_percentile: null       # (Optional) Percentile to filter low thresholds
  moving_avg_window: 1            # (Optional) Set > 1 for smoothing
  static_threshold: 25            # (Optional) A static threshold for UCUT comparison

# 4. OUTPUT & PLOTTING
plotting:
  output_dir: "results/"
  colormap: "binary"
  stressor_label: "Stressor"
  stressor_units: "mg/L"
  response_label: "Response Level"
  discharge_units: "m³/s"
```
  
## How to Add a New Dose-Response Model
This framework is designed for extensibility. To add your own model:

1. *Create a Model File:* Create a new file in the models/ directory (e.g., `models/my_toxicity_model.py`).

2. *Implement the Interface:* In your new file, import the `DoseResponseModel` and create a class that inherits from it. You must implement three methods (`validate_params, compute_stressor_threshold`) and one property (`name`).

For example
```my model
# models/my_toxicity_model.py
import numpy as np
from gedrm.model_interface import DoseResponseModel

class MyToxicityModel(DoseResponseModel):
    """
    Implementation of a custom toxicity model.
    Model: Response = k * Duration * Stressor
    """

    def validate_params(self):
        """Check if all required params were passed from config."""
        if 'k_factor' not in self.params:
            raise ValueError("MyTToxicityModel requires 'k_factor' in config.")
        self.k = self.params['k_factor']

    def compute_stressor_threshold(self, response_level: float, duration_hours: np.ndarray) -> np.ndarray:
        """
        Invert the model to solve for Stressor.
        Stressor = Response / (k * Duration)
        """
        # Avoid division by zero if duration is 0
        safe_duration = np.maximum(duration_hours, 1e-9)
        return response_level / (self.k * safe_duration)

    @property
    def name(self) -> str:
        """A short name for plots and logs."""
        return f"Toxicity Model (k={self.k})"
```
3. *Update* `config.yaml`: Point the `model` section to your new class and run, like so:.

```yaml
# 2. MODEL SELECTION
model:
  module: "models.my_toxicity_model"
  class: "MyToxicityModel"
  params:
    k_factor: 0.5  # Pass the 'k_factor' your new model needs
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
