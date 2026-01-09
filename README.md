# Forensic Cascade Inference via Neural Stick-Breaking

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)

This repository contains the official implementation for the KDD 2026 manuscript submission, "Forensic Cascade Inference via Neural Stick-Breaking". We introduce the Neural Stick-Breaking (NSB) process, a novel recursive generative model that learns complete probability distributions over countably infinite support. The NSB process uses a recurrent neural network to parameterize a stick-breaking construction, enabling flexible, data-driven modeling of transmission cascades in epidemiological outbreaks.

Our framework unifies **Neural** (learned transmission laws), **Symbolic** (probability generating function geometry), and **Spectral** (FFT-based one-pass algorithms) components to address three fundamental forensic questions:

- **Task "Who"**: Source attribution—identifying the number of patient zeros (founders) from observed cluster sizes using Bayesian inference with epidemiological priors
- **Task "How"**: Structural fingerprinting—quantifying attribution certainty through entropy analysis and extinction probability computation
- **Task "Next"**: Branching volatility—assessing superspreading risk and generational information decay through recursive PGF analysis

The spectral inversion engine achieves O(nK log(nK)) complexity, enabling real-time forensic inference even for large clusters (n > 10⁶), validated on real SARS/MERS transmission data.

## Features

-   **Neural Stick-Breaking Model:** A full PyTorch implementation of the NSB process with configurable hidden dimensions and identity-centered initialization for criticality control.
-   **Spectral Inversion Engine:** Efficient O(nK log(nK)) implementation of source attribution using FFT and the Neural Otter-Dwass identity.
-   **Forensic Task Modules:** 
    - **Task "Who"**: One-pass patient zero attribution with Bayesian priors (Flat, Poisson, Negative Binomial)
    - **Task "How"**: Structural fingerprinting with entropy, R0, and extinction probability computation
    - **Task "Next"**: Branching volatility analysis and generational PGF visualization
-   **Baselines:** Implementations of strong baselines, including Poisson MLE, Negative Binomial MLE, and Softmax Neural Networks.
-   **Reproducibility:** Complete scripts to reproduce all paper figures and experiments.
-   **Data Processing:** R and Python utilities to process the raw `outbreaktrees` dataset and extract founder counts.
-   **Testing:** Comprehensive unit tests for all core modules and algorithms.

## Project Structure

```
.
├── data/                          # Raw and processed data
│   ├── taube_2020_transmission_trees.rds    # Raw transmission tree data
│   ├── outbreaktrees_sars_mers_counts.csv   # Processed offspring counts
│   └── outbreaktrees_founder_counts.csv     # Extracted founder counts
├── experiments/                    # Experiment scripts and utilities
│   ├── exp_task_who.py            # Duality Validation Dashboard (Task "Who")
│   ├── exp_task_how_next.py       # Forensic Forecasting Dashboard (Tasks "How" & "Next")
│   ├── exp_outbreaktrees.py       # Real-world SARS/MERS analysis
│   ├── exp_synthetic.py           # Synthetic data validation
│   ├── exp_spectral.py            # Spectral analysis experiments
│   ├── exp_dynamics.py            # Dynamics and criticality analysis
│   ├── exp_complexity.py          # Computational complexity benchmarks
│   ├── exp_moment_analysis.py     # Moment analysis experiments
│   ├── exp_hidden_size.py         # Hidden dimension sensitivity
│   ├── plot_outbreaktrees.py      # Transmission tree visualization
│   ├── plot_synthetic.py          # Synthetic data visualization
│   ├── plot_utils.py              # Shared plotting utilities
│   ├── process_outbreaktrees.r    # R script to process raw data
│   ├── extract_founder_counts.r   # R script to extract founder counts
│   └── utils.py                    # Experiment utilities
├── figures/                        # Generated publication figures (PDF/PNG)
├── nsb/                            # Core NSB implementation
│   ├── model.py                   # NSB model class and training
│   ├── task_who.py                # Task "Who": Source attribution
│   ├── task_how.py                # Task "How": Structural metrics
│   ├── spectral_engine.py         # Spectral inversion engine (FFT-based)
│   ├── attention_model.py         # NSB with attention mechanism
│   ├── lstm_model.py              # NSB with LSTM cell
│   ├── gru_model.py               # NSB with GRU cell
│   ├── constrained_model.py       # Constrained NSB variants
│   ├── softmax_nn.py              # Softmax neural network baseline
│   ├── poisson.py                 # Poisson baseline
│   └── negative_binomial.py      # Negative Binomial baseline
├── tests/                          # Unit tests
│   ├── test_model.py              # NSB model tests
│   ├── test_task_who.py           # Source attribution tests
│   ├── test_task_how.py           # Structural metrics tests
│   ├── test_spectral_engine.py    # Spectral engine tests
│   └── ...                        # Additional test files
├── results/                        # Generated CSV results and LaTeX tables
├── pyproject.toml                  # Project dependencies and configuration
└── README.md                       # This file
```

## Installation and Setup

Follow these steps to set up the environment and run the code.

### 1. Clone the Repository

```bash
git clone https://anonymous.4open.science/r/nsb-kdd-revision
cd nsb-cascades
```

### 2. Create the Conda Environment

```bash
conda create --name nsb python=3.11
conda activate nsb
```

### 3. Install Python Dependencies

Install the `nsb` package and all necessary dependencies in editable mode.

```bash
pip install -e .
```

### 4. Install R and R Packages (for data processing)

To run the data processing scripts, you need to have R installed on your system.

1.  **Install R:** Download and install R from the [Comprehensive R Archive Network (CRAN)](https://cran.r-project.org/).
2.  **Add R to PATH:** Ensure that the `bin` directory of your R installation is added to your system's PATH environment variable.
3.  **Install R Packages:** Open your R console and run the following command to install the required packages:
    ```r
    install.packages(c("dplyr", "readr", "here", "igraph", "xtable", "Cairo"))
    ```
    
    Note: We use individual packages (`dplyr`, `readr`) instead of `tidyverse` for faster installation.

## Usage: Reproducing the Paper's Results

The entire experimental pipeline can be run from the command line.

### 1. Download the Data

Download the raw dataset (`taube_2020_transmission_trees.rds`) from [outbreaktrees.ecology.uga.edu](https://outbreaktrees.ecology.uga.edu/) and place it in the `data/` directory.

### 2. Process the Data

Run the R scripts to process the raw data and extract necessary information:

```bash
# Process raw transmission trees and generate offspring counts
Rscript experiments/process_outbreaktrees.r

# Extract founder counts (z) and cluster sizes (n) for source attribution
Rscript experiments/extract_founder_counts.r
```

This will generate:
- `data/outbreaktrees_sars_mers_counts.csv`: Offspring counts for training
- `data/outbreaktrees_founder_counts.csv`: Founder counts and cluster sizes for validation

### 3. Run the Main Experiments

The paper's main figures are generated by two key scripts:

```bash
# Generate Duality Validation Dashboard (Figure: Task "Who")
# This creates a 1x2 grid with:
# - Panel A: Forensic "Cold Case" Reconstruction (source attribution)
# - Panel B: Spectral Signature (PGF on unit circle)
# - Panel C: Computational Shield (scaling analysis)
python experiments/exp_task_who.py

# Generate Forensic Forecasting Dashboard (Figure: Tasks "How" & "Next")
# This creates a 2x2 grid with:
# - Panel A: Extinction Phase Transition
# - Panel B: Forensic Entropy and Information Horizon
# - Panel C: Branching Volatility (Superspreading Risk)
# - Panel D: Spectral Energy Dissipation (Generational PGF)
python experiments/exp_task_how_next.py
```

### 4. Run Additional Experiments

Other experiment scripts generate supplementary figures and analyses:

```bash
# Real-world SARS/MERS outbreak analysis
python experiments/exp_outbreaktrees.py

# Synthetic data validation
python experiments/exp_synthetic.py

# Spectral analysis and PGF visualization
python experiments/exp_spectral.py

# Dynamics and criticality analysis
python experiments/exp_dynamics.py

# Computational complexity benchmarks
python experiments/exp_complexity.py

# Moment analysis
python experiments/exp_moment_analysis.py

# Hidden dimension sensitivity
python experiments/exp_hidden_size.py
```

All figures are saved to the `figures/` directory in both PDF and PNG formats.

### 5. Run the Tests

To verify the correctness of the implementation, run the full suite of unit tests:

```bash
pytest
```

Or run specific test modules:

```bash
pytest tests/test_model.py           # Test NSB model
pytest tests/test_task_who.py        # Test source attribution
pytest tests/test_task_how.py        # Test structural metrics
pytest tests/test_spectral_engine.py # Test spectral engine
```

## Core Modules

### NSB Model (`nsb/model.py`)

The main `NSB` class implements the Neural Stick-Breaking process:

```python
from nsb.model import NSB
import numpy as np

# Initialize model
model = NSB(hidden_dim=64, init_identity=False)

# Train on offspring count data
data = np.array([1, 2, 3, 1, 4, 2, ...])  # Observed counts
model.fit(data, epochs=50, lr=1e-3, batch_size=128)

# Extract learned distribution
pmf = model.predict_pmf(k_max=150)
```

### Source Attribution (`nsb/task_who.py`)

Implements Algorithm 3: One-Pass Patient Zero Attribution:

```python
from nsb.task_who import attribute_source
import torch

# Learned offspring distribution
p_dist = torch.tensor([0.5, 0.3, 0.15, 0.05, ...])  # Normalized

# Compute posterior P(Z=z | C=n) for observed cluster size n
posterior = attribute_source(p_dist, n=41, prior_type="flat")
# Returns: torch.Tensor of shape (n,) with posterior probabilities
```

### Structural Metrics (`nsb/task_how.py`)

Computes entropy, R0, and extinction probability:

```python
from nsb.task_how import compute_how_metrics
import torch

# Learned offspring distribution
p_dist = torch.tensor([0.5, 0.3, 0.15, 0.05, ...])

# Compute metrics
metrics = compute_how_metrics(p_dist)
# Returns: {"entropy": float, "r0": float, "extinction_prob": float}
```

### Spectral Engine (`nsb/spectral_engine.py`)

Efficient FFT-based likelihood computation:

```python
from nsb.spectral_engine import SpectralEngine
import torch

# Compute likelihood surface P(C=n | Z=z) for all z
likelihoods = SpectralEngine.compute_likelihood_surface(p_dist, n=41)
# Returns: torch.Tensor of shape (n,) with likelihoods
```