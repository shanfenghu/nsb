# Neural Stick-Breaking for Cascade Forensics

This repository contains the official Python implementation and experimental code for the paper:

**Neural Stick-Breaking for Cascade Forensics**  
*In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD ’26), August 09–13, 2026, Jeju Island, Republic of Korea.*

## Abstract

Characterising transmission laws in epidemic cascades is essential for outbreak forensics, yet neural models with finite support fail to capture heavy-tailed phenomena like superspreading. We propose the Neural Stick-Breaking (NSB) process, a neuro-symbolic-spectral duality that learns infinite-support offspring distributions via a recurrent mapping with theoretical guarantees. Beyond density estimation, the NSB enables exact probabilistic inference of cascade histories in $O(nK \log nK)$ via the Fast Fourier Transform (FFT), achieving sub-second efficiency for massive clusters ($n=10^5$). This allows us to solve three forensic tasks at scale: Source Attribution (identifying patient-zeros), Structural Fingerprinting (characterising information dissipation), and Risk Assessment (analytic extinction). Empirically, the NSB and its gated variants significantly outperform neural baselines and classical estimators on the outbreaktrees (SARS/MERS-Cov) dataset for cascade modelling. For an out-of-sample Saudi Arabia 2019 MERS outbreak, our forensic analysis identifies a critical Signal-to-Noise Crossover at cluster size $n \approx 4$ and an Information Horizon at $n \approx 50$, uncovering the fundamental limits of founder identifiability in real-world outbreaks. By leveraging its neuro-symbolic-spectral duality, the NSB provides a rigorous, scalable framework for scientific discovery in cascade forensics. The source code is available at https://github.com/shanfenghu/nsb

## Features

-   **Neural Stick-Breaking Model:** A full PyTorch implementation of the NSB process with configurable hidden dimensions and identity-centered initialization for criticality control.
-   **Spectral Inversion Engine:** Efficient O(nK log(nK)) implementation of source attribution using FFT and the Neural Otter-Dwass identity.
-   **Forensic Task Modules:** 
    - **Task "Who"**: One-pass patient zero attribution with Bayesian priors (Flat, Poisson, Negative Binomial)
    - **Task "How"**: Structural fingerprinting with entropy, R0, and extinction probability computation
    - **Task "Next"**: Branching volatility analysis and generational PGF visualization
-   **Baselines:** Implementations of strong baselines, including Poisson MLE, Negative Binomial MLE, and Softmax Neural Networks.
-   **Reproducibility:** Complete scripts to reproduce all experiment figures and analyses.
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
│   ├── exp_training_dynamics.py   # Training dynamics and convergence analysis
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
git clone https://github.com/shanfenghu/nsb
cd nsb
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

The main dashboards are generated by two key scripts:

```bash
# Duality Validation Dashboard (Task "Who")
# - Forensic "Cold Case" Reconstruction (source attribution)
# - Spectral Signature (PGF on unit circle)
# - Computational Shield (scaling analysis)
python experiments/exp_task_who.py

# Forensic Forecasting Dashboard (Tasks "How" & "Next")
# - Extinction Phase Transition
# - Forensic Entropy and Information Horizon
# - Branching Volatility (Superspreading Risk)
# - Spectral Energy Dissipation (Generational PGF)
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

# Training dynamics and convergence analysis
python experiments/exp_training_dynamics.py
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

Implements One-Pass Patient Zero Attribution:

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
