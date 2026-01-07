# A Neural Stick-Breaking Process for Cascade Modelling

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)

This repository contains the official implementation for the KDD 2025 manuscript submission (ID: 101), "A Neural Stick-Breaking Process for Cascade Modelling." We introduce the Neural Stick-Breaking (NSB) process, a novel recursive generative model that learns the complete probability distribution over a countably infinite support. The NSB process uses a recurrent neural network to parameterise a stick-breaking construction, providing a flexible and data-driven alternative to traditional models for modelling biological cascades like epidemic outbreaks.

## Features

-   **Novel Model:** A full PyTorch implementation of the Neural Stick-Breaking (NSB) process.
-   **Baselines:** Implementations of strong baselines, including Poisson MLE, Negative Binomial MLE, and a Softmax Neural Network.
-   **Reproducibility:** Scripts to reproduce all experiments and generate figures.
-   **Data Processing:** Utilities to process the raw `outbreaktrees` dataset.
-   **Testing:** A comprehensive suite of unit tests to ensure code correctness.

## Project Structure

```
.
├── data/                 # Holds raw and processed data
├── experiments/          # Scripts to run experiments and generate figures
├── figures/              # Generated figures
├── nsb/                  # Core model and baseline implementations
├── results/              # Generated CSV results and LaTeX tables
├── scripts/              # Helper scripts (e.g., data processing in R)
├── tests/                # Unit tests for the codebase
├── pyproject.toml        # Project dependencies and configuration
└── README.md             # This file
```

## Installation and Setup

Follow these steps to set up the environment and run the code.

### 1. Clone the Repository

```bash
git clone https://anonymous.4open.science/r/nsb-kdd-submission_id-101
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

To run the data processing script, you need to have R installed on your system.

1.  **Install R:** Download and install R from the [Comprehensive R Archive Network (CRAN)](https://cran.r-project.org/).
2.  **Add R to PATH:** Ensure that the `bin` directory of your R installation is added to your system's PATH environment variable.
3.  **Install R Packages:** Open your R console and run the following command to install the required packages:
    ```r
    install.packages(c("tidyverse", "here", "igraph", "xtable", "Cairo"))
    ```

## Usage: Reproducing the Paper's Results

The entire experimental pipeline can be run from the command line.

### 1. Download the Data

Download the raw dataset (`taube_2020_transmission_trees.rds`) from [outbreaktrees.ecology.uga.edu](https://outbreaktrees.ecology.uga.edu/) and place it in the `data/` directory.

### 2. Process the Data

Run the R script to process the raw data and generate the clean `outbreaktrees_sars_mers_counts.csv` file.

```bash
Rscript experiments/process_outbreaktrees.r
```

### 3. Run the Experiments

Run the Python scripts located in the `experiments/` folder. Each script corresponds to a specific experiment in the paper and will save its results (CSVs, LaTeX tables, and figures) to the `results/` and `figures/` directories.

```bash
# Example: Run the main synthetic data validation
python experiments/exp_approximation.py

# Run the criticality validation
python experiments/exp_criticality.py

# ... and so on for all experiment scripts.
```

### 4. Run the Tests (Optional)

To verify the correctness of the implementation, you can run the full suite of unit tests.

```bash
pytest
```