# Wind Power Forecasting with Tensor Decomposition (CP & Tucker)

This project evaluates how tensor decomposition (CP and Tucker) affects the performance of short-term wind turbine power forecasting. We use real SCADA data and compare multiple forecasting models trained on raw vs. decomposed reconstructions.

## Overview

### 1. Data Preprocessing
- Load SCADA CSV
- Parse timestamps
- Normalize features
- Build tensor (Days × Window × Features)

### 2. Tensor Decomposition
- CP ranks
- Tucker ranks
- Reconstruction error analysis

### 3. Forecasting Models
- Baseline
- Ridge
- MLP
- CNN
- LSTM

### 4. Evaluation
- CSV summaries
- Plots for CP/Tucker errors
- Model performance

## Repository Structure

```
MLGproject/
│
├── preprocess.py
├── windows.py
├── decompose.py
├── models.py
├── run.py
├── analyze_results.py
├── results/
└── data/
```

## Installation

```bash
pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn tensorly==0.9.0 tensorflow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Running

```bash
python run.py
python analyze_results.py
```

## Data

Wind Turbine SCADA Dataset from Kaggle (10min readings).
