# BreastCancerSense

A machine learning tool for early breast cancer detection using clinical data.

## Purpose
BreastCancerSense predicts breast cancer using clinical data to aid early detection.

## Overview
- Predicts breast cancer using the Coimbra Dataset (116 samples, 9 features).
- Uses XGBoost with GridSearchCV for training.
- Features a Streamlit app for user predictions and visualizations.

## Requirements
- Python 3
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `streamlit`

## How to Run
1. Navigate to the `code` directory:
2. Train the model:
3. Run the app:

## Directory Structure
- `data/dataR2.csv`: Dataset.
- `code/train_model.py`: Trains the model.
- `code/predict_ui.py`: Streamlit app.
- `output/`: Saves model, scaler, plots.

## Author
Diksha Gupta, 2025 Major Project