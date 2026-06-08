# Improved Fraud Detection for E-commerce and Banking

This project implements a reproducible fraud detection workflow for two imbalanced classification problems:

- E-commerce fraud using `Fraud_Data.csv` and `IpAddress_to_Country.csv`.
- Bank card fraud using `creditcard.csv`.

The repository is organized to support three stages of work:

1. Data analysis and preprocessing.
2. Model training and comparison.
3. Explainability with SHAP.

## Project Goals

- Clean and standardize the raw datasets.
- Enrich e-commerce transactions with geolocation information from the IP lookup table.
- Engineer behavioral features such as `time_since_signup`, `hour_of_day`, `day_of_week`, and transaction counts.
- Handle class imbalance with training-only resampling.
- Train baseline and ensemble fraud models using metrics that are appropriate for imbalanced data.
- Explain model decisions with SHAP and translate them into business actions.

## Data Sources

- `Fraud_Data.csv`: e-commerce transactions with user, device, time, and browser context.
- `IpAddress_to_Country.csv`: IP range to country lookup table used for geolocation enrichment.
- `creditcard.csv`: anonymized bank card transactions with PCA-derived features.

## Repository Structure

- `data/` contains raw and processed datasets. The folder is kept in the repo with placeholder files, while the large CSVs remain ignored.
- `models/` stores saved model artifacts and is also kept with a placeholder file.
- `notebooks/` contains the EDA, feature engineering, modeling, and SHAP notebooks.
- `src/` contains reusable preprocessing, feature engineering, and modeling helpers.
- `tests/` contains lightweight checks.

## Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If you use Jupyter directly, also register the environment kernel:

```bash
python -m ipykernel install --user --name fraud-detection --display-name "fraud-detection"
```

## How To Run

### 1. EDA and preprocessing

Run `notebooks/eda-fraud-data.ipynb` to:

- Load and clean the fraud dataset.
- Convert timestamps and IP addresses safely.
- Merge IP ranges to countries.
- Engineer time-based and behavioral features.
- Save processed outputs in `data/processed/`.

### 2. Modeling

Run `notebooks/modeling.ipynb` to:

- Load the processed datasets.
- Split the data with stratification.
- Train the logistic regression baseline.
- Train and tune the ensemble model.
- Compare models with AUC-PR, F1, and confusion matrices.

### 3. Explainability

Run `notebooks/shap-explainability.ipynb` to:

- Review feature importance from the final model.
- Generate SHAP summary plots.
- Inspect individual force plots for true positives, false positives, and false negatives.

## Notes On The Fraud EDA Pipeline

The core Task 1 logic is implemented in `src/fraud_pipeline.py` and `src/data_io.py` so the notebook stays thin and readable. The notebook should call these helpers instead of repeating the transformations inline.

## Deliverables

- Cleaned and processed datasets.
- EDA figures and notes.
- Feature engineering and resampling justification.
- Trained model artifacts in `models/`.
- SHAP visualizations and recommendations.
