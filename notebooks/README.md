# Notebooks

This folder contains the analysis workflow for the fraud detection project.

- `eda-fraud-data.ipynb`: e-commerce data cleaning, IP geolocation enrichment, and feature engineering.
- `eda-creditcard.ipynb`: exploratory analysis for the credit card fraud dataset.
- `feature-engineering.ipynb`: reusable feature construction experiments.
- `modeling.ipynb`: model training, comparison, and validation.
- `shap-explainability.ipynb`: model interpretation and SHAP analysis.

Prefer calling the helper functions in `src/` from the notebooks so the workflow stays short, readable, and easy to reproduce.
