"""Input/output helpers for the fraud detection project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataLoadError(RuntimeError):
    """Raised when a dataset cannot be loaded safely."""


def load_csv_data(file_path: str | Path, description: str) -> pd.DataFrame:
    """Load a CSV file with explicit error handling."""

    path = Path(file_path)
    if not path.exists():
        raise DataLoadError(f"{description} not found at: {path}")

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"{description} is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"{description} could not be parsed: {path}") from exc
    except Exception as exc:  # pragma: no cover - defensive wrapper
        raise DataLoadError(f"Unexpected error loading {description} from {path}") from exc


def load_fraud_sources(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Fraud_Data and IP lookup tables from the raw data directory."""

    raw_path = Path(raw_dir)
    fraud_df = load_csv_data(raw_path / "Fraud_Data.csv", "Fraud_Data.csv")
    ip_df = load_csv_data(raw_path / "IpAddress_to_Country.csv", "IpAddress_to_Country.csv")
    return fraud_df, ip_df
