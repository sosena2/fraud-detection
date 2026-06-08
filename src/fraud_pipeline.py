"""Preprocessing and feature engineering helpers for fraud detection."""

from __future__ import annotations

from os import PathLike
from typing import Iterable

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data_io import DataLoadError, load_fraud_sources


REQUIRED_FRAUD_COLUMNS = {
    "signup_time",
    "purchase_time",
    "ip_address",
    "user_id",
    "device_id",
    "purchase_value",
    "age",
    "source",
    "browser",
    "sex",
    "class",
}

REQUIRED_IP_COLUMNS = {
    "lower_bound_ip_address",
    "upper_bound_ip_address",
    "country",
}


def _require_columns(df: pd.DataFrame, required: Iterable[str], dataset_name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def clean_fraud_transactions(fraud_df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and normalize timestamps for the fraud dataset."""

    _require_columns(fraud_df, REQUIRED_FRAUD_COLUMNS, "Fraud_Data.csv")

    cleaned = fraud_df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned["signup_time"] = pd.to_datetime(cleaned["signup_time"], errors="coerce")
    cleaned["purchase_time"] = pd.to_datetime(cleaned["purchase_time"], errors="coerce")

    if cleaned[["signup_time", "purchase_time"]].isna().any().any():
        raise DataLoadError("Timestamp conversion failed for one or more fraud records.")

    return cleaned


def ip_to_int(ip_str: object) -> float:
    """Convert a dotted IPv4 address to an integer representation."""

    try:
        if pd.isna(ip_str):
            return np.nan

        if isinstance(ip_str, (int, np.integer)):
            return int(ip_str)

        if isinstance(ip_str, (float, np.floating)):
            return int(ip_str)

        ip_text = str(ip_str).strip()
        try:
            numeric_value = float(ip_text)
        except (TypeError, ValueError):
            numeric_value = None
        else:
            if ip_text.replace(".", "", 1).isdigit():
                return int(numeric_value)

        parts = ip_text.split(".")
        if len(parts) == 1:
            return int(float(parts[0]))
        if len(parts) != 4:
            return np.nan
        return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
    except (TypeError, ValueError):
        return np.nan


def append_ip_integer_columns(ip_df: pd.DataFrame) -> pd.DataFrame:
    """Convert the IP lookup table into integer bounds for range matching."""

    _require_columns(ip_df, REQUIRED_IP_COLUMNS, "IpAddress_to_Country.csv")

    converted = ip_df.copy()
    converted["lower_int"] = converted["lower_bound_ip_address"].apply(ip_to_int)
    converted["upper_int"] = converted["upper_bound_ip_address"].apply(ip_to_int)
    return converted


def merge_fraud_with_country_lookup(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """Attach country labels using a range-based lookup with explicit fallbacks."""

    cleaned_fraud = fraud_df.copy()
    if "ip_int" not in cleaned_fraud.columns:
        cleaned_fraud["ip_int"] = cleaned_fraud["ip_address"].apply(ip_to_int)

    cleaned_ip = append_ip_integer_columns(ip_df)
    ip_ranges = cleaned_ip.dropna(subset=["lower_int", "upper_int"]).sort_values("lower_int").reset_index(drop=True)

    valid_fraud = cleaned_fraud.dropna(subset=["ip_int"]).sort_values("ip_int").reset_index(drop=True)
    invalid_fraud = cleaned_fraud[cleaned_fraud["ip_int"].isna()].copy()
    invalid_fraud["country"] = "Unknown"

    if valid_fraud.empty:
        raise DataLoadError("No valid IP addresses were available for country matching.")
    if ip_ranges.empty:
        raise DataLoadError("No valid IP ranges were available for country matching.")

    try:
        matched = pd.merge_asof(
            valid_fraud,
            ip_ranges[["lower_int", "upper_int", "country"]],
            left_on="ip_int",
            right_on="lower_int",
            direction="backward",
        )
    except ValueError as exc:
        raise DataLoadError("Country merge failed while matching IP ranges.") from exc

    matched["country"] = np.where(matched["ip_int"] <= matched["upper_int"], matched["country"], "Unknown")
    merged = pd.concat([matched, invalid_fraud], ignore_index=True, sort=False)
    merged["country"] = merged["country"].fillna("Unknown")
    return merged


def engineer_fraud_features(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal and behavioral features for fraud detection."""

    _require_columns(merged_df, {"purchase_time", "signup_time", "user_id", "device_id"}, "merged fraud data")

    engineered = merged_df.copy()
    engineered["time_since_signup"] = (engineered["purchase_time"] - engineered["signup_time"]).dt.total_seconds() / 3600
    engineered["hour_of_day"] = engineered["purchase_time"].dt.hour
    engineered["day_of_week"] = engineered["purchase_time"].dt.dayofweek
    engineered = engineered.sort_values(["user_id", "purchase_time"]).reset_index(drop=True)
    engineered["user_txn_count"] = engineered.groupby("user_id").cumcount() + 1
    engineered["device_txn_count"] = engineered.groupby("device_id")["device_id"].transform("count")

    if engineered[["time_since_signup", "hour_of_day", "day_of_week", "user_txn_count", "device_txn_count"]].isna().any().any():
        raise DataLoadError("Feature engineering produced missing values in derived columns.")

    return engineered


def encode_and_scale_features(
    df: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    target_col: str = "class",
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categoricals and standardize numeric columns."""

    _require_columns(df, set(cat_cols) | set(num_cols) | {target_col}, "modeling frame")

    encoded = pd.get_dummies(df.copy(), columns=cat_cols, drop_first=True)
    encoded_cat_cols = [column for column in encoded.columns if any(column.startswith(f"{base}_") for base in cat_cols)]
    scaler = StandardScaler()
    encoded[num_cols] = scaler.fit_transform(encoded[num_cols])
    feature_cols = num_cols + encoded_cat_cols
    return encoded, feature_cols


def split_and_resample_features(
    encoded_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "class",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, object]:
    """Split the data and apply SMOTE on the training set only."""

    _require_columns(encoded_df, set(feature_cols) | {target_col}, "encoded fraud data")

    X = encoded_df[feature_cols]
    y = encoded_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    smote = SMOTE(random_state=random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_sm": X_train_sm,
        "y_train_sm": y_train_sm,
    }


def prepare_fraud_eda_dataset(raw_dir: str | PathLike[str]) -> dict[str, object]:
    """Load, clean, merge, and engineer the fraud dataset in one step."""

    fraud_df, ip_df = load_fraud_sources(raw_dir)
    cleaned_fraud = clean_fraud_transactions(fraud_df)
    cleaned_fraud["ip_int"] = cleaned_fraud["ip_address"].apply(ip_to_int)
    merged = merge_fraud_with_country_lookup(cleaned_fraud, ip_df)
    engineered = engineer_fraud_features(merged)
    return {"fraud_df": cleaned_fraud, "ip_df": append_ip_integer_columns(ip_df), "merged_df": engineered}
