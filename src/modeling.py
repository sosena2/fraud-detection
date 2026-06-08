"""Baseline and ensemble model helpers for fraud detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score


@dataclass
class ModelEvaluation:
    """Container for common fraud classification metrics."""

    auc_pr: float
    f1: float
    confusion: np.ndarray


def evaluate_classifier(model, X_test, y_test) -> ModelEvaluation:
    """Evaluate a fitted classifier using imbalanced-classification metrics."""

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return ModelEvaluation(
        auc_pr=average_precision_score(y_test, y_prob),
        f1=f1_score(y_test, y_pred),
        confusion=confusion_matrix(y_test, y_pred),
    )


def train_logistic_regression_baseline(X_train, y_train, random_state: int = 42):
    """Train an interpretable baseline logistic regression model."""

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest_model(
    X_train,
    y_train,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int | None = None,
):
    """Train a random forest model suitable for a stronger benchmark."""

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
