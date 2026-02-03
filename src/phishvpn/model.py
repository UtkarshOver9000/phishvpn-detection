"""
Model pipeline for phishing detection on VPN sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_schema import SCHEMA


@dataclass
class TrainResult:
    model: Pipeline
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def build_pipeline() -> Pipeline:
    categorical = SCHEMA.categorical
    numeric = SCHEMA.numeric

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )

    clf = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        n_jobs=None,
    )

    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", clf),
        ]
    )


def train_model(df: pd.DataFrame, test_size: float = 0.2, seed: int = 7) -> TrainResult:
    x = df[SCHEMA.categorical + SCHEMA.numeric]
    y = df[SCHEMA.target]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )

    model = build_pipeline()
    model.fit(x_train, y_train)
    return TrainResult(model, x_train, x_test, y_train, y_test)


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = model.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, proba)
    return {
        "roc_auc": float(auc),
        "classification_report": report,
    }


def save_model(model: Pipeline, path: str) -> None:
    dump(model, path)


def load_model(path: str) -> Pipeline:
    return load(path)
