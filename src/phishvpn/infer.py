"""
Run inference on a CSV file and output risk scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .model import load_model
from .data_schema import SCHEMA


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for phishing detection model")
    parser.add_argument("--data", type=Path, required=True, help="CSV dataset path")
    parser.add_argument("--model", type=Path, required=True, help="Model path (.joblib)")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.data)

    model = load_model(str(args.model))
    features = df[SCHEMA.categorical + SCHEMA.numeric]
    proba = model.predict_proba(features)[:, 1]

    out_df = df.copy()
    out_df["phishing_risk_score"] = proba
    out_df["phishing_pred"] = (proba >= 0.5).astype(int)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Predictions saved to: {args.out}")


if __name__ == "__main__":
    main()
