"""
Evaluate a trained model on a dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .model import evaluate_model, load_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate phishing detection model")
    parser.add_argument("--data", type=Path, required=True, help="CSV dataset path")
    parser.add_argument("--model", type=Path, required=True, help="Model path (.joblib)")
    parser.add_argument("--out", type=Path, default=Path("models/eval.json"), help="Output JSON metrics")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.data)
    model = load_model(str(args.model))

    x_test = df.drop(columns=["is_phishing"])
    y_test = df["is_phishing"]
    metrics = evaluate_model(model, x_test, y_test)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation saved to: {args.out}")


if __name__ == "__main__":
    main()
