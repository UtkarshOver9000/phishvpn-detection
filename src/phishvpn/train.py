"""
Train the phishing detection model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .model import evaluate_model, save_model, train_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train phishing detection model")
    parser.add_argument("--data", type=Path, required=True, help="CSV dataset path")
    parser.add_argument("--model-out", type=Path, required=True, help="Output model path (.joblib)")
    parser.add_argument("--metrics-out", type=Path, default=Path("models/metrics.json"), help="Metrics JSON path")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.data)
    result = train_model(df, test_size=args.test_size, seed=args.seed)
    metrics = evaluate_model(result.model, result.x_test, result.y_test)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    save_model(result.model, str(args.model_out))

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to: {args.model_out}")
    print(f"Metrics saved to: {args.metrics_out}")


if __name__ == "__main__":
    main()
