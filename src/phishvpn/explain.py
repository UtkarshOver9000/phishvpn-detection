"""
LLM-assisted triage for high-risk sessions using OpenAI.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from openai import OpenAI

from .data_schema import SCHEMA
from .model import load_model


def _session_to_prompt(row: dict, risk_score: float) -> str:
    payload = {
        "session": row,
        "risk_score": round(float(risk_score), 4),
    }
    return (
        "You are a security analyst. Given the JSON session and risk_score, "
        "summarize why this looks risky and suggest a next action. "
        "Return exactly 3 bullet points, each under 18 words.\n\n"
        f"{json.dumps(payload, ensure_ascii=True)}"
    )


def _iter_candidates(df: pd.DataFrame, risk_scores: Iterable[float], threshold: float):
    for i, score in enumerate(risk_scores):
        if score >= threshold:
            yield i, float(score)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain high-risk VPN sessions with OpenAI")
    parser.add_argument("--data", type=Path, required=True, help="CSV dataset path")
    parser.add_argument("--model", type=Path, required=True, help="Model path (.joblib)")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--threshold", type=float, default=0.7, help="Risk threshold")
    parser.add_argument("--limit", type=int, default=50, help="Max sessions to explain")
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini", help="OpenAI model name")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.data)
    model = load_model(str(args.model))

    features = df[SCHEMA.categorical + SCHEMA.numeric]
    risk_scores = model.predict_proba(features)[:, 1]

    client = OpenAI()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for idx, score in _iter_candidates(df, risk_scores, args.threshold):
            if count >= args.limit:
                break
            row = df.iloc[idx].to_dict()
            prompt = _session_to_prompt(row, score)

            response = client.responses.create(
                model=args.model_name,
                input=prompt,
            )

            text = response.output_text.strip()
            record = {
                "row_index": int(idx),
                "risk_score": score,
                "explanation": text,
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1

    print(f"Wrote {count} explanations to: {args.out}")


if __name__ == "__main__":
    main()
