# Phishing Detection for Suspicious VPN Connections (Python-Only)

![CI](https://github.com/UtkarshOver9000/phishvpn-detection/actions/workflows/ci.yml/badge.svg)

This repository is a complete, Python-only project for detecting phishing activity associated with suspicious VPN connections across the globe. It provides a full baseline ML pipeline: data schema, synthetic data generator, feature preprocessing, model training, evaluation, and inference CLI.

**Why synthetic data?** Real VPN/phishing telemetry is sensitive and usually private. This repo includes a realistic *synthetic* generator so you can run the full pipeline end-to-end. Replace it with your own data if available.

## Web API & interactive dashboard

Beyond the CLI pipeline below, `src/phishvpn/api/app.py` wraps the model in a small
FastAPI service with a dashboard — it trains itself in-memory from synthetic data on
startup (no model file to manage) and exposes a `/v1/score` endpoint plus a one-page
sandbox UI with three preset scenarios (benign / borderline / suspicious) you can fire
with one click.

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m uvicorn phishvpn.api.app:app --reload --port 8000
```

- Dashboard: http://localhost:8000
- Interactive API docs (Swagger): http://localhost:8000/docs

## Problem Statement
Given VPN connection logs and related security telemetry, predict whether a session is likely to be associated with phishing activity. The model should generalize across geographies and providers, handle categorical + numeric signals, and provide risk scores to support security triage.

## Repository Structure
- `src/phishvpn/` core library (schema, features, model, train/eval/infer)
- `data/` placeholder for datasets
- `tests/` minimal sanity tests
- `requirements.txt` Python dependencies

## Quickstart
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# For running modules from source without installing:
set PYTHONPATH=src

# 1) Generate synthetic data
python -m phishvpn.synthetic_data --out data/sessions.csv --rows 50000

# 2) Train
python -m phishvpn.train --data data/sessions.csv --model-out models/phishvpn.joblib

# 3) Evaluate
python -m phishvpn.evaluate --data data/sessions.csv --model models/phishvpn.joblib

# 4) Inference
python -m phishvpn.infer --data data/sessions.csv --model models/phishvpn.joblib --out data/preds.csv
```

## Run the tests

```bash
pip install -r requirements.txt
pytest --cov=src --cov-report=term-missing
```

20 tests, 82% line coverage: schema/generator sanity, model train/evaluate/save-load,
the train/evaluate/infer CLIs end-to-end, and the pure helper functions in
`explain.py` (its OpenAI-calling `main()` is intentionally left untested — it needs
a live API key/network access, which a unit suite shouldn't depend on). Lint with
`ruff check .`. CI (`.github/workflows/ci.yml`) runs lint + tests + a full
generate/train smoke test on Python 3.10–3.12 for every push and PR.

## Model performance

Trained with `python -m phishvpn.train --data data/sessions.csv --model-out models/phishvpn.joblib`
on 50,000 synthetic sessions (80/20 stratified split, logistic regression, seed 7):

| Metric | Score |
|---|---|
| ROC-AUC | 0.919 |
| Precision (phishing class) | 0.483 |
| Recall (phishing class) | 0.846 |
| F1 (phishing class) | 0.615 |
| Accuracy | 83.6% |

These numbers are on synthetic data with a known-imbalanced positive rate (~15%), not
real telemetry — see Limitations. Getting here required fixing two real bugs in the
original baseline, not just tuning:

- **`asn` (network ID) was one-hot encoded as a categorical feature.** It's a
  near-unique identifier — ~28k distinct values across 50k rows — so the linear model
  was memorizing training rows via their ASN instead of learning generalizable
  signal: in-sample AUC was 0.96, held-out AUC was 0.54. It's now excluded from model
  features (`data_schema.py`'s `IDENTIFIER_COLUMNS`) but still generated/recorded for
  telemetry.
- **The synthetic label was driven by rare binary thresholds** (e.g.
  `suspicious_url_count > 1`) that individually fired on a small slice of rows and
  rarely co-occurred, combined with a noise term large enough to erase most of what
  little signal existed — the Bayes-optimal ceiling given the old formula was ~0.6
  ROC-AUC, regardless of model quality. `synthetic_data.py` now uses continuous
  z-scored contributions from every risk-relevant feature, giving every row a graded,
  separable score.

```bash
python -m phishvpn.evaluate --data data/sessions.csv --model models/phishvpn.joblib
```

## OpenAI-Assisted Triage (Best Real-World Add-On)
Use OpenAI to generate concise analyst-facing explanations for high-risk sessions.

```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

```powershell
python -m phishvpn.explain --data data/sessions.csv --model models/phishvpn.joblib --out data/explanations.jsonl --threshold 0.7 --limit 25
```

## Data Schema (Core Columns)
See `src/phishvpn/data_schema.py` for full schema.

- **Categorical model features**: `country`, `region`, `vpn_provider`, `protocol`, `device_type`, `auth_method`, `mfa_used`
- **Numeric model features**: `login_failures_24h`, `unique_ips_24h`, `session_duration_s`, `domain_similarity_score`, `suspicious_url_count`, `cert_age_days`, `new_account_days`, `account_age_days`, `hour_of_day`, `day_of_week`
- **Identifier (recorded, not fed to the model)**: `asn` — high-cardinality, excluded from `CATEGORICAL_COLUMNS` (see Model performance)

Target label:
- `is_phishing` (1 = phishing activity likely, 0 = benign)

## Notes and Limitations
This baseline is intentionally simple and interpretable. For production:
- Replace synthetic data with real telemetry — the model has not been validated against real-world sessions, only against its own synthetic generator.
- Add drift detection and region-aware validation
- Incorporate temporal features and graph signals
- Integrate human-in-the-loop review and explainability
- Precision on the phishing class is 0.48 at the default 0.5 threshold — in a real triage
  system this threshold should be tuned against the actual cost of false positives vs.
  missed detections, not left at the default.

## License
MIT
