# Phishing Detection for Suspicious VPN Connections (Python-Only)

This repository is a complete, Python-only project for detecting phishing activity associated with suspicious VPN connections across the globe. It provides a full baseline ML pipeline: data schema, synthetic data generator, feature preprocessing, model training, evaluation, and inference CLI.

**Why synthetic data?** Real VPN/phishing telemetry is sensitive and usually private. This repo includes a realistic *synthetic* generator so you can run the full pipeline end-to-end. Replace it with your own data if available.

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

High-signal example fields:
- `country`, `region`, `asn`, `vpn_provider`, `protocol`
- `login_failures_24h`, `unique_ips_24h`, `session_duration_s`
- `domain_similarity_score`, `suspicious_url_count`
- `cert_age_days`, `new_account_days`
- `mfa_used`, `account_age_days`

Target label:
- `is_phishing` (1 = phishing activity likely, 0 = benign)

## Notes and Limitations
This baseline is intentionally simple and interpretable. For production:
- Replace synthetic data with real telemetry
- Add drift detection and region-aware validation
- Incorporate temporal features and graph signals
- Integrate human-in-the-loop review and explainability

## License
MIT
