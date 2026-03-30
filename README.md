# Fraud Risk Scoring System for Imbalanced Tabular Data

Interview-ready ML risk scoring project for fraud prioritization on tabular transactions.

This repository focuses on **probability-based decision support**: calibrated fraud scores, threshold regimes, cost-aware operating points, and analyst-facing outputs via API and Streamlit.

## Why this project
Fraud operations rarely optimize raw accuracy. Teams need:
- reliable risk probabilities,
- configurable operating thresholds,
- clear precision/recall tradeoffs,
- transparent business policy communication.

This project is built around those requirements.

## Why PaySim (and what is real vs synthetic)
The default dataset is **PaySim** from Kaggle (`ealaxi/paysim1`).

Why it was chosen:
- feature space is business-interpretable (`type`, `amount`, balance fields),
- better for decision-policy discussions than anonymous PCA features.

Important caveat:
- PaySim is **synthetic/simulated** transaction data, not production bank logs.
- Claims in this repo are framed accordingly.

## Data decisions (explicit and auditable)
Configured in `configs/config.yaml`, surfaced in `data/artifacts/audit_report.json`.

- Target: `isFraud`
- Identifier drop: `nameOrig`, `nameDest`
- Leakage-risk drop: `isFlaggedFraud`
- Duplicate policy: `drop_exact`
- Optional row cap for CPU-friendly reproducibility: `dataset.sample_max_rows`

## System scope
- Data ingestion + normalization: `src/data/loaders.py`
- Dataset audit/provenance/leakage notes: `src/data/audit.py`
- Preprocessing pipeline (reused for train/infer): `src/data/feature_pipeline.py`
- Train/val/test split: `src/data/split.py`
- Model training/comparison/calibration: `src/models/train.py`
- Threshold/cost policy logic: `src/evaluation/thresholding.py`
- Error analysis and evaluation plots: `src/evaluation/*`
- API scoring service: `src/api/*`
- Streamlit decision dashboard: `src/demo/streamlit_app.py`

## Modeling and selection policy
Compared models:
- Logistic Regression
- Random Forest
- Boosted entry (`xgboost` config slot; may fallback to `HistGradientBoosting` if XGBoost unavailable)

Selection:
- Primary serving model is chosen by **validation PR-AUC** when `primary_model: auto`.

Calibration:
- Applied per model with `isotonic`/`sigmoid`.
- Calibrated version is used only if validation **Brier score improves**.

## Current model-result transparency
Final run summary is captured in:
- `data/artifacts/final_model_summary.json`
- `data/artifacts/final_project_claims.md`

This includes:
- selected serving model,
- backend actually used by the boosted slot,
- calibration state,
- ranking by test PR-AUC,
- explicit note when boosted model is not selected.

If boosted underperforms, this repo does not hide it. Common reasons on this setup include:
- backend fallback (if XGBoost is unavailable),
- conservative default parameters,
- no large hyperparameter sweep in this portfolio version.

## Threshold regimes and business meaning
Regimes are stored in `data/artifacts/threshold_summary.json` and include metrics/costs.

- `balanced_f1`: general triage balance
- `high_precision`: fewer false alerts; stricter review load
- `high_recall`: catch more fraud; higher alert volume
- `cost_sensitive`: minimize weighted FP/FN decision cost

Cost-sensitive mode uses configured costs:
- `false_positive_cost`
- `false_negative_cost`

Threshold metadata and policy context are included under `_meta` in threshold summary.

## API (decision-support oriented)
Endpoints:
- `GET /health`
- `POST /score_transaction`
- `GET /model_summary`
- `GET /threshold_summary`
- `GET /metrics_summary`

`/score_transaction` returns:
- `risk_score`
- `risk_band`
- `decision_regime` + regime note
- `review_posture`
- `recommended_action`
- regime metrics (precision/recall/F1/cost where available)
- top contributing factors

## Streamlit demo
Tabs:
- Risk Scoring
- Model Comparison
- Threshold Analysis
- Error Analysis

The scoring tab compares the same transaction under all threshold regimes and surfaces suggested review posture/action.

## Artifacts to review first
- `data/artifacts/final_model_summary.json`
- `data/artifacts/model_comparison_metrics.json`
- `data/artifacts/threshold_summary.json`
- `data/artifacts/threshold_table.csv`
- `data/artifacts/audit_report.json`
- `data/artifacts/final_project_claims.md`

## Run commands
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_data.py
python scripts/audit_data.py
python scripts/train_models.py
python scripts/run_evaluation.py

python scripts/launch_api.py
python scripts/launch_demo.py

pytest -q
```

## Limitations (explicit)
- Synthetic dataset; external validity to real fraud ops is limited.
- Offline batch pipeline only; no real-time feature store or online learning loop.
- Limited hyperparameter tuning by design (clarity/reproducibility over benchmark chasing).
- Explainability is lightweight and practical, not a full SHAP-heavy workflow.
