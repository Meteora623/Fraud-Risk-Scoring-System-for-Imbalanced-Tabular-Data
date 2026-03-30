from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.thresholding import regime_descriptions
from src.models.infer import RiskScorer
from src.utils.config import load_config
from src.utils.io import read_json

st.set_page_config(page_title="Fraud Risk Scoring", layout="wide")

cfg = load_config()
model_bundle = Path(cfg.paths.model_dir) / "serving_bundle.joblib"
threshold_file = Path(cfg.paths.threshold_summary_path)
metrics_file = Path(cfg.paths.comparison_metrics_path)
scored_path = Path(cfg.paths.artifact_dir) / "scored_holdout.csv"
threshold_table_path = Path(cfg.paths.artifact_dir) / "threshold_table.csv"
error_path = Path(cfg.paths.artifact_dir) / "error_analysis.json"
final_summary_path = Path(cfg.paths.artifact_dir) / "final_model_summary.json"

st.title("Fraud Risk Scoring System")
st.caption("PaySim-based transaction risk scoring with calibration and threshold-aware decisions")

missing = [p for p in [model_bundle, threshold_file, metrics_file, scored_path] if not p.exists()]
if missing:
    st.error("Required artifacts missing. Run: download_data.py -> audit_data.py -> train_models.py -> run_evaluation.py")
    st.stop()

scorer = RiskScorer(model_bundle, threshold_file)
metrics_summary = read_json(metrics_file)
scored_df = pd.read_csv(scored_path)
threshold_summary = read_json(threshold_file)
final_summary = read_json(final_summary_path) if final_summary_path.exists() else {}


@st.cache_data
def _load_threshold_table(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def _load_error(path: Path):
    if path.exists():
        return read_json(path)
    return {}


def _default_numeric(col: str, fallback: float = 0.0) -> float:
    if col in scored_df.columns and pd.api.types.is_numeric_dtype(scored_df[col]):
        return float(scored_df[col].median())
    return fallback


tab_score, tab_compare, tab_threshold, tab_error = st.tabs(
    ["Risk Scoring", "Model Comparison", "Threshold Analysis", "Error Analysis"]
)

with tab_score:
    st.subheader("Score a Transaction")

    type_options = ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT", "CASH_IN"]
    if "type" in scored_df.columns:
        observed = scored_df["type"].dropna().astype(str).str.upper().value_counts().index.tolist()
        if observed:
            type_options = observed

    c1, c2 = st.columns(2)
    with c1:
        step = st.number_input("step", min_value=0, value=int(_default_numeric("step", 1)), step=1)
        tx_type = st.selectbox("type", options=type_options, index=0)
        amount = st.number_input("amount", min_value=0.0, value=_default_numeric("amount", 1000.0))
        oldbalance_org = st.number_input("oldbalanceOrg", min_value=0.0, value=_default_numeric("oldbalanceOrg", 5000.0))
    with c2:
        newbalance_orig = st.number_input("newbalanceOrig", min_value=0.0, value=_default_numeric("newbalanceOrig", 4000.0))
        oldbalance_dest = st.number_input("oldbalanceDest", min_value=0.0, value=_default_numeric("oldbalanceDest", 2000.0))
        newbalance_dest = st.number_input("newbalanceDest", min_value=0.0, value=_default_numeric("newbalanceDest", 3000.0))
        is_flagged = st.selectbox("isFlaggedFraud (optional)", options=[None, 0, 1], index=0)

    threshold_mode = st.selectbox(
        "Decision regime",
        options=["balanced_f1", "high_precision", "high_recall", "cost_sensitive"],
        index=0,
        help="Switch operating mode based on business tradeoff.",
    )

    transaction = {
        "step": int(step),
        "type": tx_type,
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalance_org),
        "newbalanceOrig": float(newbalance_orig),
        "oldbalanceDest": float(oldbalance_dest),
        "newbalanceDest": float(newbalance_dest),
    }
    if is_flagged is not None:
        transaction["isFlaggedFraud"] = int(is_flagged)

    if st.button("Score Transaction"):
        result = scorer.score_record(transaction, threshold_key=threshold_mode)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Score", f"{result['risk_score']:.4f}")
        c2.metric("Predicted Class", str(result["predicted_class"]))
        c3.metric("Risk Band", result["risk_band"].upper())
        c4.metric("Threshold", f"{result['threshold_used']:.2f}")

        st.write(f"Review posture: {result['review_posture']}")
        st.write(f"Recommended action: {result['recommended_action']}")
        st.write(f"Decision note: {result['decision_note']}")

        if result.get("regime_metrics"):
            st.write("Selected regime metrics")
            st.dataframe(pd.DataFrame([result["regime_metrics"]]), use_container_width=True)

        if result.get("cost_context"):
            st.write("Cost context (configured)")
            st.dataframe(pd.DataFrame([result["cost_context"]]), use_container_width=True)

        st.write("Top contributing factors")
        st.write(result["top_factors"])

        comp_rows = []
        for mode in ["balanced_f1", "high_precision", "high_recall", "cost_sensitive"]:
            mode_result = scorer.score_record(transaction, threshold_key=mode)
            comp_rows.append(
                {
                    "regime": mode,
                    "threshold": mode_result["threshold_used"],
                    "predicted_class": mode_result["predicted_class"],
                    "risk_band": mode_result["risk_band"],
                    "review_posture": mode_result["review_posture"],
                    "recommended_action": mode_result["recommended_action"],
                    "expected_cost": mode_result["regime_metrics"].get("expected_cost"),
                }
            )
        st.write("Regime comparison for this transaction")
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

with tab_compare:
    st.subheader("Model Comparison")
    rows = []
    for model_name, payload in metrics_summary.items():
        test = payload.get("test", {})
        rows.append(
            {
                "model": model_name,
                "selected_for_serving": payload.get("selected_for_serving", False),
                "model_backend": payload.get("model_backend"),
                "estimator_class": payload.get("selected_estimator_class") or payload.get("estimator_class"),
                "calibration_state": payload.get("calibration_state"),
                "pr_auc": test.get("pr_auc"),
                "roc_auc": test.get("roc_auc"),
                "precision": test.get("precision"),
                "recall": test.get("recall"),
                "f1": test.get("f1"),
                "brier": test.get("brier"),
            }
        )
    cmp_df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    st.dataframe(cmp_df, use_container_width=True)

    boosted_note = final_summary.get("boosted_model_observation", {}).get("note") if final_summary else None
    if boosted_note:
        st.info(f"Boosted model note: {boosted_note}")

with tab_threshold:
    st.subheader("Threshold Regimes")
    regime_rows = {
        k: v for k, v in threshold_summary.items() if isinstance(v, dict) and "threshold" in v
    }
    meta = threshold_summary.get("_meta", {})

    ts = pd.DataFrame(regime_rows).T
    st.dataframe(ts, use_container_width=True)

    if meta:
        st.write("Threshold policy metadata")
        st.dataframe(pd.DataFrame([meta]), use_container_width=True)

    desc = regime_descriptions()
    st.write("Regime intent")
    st.dataframe(pd.DataFrame([{"regime": k, "description": v} for k, v in desc.items()]))

    tradeoff_df = _load_threshold_table(threshold_table_path)
    if len(tradeoff_df):
        st.line_chart(tradeoff_df.set_index("threshold")[["precision", "recall", "f1"]])
        if "cost_per_txn" in tradeoff_df.columns:
            st.line_chart(tradeoff_df.set_index("threshold")[["cost_per_txn"]])

with tab_error:
    st.subheader("Error Analysis")
    error_report = _load_error(error_path)
    if error_report:
        st.write("Counts", error_report.get("counts", {}))
        fp_num = error_report.get("false_positive_patterns", {}).get("numeric_shift", {})
        fn_num = error_report.get("false_negative_patterns", {}).get("numeric_shift", {})
        if fp_num:
            st.write("False Positive Numeric Shift")
            st.dataframe(pd.DataFrame(fp_num).T.head(10), use_container_width=True)
        if fn_num:
            st.write("False Negative Numeric Shift")
            st.dataframe(pd.DataFrame(fn_num).T.head(10), use_container_width=True)
    else:
        st.info("Run scripts/run_evaluation.py to generate error analysis artifacts.")
