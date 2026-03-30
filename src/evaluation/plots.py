from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, output_dir: str | Path, file_name: str) -> str:
    out_dir = _ensure_dir(output_dir)
    path = out_dir / file_name
    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, output_dir: str | Path, file_name: str) -> str:
    out_dir = _ensure_dir(output_dir)
    path = out_dir / file_name
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str | Path, file_name: str) -> str:
    out_dir = _ensure_dir(output_dir)
    path = out_dir / file_name
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_calibration(y_true: np.ndarray, y_score: np.ndarray, output_dir: str | Path, file_name: str) -> str:
    out_dir = _ensure_dir(output_dir)
    path = out_dir / file_name
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], "--", label="Perfectly calibrated")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_score_distribution(y_true: np.ndarray, y_score: np.ndarray, output_dir: str | Path, file_name: str) -> str:
    out_dir = _ensure_dir(output_dir)
    path = out_dir / file_name
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_score[y_true == 0], bins=50, alpha=0.6, label="Legit", density=True)
    ax.hist(y_score[y_true == 1], bins=50, alpha=0.6, label="Fraud", density=True)
    ax.set_xlabel("Predicted fraud probability")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by Class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_threshold_tradeoff(table, output_dir: str | Path, file_name: str) -> str:
    out_dir = _ensure_dir(output_dir)
    path = out_dir / file_name
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table["threshold"], table["precision"], label="Precision")
    ax.plot(table["threshold"], table["recall"], label="Recall")
    ax.plot(table["threshold"], table["f1"], label="F1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Threshold Tradeoff")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)
