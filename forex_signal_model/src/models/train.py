from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier


@dataclass(frozen=True)
class TrainConfig:
    n_splits: int = 5
    learning_rate: float = 0.05
    max_depth: int = 6
    n_estimators: int = 500
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0


def get_default_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"y", "y_conf"}
    return [c for c in df.columns if c not in exclude and df[c].dtype.kind in "fi"]


def train_xgb_classifier(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "y",
    config: TrainConfig = TrainConfig(),
    model_out_path: str = "model.joblib",
) -> Dict:
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    y = df.loc[X.index, target_col].astype(int)

    tscv = TimeSeriesSplit(n_splits=config.n_splits)

    # Initialize model
    model = XGBClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )

    last_report = None
    aucs = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # Calculate class imbalance for this fold
        pos_count = y_tr.sum()
        neg_count = len(y_tr) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        model.set_params(scale_pos_weight=scale_pos_weight)

        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        pred = (proba >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y_va, proba)
            aucs.append(auc)
        except Exception:
            pass

        last_report = classification_report(y_va, pred, output_dict=True, zero_division=0)

    # Fit on full data
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    model.set_params(scale_pos_weight=scale_pos_weight)
    model.fit(X, y)

    artifacts = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "cv_auc_mean": float(np.mean(aucs)) if aucs else None,
        "cv_report_last": last_report,
    }

    joblib.dump({"model": model, "artifacts": artifacts}, model_out_path)
    artifacts["model_path"] = model_out_path
    return artifacts

