from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


import shap

@dataclass(frozen=True)
class SignalConfig:
    buy_threshold: float = 0.55
    sell_threshold: float = 0.45
    min_confidence: float = 0.0


def predict_signal(model_bundle: dict, row: pd.Series, feature_cols: list, cfg: SignalConfig) -> Dict:
    model = model_bundle["model"]
    X = row[feature_cols].to_frame().T
    proba_buy = float(model.predict_proba(X)[:, 1][0])
    proba_sell = 1.0 - proba_buy

    side: Optional[str] = None
    conf: float = 0.0

    if proba_buy >= cfg.buy_threshold:
        side = "BUY"
        conf = proba_buy
    elif proba_sell >= cfg.buy_threshold:
        side = "SELL"
        conf = proba_sell
    else:
        side = "HOLD"
        conf = max(proba_buy, proba_sell)

    if conf < cfg.min_confidence:
        side = "HOLD"

    reason = "No clear signal."
    if side in ["BUY", "SELL"]:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                vals = shap_values[1][0] if side == "BUY" else shap_values[0][0]
            else:
                vals = shap_values[0]
                if side == "SELL":
                    vals = -vals

            top_indices = np.argsort(vals)[-3:][::-1]
            top_features = [feature_cols[i] for i in top_indices if vals[i] > 0]
            
            if top_features:
                reason = f"AI detected strong signals from: {', '.join(top_features)}."
        except Exception as e:
            reason = f"Could not generate reason: {e}"

    return {
        "side": side,
        "proba_buy": proba_buy,
        "proba_sell": proba_sell,
        "confidence": conf,
        "reason": reason,
    }

