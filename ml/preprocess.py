"""
Preprocessing — reproduces the notebook transformation EXACTLY.

Notebook pipeline:
    X = df.drop(columns=['class'])
    bins = sc.woebin(...)                # fit on train (saved as woe_bins.pkl)
    X_woe = sc.woebin_ply(X, bins)       # apply WoE
    X_woe_filtered = X_woe[feature_columns]   # 16 selected features (IV >= 0.01)

This module performs only the *apply* step (no fitting), so there is zero
risk of changing model behaviour.
"""
from __future__ import annotations

import contextlib
import io
import logging

import joblib
import pandas as pd
import scorecardpy as sc

from ml.reference import RAW_COLUMNS

log = logging.getLogger("preprocess")


class Preprocessor:
    """Applies the trained WoE bins and selects the model's feature columns."""

    def __init__(self, woe_bins_path, feature_columns_path):
        self.bins = joblib.load(woe_bins_path)
        self.feature_columns: list[str] = list(joblib.load(feature_columns_path))

    def transform(self, raw: dict | list[dict]) -> pd.DataFrame:
        """Raw application dict(s) -> WoE feature matrix in model column order."""
        rows = [raw] if isinstance(raw, dict) else list(raw)

        # Build a frame with the exact raw column order used at training time.
        df = pd.DataFrame(rows)[RAW_COLUMNS].copy()

        # scorecardpy prints "[INFO] converting into woe values ..." — silence it.
        with contextlib.redirect_stdout(io.StringIO()):
            woe = sc.woebin_ply(df, self.bins, no_cores=1)

        # Select the 16 model features, in the trained order.
        features = woe[self.feature_columns]

        # A category level absent from the training data has no WoE bin, so
        # woebin_ply yields NaN — which LogisticRegression rejects (→ 500).
        # Treat unseen levels as neutral evidence (WoE = 0), the standard
        # scorecard fallback. No-op on trained data, so parity is preserved.
        if features.isna().any().any():
            missing = features.columns[features.isna().any()].tolist()
            log.warning("Unseen WoE level(s); filling neutral 0.0 for: %s", missing)
            features = features.fillna(0.0)

        return features
