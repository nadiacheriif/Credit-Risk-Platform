"""Regression tests for the NaN-WoE crash (unseen category levels)."""
import numpy as np

from ml.inference import get_engine
from ml.preprocess import Preprocessor
from app.core.config import settings
from ml.reference import CATEGORICAL_FIELDS, example_application


def test_every_allowed_category_scores_without_error():
    """No offered dropdown value may produce a NaN WoE / 500."""
    engine = get_engine()
    base = example_application()
    for field, levels in CATEGORICAL_FIELDS.items():
        for code in levels:
            app = base | {field: code}
            result = engine.predict(app)  # must not raise
            assert 0.0 <= result.probability_good <= 1.0


def test_unseen_level_fills_neutral_instead_of_nan():
    """An unseen level (e.g. the absent A95) becomes neutral WoE=0, not NaN."""
    pre = Preprocessor(
        settings.ml_dir / "woe_bins.pkl", settings.ml_dir / "feature_columns.pkl"
    )
    app = example_application() | {"personal_status": "A95"}  # never in training data
    features = pre.transform(app)
    assert not np.isnan(features.to_numpy()).any()
    assert features["personal_status_woe"].iloc[0] == 0.0
