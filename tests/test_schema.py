"""Schema validation tests."""
import pytest
from pydantic import ValidationError

from app.models.schemas import LoanApplication
from ml.reference import RAW_COLUMNS, example_application


def test_example_is_valid():
    app = LoanApplication(**example_application())
    # All 20 raw model-input columns must be present after validation.
    assert set(RAW_COLUMNS).issubset(app.model_dump().keys())


def test_invalid_category_rejected():
    with pytest.raises(ValidationError):
        LoanApplication(**(example_application() | {"account_status": "ZZZ"}))


@pytest.mark.parametrize("field,value", [
    ("age", 17),               # below 18
    ("installment_rate", 5),   # outside 1..4
    ("duration", 0),           # must be positive
    ("credit_amount", 0),      # must be positive
])
def test_numeric_bounds_enforced(field, value):
    with pytest.raises(ValidationError):
        LoanApplication(**(example_application() | {field: value}))
