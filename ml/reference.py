"""
Single source of truth for the model's RAW input schema.

The notebook trained on the Statlog German Credit dataset (20 features).
These definitions drive:
  * the Pydantic request schema (validation + allowed categories)
  * the web form (dropdown labels, grouping, numeric bounds)
  * preprocessing column order

Categorical codes (A11, A30, ...) and human labels come straight from
`german.doc`. Numeric bounds come from the notebook's credit-policy asserts.
"""
from __future__ import annotations

# Raw column order exactly as fed to scorecardpy in the notebook
# (df.drop(columns=['class']) preserves this order).
RAW_COLUMNS = [
    "account_status", "duration", "credit_history", "purpose",
    "credit_amount", "savings", "employment", "installment_rate",
    "personal_status", "guarantors", "residence", "property",
    "age", "other_installments", "housing", "credit_cards",
    "job", "dependents", "phone", "foreign_worker",
]

# Categorical fields: code -> human label
CATEGORICAL_FIELDS: dict[str, dict[str, str]] = {
    "account_status": {
        "A11": "< 0 DM",
        "A12": "0 – 200 DM",
        "A13": ">= 200 DM / salary assignment >= 1yr",
        "A14": "No checking account",
    },
    "credit_history": {
        "A30": "No credits taken / all paid back duly",
        "A31": "All credits at this bank paid back duly",
        "A32": "Existing credits paid back duly till now",
        "A33": "Delay in paying off in the past",
        "A34": "Critical account / credits elsewhere",
    },
    "purpose": {
        "A40": "Car (new)",
        "A41": "Car (used)",
        "A42": "Furniture / equipment",
        "A43": "Radio / television",
        "A44": "Domestic appliances",
        "A45": "Repairs",
        "A46": "Education",
        "A48": "Retraining",
        "A49": "Business",
        "A410": "Others",
    },
    "savings": {
        "A61": "< 100 DM",
        "A62": "100 – 500 DM",
        "A63": "500 – 1000 DM",
        "A64": ">= 1000 DM",
        "A65": "Unknown / no savings account",
    },
    "employment": {
        "A71": "Unemployed",
        "A72": "< 1 year",
        "A73": "1 – 4 years",
        "A74": "4 – 7 years",
        "A75": ">= 7 years",
    },
    # NOTE: code A95 ("female: single") is documented in the dataset spec but never
    # occurs in the 1000-row German Credit data, so the trained WoE has no bin for
    # it. Offering it would yield a NaN WoE → model error, so it is omitted.
    "personal_status": {
        "A91": "Male: divorced / separated",
        "A92": "Female: divorced / separated / married",
        "A93": "Male: single",
        "A94": "Male: married / widowed",
    },
    "guarantors": {
        "A101": "None",
        "A102": "Co-applicant",
        "A103": "Guarantor",
    },
    "property": {
        "A121": "Real estate",
        "A122": "Building society savings / life insurance",
        "A123": "Car or other",
        "A124": "Unknown / no property",
    },
    "other_installments": {
        "A141": "Bank",
        "A142": "Stores",
        "A143": "None",
    },
    "housing": {
        "A151": "Rent",
        "A152": "Own",
        "A153": "For free",
    },
    "job": {
        "A171": "Unemployed / unskilled non-resident",
        "A172": "Unskilled resident",
        "A173": "Skilled employee / official",
        "A174": "Management / self-employed / highly qualified",
    },
    "phone": {
        "A191": "None",
        "A192": "Yes, registered under customer name",
    },
    "foreign_worker": {
        "A201": "Yes",
        "A202": "No",
    },
}

# Numeric fields: label, bounds (from notebook policy asserts), default
NUMERIC_FIELDS: dict[str, dict] = {
    "duration":         {"label": "Loan duration (months)",          "min": 1,   "max": 120,    "default": 24},
    "credit_amount":    {"label": "Credit amount (DM)",              "min": 1,   "max": 100000, "default": 5000},
    "installment_rate": {"label": "Installment rate (% of income)",  "min": 1,   "max": 4,      "default": 2},
    "residence":        {"label": "Present residence since (years)", "min": 1,   "max": 4,      "default": 2},
    "age":              {"label": "Age (years)",                     "min": 18,  "max": 120,    "default": 35},
    "credit_cards":     {"label": "Existing credits at this bank",   "min": 0,   "max": 10,     "default": 1},
    "dependents":       {"label": "People liable for maintenance",   "min": 0,   "max": 10,     "default": 1},
}

# UI grouping (banking-style sections)
FORM_SECTIONS = [
    {
        "title": "Personal",
        "icon": "user",
        "fields": ["age", "personal_status", "job", "employment",
                   "residence", "housing", "dependents", "foreign_worker", "phone"],
    },
    {
        "title": "Financial",
        "icon": "wallet",
        "fields": ["account_status", "savings", "property",
                   "credit_history", "credit_cards", "other_installments", "guarantors"],
    },
    {
        "title": "Loan Details",
        "icon": "file",
        "fields": ["purpose", "credit_amount", "duration", "installment_rate"],
    },
]

# Human-friendly labels for categorical fields (for section headers / tables)
CATEGORICAL_LABELS = {
    "account_status": "Checking account status",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "savings": "Savings account / bonds",
    "employment": "Present employment since",
    "personal_status": "Personal status & sex",
    "guarantors": "Other debtors / guarantors",
    "property": "Property",
    "other_installments": "Other installment plans",
    "housing": "Housing",
    "job": "Job",
    "phone": "Telephone",
    "foreign_worker": "Foreign worker",
}


def example_application() -> dict:
    """A realistic, valid sample application (a known 'good' profile)."""
    return {
        "account_status": "A14", "duration": 12, "credit_history": "A32",
        "purpose": "A43", "credit_amount": 1500, "savings": "A65",
        "employment": "A75", "installment_rate": 2, "personal_status": "A93",
        "guarantors": "A101", "residence": 2, "property": "A121", "age": 40,
        "other_installments": "A143", "housing": "A152", "credit_cards": 1,
        "job": "A173", "dependents": 1, "phone": "A192", "foreign_worker": "A202",
    }
