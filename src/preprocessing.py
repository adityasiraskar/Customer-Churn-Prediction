"""
preprocessing.py

Single source of truth for cleaning / transforming the raw BankChurners data.
Used by:
  - notebooks/02_preprocessing.ipynb (training time, on the full raw CSV)
  - scripts/build_artifacts.py (fits & saves the scalers/encoders)
  - app/streamlit_app.py (inference time, on raw customer input)

Keeping this logic in one place guarantees the Streamlit app preprocesses
new customers exactly the same way the model was trained.
"""

from __future__ import annotations

import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# ----------------------------------------------------------------------
# Column groups (must match Table 2 of the reference paper / notebook 02)
# ----------------------------------------------------------------------
ID_COLS = ['CLIENTNUM']
TARGET_COL = 'Attrition_Flag'

BINARY_CAT_COLS = ['Gender']
ONEHOT_COLS = ['Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

STANDARDIZE_COLS = ['Customer_Age', 'Months_on_book']
NORMALIZE_COLS = ['Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy',
                   'Total_Trans_Amt', 'Total_Trans_Ct']

UNCHANGED_COLS = [
    'Dependent_count', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

RAW_FEATURE_COLS = (
    BINARY_CAT_COLS + ONEHOT_COLS + STANDARDIZE_COLS + NORMALIZE_COLS + UNCHANGED_COLS
)


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop CLIENTNUM and Kaggle's auto-generated Naive-Bayes helper columns."""
    nb_cols = [c for c in df.columns if 'Naive_Bayes' in c]
    cols_to_drop = [c for c in ID_COLS + nb_cols if c in df.columns]
    return df.drop(columns=cols_to_drop)


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Divorced->Single and College->Graduate, as specified in the paper."""
    df = df.copy()
    if 'Marital_Status' in df.columns:
        df['Marital_Status'] = df['Marital_Status'].replace('Divorced', 'Single')
    if 'Education_Level' in df.columns:
        df['Education_Level'] = df['Education_Level'].replace('College', 'Graduate')
    return df


def drop_unknown_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows containing the literal string 'Unknown' in any column."""
    mask = (df == 'Unknown').any(axis=1)
    return df[~mask].reset_index(drop=True)


class ChurnPreprocessor:
    """
    Fits on raw training data, can be saved/loaded with joblib, and transforms
    either a full DataFrame (training) or a single-row DataFrame (inference)
    into the exact feature matrix the model expects.
    """

    def __init__(self):
        self.gender_encoder: LabelEncoder | None = None
        self.std_scaler: StandardScaler | None = None
        self.norm_scaler: MinMaxScaler | None = None
        self.onehot_categories: dict[str, list[str]] = {}
        self.final_feature_columns: list[str] = []

    # ------------------------------------------------------------------
    # Fit (training time only)
    # ------------------------------------------------------------------
    def fit(self, raw_df: pd.DataFrame) -> "ChurnPreprocessor":
        df = drop_unused_columns(raw_df)
        df = clean_categoricals(df)
        df = drop_unknown_rows(df)

        # Gender label encoding
        self.gender_encoder = LabelEncoder()
        self.gender_encoder.fit(df['Gender'])

        # Remember the categories seen for each one-hot column (fixes column order/leakage)
        for col in ONEHOT_COLS:
            self.onehot_categories[col] = sorted(df[col].unique().tolist())

        # Fit scalers
        self.std_scaler = StandardScaler().fit(df[STANDARDIZE_COLS])
        self.norm_scaler = MinMaxScaler().fit(df[NORMALIZE_COLS])

        # Build the final transformed dataframe once to lock in column order
        transformed = self._transform_core(df, is_training=True)
        self.final_feature_columns = [c for c in transformed.columns if c != TARGET_COL]

        return self

    # ------------------------------------------------------------------
    # Internal shared transform logic
    # ------------------------------------------------------------------
    def _transform_core(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        df = df.copy()

        df['Gender'] = self.gender_encoder.transform(df['Gender'])

        if TARGET_COL in df.columns:
            le_target = LabelEncoder()
            # Existing Customer / Attrited Customer -> we want churn = 1
            df[TARGET_COL] = df[TARGET_COL].apply(
                lambda v: 1 if v == 'Attrited Customer' else 0
            )

        # One-hot encode using the categories seen at fit time (consistent columns)
        for col in ONEHOT_COLS:
            for category in self.onehot_categories[col]:
                df[f'{col}_{category}'] = (df[col] == category).astype(int)
        df = df.drop(columns=ONEHOT_COLS)

        df[STANDARDIZE_COLS] = self.std_scaler.transform(df[STANDARDIZE_COLS])
        df[NORMALIZE_COLS] = self.norm_scaler.transform(df[NORMALIZE_COLS])

        return df

    # ------------------------------------------------------------------
    # Transform (training OR inference)
    # ------------------------------------------------------------------
    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw input (single row or full dataframe) into the exact
        feature matrix the model expects (same columns, same order).
        Missing one-hot columns are filled with 0; target column is dropped if present.
        """
        df = raw_df.copy()

        # Only drop unused/unknown-row logic when target/CLIENTNUM are present
        # (at inference time there is no CLIENTNUM and no 'Unknown' filtering needed)
        if any(c in df.columns for c in ID_COLS):
            df = drop_unused_columns(df)

        df = clean_categoricals(df)

        transformed = self._transform_core(df, is_training=False)

        # Ensure all expected columns exist, in the right order
        for col in self.final_feature_columns:
            if col not in transformed.columns:
                transformed[col] = 0

        target = transformed[TARGET_COL] if TARGET_COL in transformed.columns else None
        transformed = transformed[self.final_feature_columns]

        if target is not None:
            transformed[TARGET_COL] = target

        return transformed

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ChurnPreprocessor":
        return joblib.load(path)
