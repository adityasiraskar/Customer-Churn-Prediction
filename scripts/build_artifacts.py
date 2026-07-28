"""
scripts/build_artifacts.py

Run this ONCE (after you have data/raw/BankChurners.csv) to fit and save the
preprocessing artifacts the Streamlit app needs at inference time.

Usage:
    python scripts/build_artifacts.py

Outputs:
    models/saved_models/preprocessor.pkl
"""

import os
import sys

import joblib
import pandas as pd

# Allow running this script directly from the repo root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import ChurnPreprocessor  # noqa: E402

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'BankChurners.csv')
ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models', 'preprocessor.pkl')
FEATURE_COLUMNS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models', 'feature_columns.pkl')


def load_feature_columns() -> list[str] | None:
    if not os.path.exists(FEATURE_COLUMNS_PATH):
        print(
            "No feature_columns.pkl found. The preprocessor will keep all "
            "transformed feature columns."
        )
        return None

    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    print(f"Loaded {len(feature_columns)} model feature columns from: {FEATURE_COLUMNS_PATH}")
    return list(feature_columns)


def main():
    print(f"Loading raw data from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    print("Raw shape:", df.shape)

    preprocessor = ChurnPreprocessor(feature_columns=load_feature_columns()).fit(df)

    preprocessor.save(ARTIFACT_PATH)
    print(f"Saved preprocessor to: {ARTIFACT_PATH}")
    print(f"Final feature columns ({len(preprocessor.final_feature_columns)}):")
    for c in preprocessor.final_feature_columns:
        print(" -", c)


if __name__ == "__main__":
    main()
