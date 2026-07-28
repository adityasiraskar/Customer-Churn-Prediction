from pathlib import Path
import sys

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import ChurnPreprocessor  # noqa: E402

RAW_DATA_PATH = ROOT / "data" / "raw" / "BankChurners.csv"
MODEL_PATH = ROOT / "models" / "saved_models" / "best_model_RF.pkl"
PREPROCESSOR_PATH = ROOT / "models" / "saved_models" / "preprocessor.pkl"
FEATURE_COLUMNS_PATH = ROOT / "models" / "saved_models" / "feature_columns.pkl"


def _model_input(model, features_df):
    if getattr(model, "feature_names_in_", None) is not None:
        return features_df
    return features_df.to_numpy()


def test_preprocessor_respects_saved_feature_columns():
    raw_df = pd.read_csv(RAW_DATA_PATH)
    expected_columns = list(joblib.load(FEATURE_COLUMNS_PATH))

    preprocessor = ChurnPreprocessor(feature_columns=expected_columns).fit(raw_df)
    features_df = preprocessor.transform(
        raw_df.head(5).drop(columns=["CLIENTNUM", "Attrition_Flag"], errors="ignore")
    )

    assert list(features_df.columns) == expected_columns


def test_saved_model_accepts_saved_preprocessor_output():
    model = joblib.load(MODEL_PATH)
    preprocessor = ChurnPreprocessor.load(PREPROCESSOR_PATH)
    raw_df = pd.read_csv(RAW_DATA_PATH).head(3)
    features_df = preprocessor.transform(
        raw_df.drop(columns=["CLIENTNUM", "Attrition_Flag"], errors="ignore")
    )

    assert features_df.shape[1] == getattr(model, "n_features_in_", features_df.shape[1])

    probabilities = model.predict_proba(_model_input(model, features_df))
    assert probabilities.shape == (3, 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
