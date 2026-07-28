# Customer Churn Prediction in the Banking Sector

An end-to-end machine learning project for predicting credit card customer churn in
the banking sector. The project includes exploratory analysis, preprocessing,
customer segmentation, model training, model comparison, and an interactive
Streamlit prediction app.

The workflow is inspired by:

Tran, H., Le, N., & Nguyen, V.-H. (2023). Customer churn prediction in the
banking sector using machine learning-based classification models.
Interdisciplinary Journal of Information, Knowledge, and Management, 18, 87-105.
https://doi.org/10.28945/5086

## Project Goals

This project answers two practical questions:

1. Which machine learning model predicts bank customer churn best?
2. Does customer segmentation with K-Means improve churn prediction performance?

The current saved app artifacts serve a Random Forest model.

## Dataset

Dataset: Credit Card Customers - Kaggle
https://www.kaggle.com/datasets/anwarsan/credit-card-bank-churn

Expected raw file:

```text
data/raw/BankChurners.csv
```

The raw dataset contains 10,127 customers. The target column is
`Attrition_Flag`, where `Attrited Customer` represents churn and
`Existing Customer` represents non-churn.

The raw CSV is intentionally ignored by Git because it is a local dataset file.

## Project Structure

```text
Customer-Churn-Prediction/
|-- app/
|   `-- streamlit_app.py
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|   `-- saved_models/
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_preprocessing.ipynb
|   |-- 03_clustering.ipynb
|   |-- 04_modeling.ipynb
|   `-- 05_results_comparison.ipynb
|-- outputs/
|   `-- metrics/
|-- scripts/
|   `-- build_artifacts.py
|-- src/
|   `-- preprocessing.py
|-- tests/
|   `-- test_serving_artifacts.py
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Quick Start on Windows PowerShell

Run these commands from PowerShell.

```powershell
cd "D:\Churn Prediction\Customer-Churn-Prediction"
```

If you want to use the existing virtual environment in the parent folder:

```powershell
..\.venv-1\Scripts\Activate.ps1
```

If activation is blocked by PowerShell policy for this terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
..\.venv-1\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Build the serving preprocessor artifact:

```powershell
python scripts\build_artifacts.py
```

Run tests:

```powershell
pytest
```

Run the Streamlit app:

```powershell
streamlit run app\streamlit_app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Fresh Environment Setup

If the existing parent virtual environment is missing or you want a new one
inside the project, run:

```powershell
cd "D:\Churn Prediction\Customer-Churn-Prediction"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then continue with:

```powershell
python scripts\build_artifacts.py
pytest
streamlit run app\streamlit_app.py
```

## Full Rebuild Workflow

Use this path if you want to regenerate processed data, model artifacts, and
metrics from the raw Kaggle dataset.

1. Place the raw dataset here:

```text
data/raw/BankChurners.csv
```

2. Start Jupyter:

```powershell
jupyter notebook
```

3. Run notebooks in this order:

```text
notebooks/01_eda.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_clustering.ipynb
notebooks/04_modeling.ipynb
notebooks/05_results_comparison.ipynb
```

4. Rebuild the serving preprocessor after training:

```powershell
python scripts\build_artifacts.py
```

5. Verify and run the app:

```powershell
pytest
streamlit run app\streamlit_app.py
```

## What the App Uses

The Streamlit app loads:

```text
models/saved_models/best_model_*.pkl
models/saved_models/preprocessor.pkl
models/saved_models/feature_columns.pkl
```

`scripts/build_artifacts.py` fits the shared preprocessor in
`src/preprocessing.py` and aligns it with `feature_columns.pkl`, so the app
serves the exact feature shape expected by the trained model.

## Model Training Summary

The project compares:

| Model | Name |
|---|---|
| KNN | K-Nearest Neighbors |
| LR | Logistic Regression |
| DT | Decision Tree |
| RF | Random Forest |
| SVM | Support Vector Machine |

The workflow applies SMOTE only to the training split to avoid test data
leakage. Results are compared with and without K-Means segmentation.

## Useful Commands

Compile Python files:

```powershell
python -m compileall src scripts app tests
```

Run tests:

```powershell
pytest
```

Rebuild serving artifact:

```powershell
python scripts\build_artifacts.py
```

Run app on a specific port:

```powershell
streamlit run app\streamlit_app.py --server.port 8501
```

Remove local cache files:

```powershell
Remove-Item -Recurse -Force .pytest_cache, src\__pycache__, app\__pycache__, scripts\__pycache__, tests\__pycache__ -ErrorAction SilentlyContinue
```

Remove local MLflow files after closing Jupyter/Python processes:

```powershell
Remove-Item -Recurse -Force notebooks\mlruns, notebooks\mlflow.db -ErrorAction SilentlyContinue
```

## Git Ignore Notes

The `.gitignore` file excludes local data, generated model pickle files,
MLflow runs/databases, Python caches, notebook checkpoints, virtual
environments, environment files, and OS clutter.

Keep source code, notebooks, tests, and lightweight metrics in Git. Regenerate
large runtime artifacts locally when needed.

## Troubleshooting

If the app says the preprocessor is missing:

```powershell
python scripts\build_artifacts.py
```

If the app says the model and preprocessor feature counts do not match, retrain
or rebuild artifacts in this order:

```powershell
jupyter notebook
python scripts\build_artifacts.py
pytest
streamlit run app\streamlit_app.py
```

If `notebooks\mlflow.db` cannot be deleted, close any running Jupyter, MLflow,
or Python process first, then run:

```powershell
Remove-Item -LiteralPath "notebooks\mlflow.db" -Force
```

## License

This project is for educational and portfolio use. The Kaggle dataset is subject
to its own license terms. The reference paper is licensed under CC BY-NC 4.0.
