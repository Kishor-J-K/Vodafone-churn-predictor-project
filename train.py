"""
train.py — Train XGBoost churn model on Vodafone/Telco dataset.

Usage:
    python train.py
    python train.py --data "C:/path/to/voice_customer_churn.csv"
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "voice_customer_churn.csv")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


def load_and_clean(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    df = pd.read_excel(path) if ext in (".xlsx", ".xls") else pd.read_csv(path)

    print(f"Columns found     : {df.columns.tolist()}")
    print(f"Shape             : {df.shape}")
    print(f"Nulls per column  :\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")

    # Drop ID column
    id_cols = [c for c in df.columns if c.lower() in ("customerid", "customer_id", "id")]
    df.drop(columns=id_cols, inplace=True, errors="ignore")

    # Fix TotalCharges stored as string
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Detect and normalise Churn column
    churn_col = next((c for c in df.columns if c.lower() == "churn"), None)
    if churn_col is None:
        raise ValueError(f"No 'Churn' column found. Columns: {df.columns.tolist()}")
    if churn_col != "Churn":
        df.rename(columns={churn_col: "Churn"}, inplace=True)

    if df["Churn"].dtype == object:
        df["Churn"] = (df["Churn"].str.strip().str.lower() == "yes").astype(int)
    else:
        df["Churn"] = df["Churn"].astype(int)

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    print(f"Numeric features    : {num_cols}")
    print(f"Categorical features: {cat_cols}\n")

    # Numeric pipeline: impute median → scale
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # Categorical pipeline: impute most_frequent → one-hot encode
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ])


def evaluate(name: str, clf, X_test, y_test):
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    f1  = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"  F1={f1:.4f}  AUC={auc:.4f}")
    print(classification_report(y_test, preds, target_names=["Stay", "Churn"]))
    return f1, auc


def train(data_path: str):
    print(f"\nLoading data from: {data_path}\n")
    df = load_and_clean(data_path)

    print(f"Churn rate: {df['Churn'].mean():.2%}")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc  = preprocessor.transform(X_test)

    # SMOTE — safe now, NaNs handled in preprocessor
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_enc, y_train)
    print(f"After SMOTE — train size: {X_res.shape[0]}, churn rate: {y_res.mean():.2%}\n")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost":             XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1
        ),
    }

    results = {}
    for name, clf in models.items():
        print(f"Training {name}...")
        clf.fit(X_res, y_res)
        f1, auc = evaluate(name, clf, X_test_enc, y_test)
        results[name] = {"model": clf, "f1": f1, "auc": auc}

    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_name]["model"]
    print(f"\n★  Best model: {best_name}  (F1={results[best_name]['f1']:.4f}  AUC={results[best_name]['auc']:.4f})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model,   os.path.join(MODEL_DIR, "churn_model.pkl"))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))
    print(f"Saved model        → {MODEL_DIR}/churn_model.pkl")
    print(f"Saved preprocessor → {MODEL_DIR}/preprocessor.pkl")

    # Feature importance plot
    if hasattr(best_model, "feature_importances_"):
        num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_encoder = preprocessor.named_transformers_["cat"]["encoder"]
        cat_cols_encoded = cat_encoder.get_feature_names_out().tolist()
        feature_names = num_cols + cat_cols_encoded

        importances = pd.Series(best_model.feature_importances_, index=feature_names)
        top = importances.nlargest(15).sort_values()
        fig, ax = plt.subplots(figsize=(9, 6))
        top.plot(kind="barh", ax=ax, color="#185FA5")
        ax.set_title("Top 15 feature importances")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(os.path.join(MODEL_DIR, "feature_importance.png"), dpi=150)
        print("Saved feature importance → model/feature_importance.png")

    # Confusion matrix
    preds = best_model.predict(X_test_enc)
    cm = confusion_matrix(y_test, preds)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Stay", "Churn"]).plot(ax=ax2)
    ax2.set_title(f"Confusion matrix — {best_name}")
    fig2.tight_layout()
    fig2.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    print("Saved confusion matrix  → model/confusion_matrix.png")

    return best_model, preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH, help="Path to CSV or XLSX data file")
    args = parser.parse_args()
    train(args.data)