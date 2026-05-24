"""
modelling.py — Versi MLflow Project untuk CI Pipeline
Dipanggil via: mlflow run MLProject/ -P n_estimators=200
"""
import pandas as pd
import numpy as np
import argparse, os, pickle, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, log_loss)
import mlflow
import mlflow.sklearn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',
        type=str, default='WA_Fn-UseC_-Telco-Customer-Churn_preprocessing')
    p.add_argument('--n_estimators',      type=int, default=200)
    p.add_argument('--max_depth',         type=int, default=10)
    p.add_argument('--min_samples_split', type=int, default=5)
    p.add_argument('--min_samples_leaf',  type=int, default=2)
    return p.parse_args()


def load_data(data_path):
    X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
    X_test  = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv')).squeeze()
    y_test  = pd.read_csv(os.path.join(data_path, 'y_test.csv')).squeeze()
    print(f"[INFO] Data: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def main():
    args = parse_args()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("Telco-Churn-CI-Pipeline")

    X_train, X_test, y_train, y_test = load_data(args.data_path)

    with mlflow.start_run(run_name=f"RF-n{args.n_estimators}-d{args.max_depth}"):
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        mlflow.log_params({
            'n_estimators'     : args.n_estimators,
            'max_depth'        : args.max_depth,
            'min_samples_split': args.min_samples_split,
            'min_samples_leaf' : args.min_samples_leaf,
        })

        metrics = {
            'training_accuracy': accuracy_score(y_train, model.predict(X_train)),
            'test_accuracy'    : accuracy_score(y_test, y_pred),
            'f1_score'         : f1_score(y_test, y_pred),
            'precision'        : precision_score(y_test, y_pred),
            'recall'           : recall_score(y_test, y_pred),
            'roc_auc'          : roc_auc_score(y_test, y_proba),
            'log_loss'         : log_loss(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/model.pkl", "wb") as f:
            pickle.dump(model, f)

        print("\n=== HASIL CI PIPELINE ===")
        for k, v in metrics.items():
            print(f"  {k:<25}: {v:.4f}")


if __name__ == "__main__":
    main()