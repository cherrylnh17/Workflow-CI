import os
import logging
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = 'HateSpeech_Classification'


def load_data(data_dir: str):
    logger.info(f"Memuat data dari: {data_dir}")
    X_train = sp.load_npz(os.path.join(data_dir, 'X_train_tfidf.npz'))
    X_test  = sp.load_npz(os.path.join(data_dir, 'X_test_tfidf.npz'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test  = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    logger.info(f"Data dimuat — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train(data_dir: str = 'data_preprocessing'):
    X_train, X_test, y_train, y_test = load_data(data_dir)

    # Selalu tracking ke lokal agar mlruns/ terbentuk di disk
    # CI akan log ulang ke DagsHub via step terpisah
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    mlflow.sklearn.autolog(
        log_models=True,
        log_input_examples=False,
        log_model_signatures=True,
    )

    logger.info("Memulai training Logistic Regression...")

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    mlflow.log_metric('test_accuracy', acc)
    mlflow.log_metric('test_precision_weighted', prec)
    mlflow.log_metric('test_recall_weighted', rec)
    mlflow.log_metric('test_f1_weighted', f1)
    mlflow.log_metric('test_roc_auc', auc)
    mlflow.log_param('data_dir', data_dir)

    logger.info(f"Accuracy : {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall   : {rec:.4f}")
    logger.info(f"F1-Score : {f1:.4f}")
    logger.info(f"ROC-AUC  : {auc:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Non-HS', 'HS']))

    logger.info("Training selesai!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='data_preprocessing',
                        help='Path ke folder preprocessing')
    args = parser.parse_args()
    train(data_dir=args.data_dir)