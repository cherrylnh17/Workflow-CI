import os
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Konfigurasi
DATA_DIR = 'data_preprocessing'
EXPERIMENT_NAME = 'HateSpeech_Classification'
MODEL_NAME = 'LogisticRegression_HateSpeech'

# Load data
def load_preprocessed_data(data_dir: str):
    logger.info(f"Memuat data dari: {data_dir}")
    X_train = sp.load_npz(os.path.join(data_dir, 'X_train_tfidf.npz'))
    X_test  = sp.load_npz(os.path.join(data_dir, 'X_test_tfidf.npz'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test  = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    logger.info(f"Data dimuat — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# Main training
def train():
    X_train, X_test, y_train, y_test = load_preprocessed_data(DATA_DIR)

    mlflow.set_experiment(EXPERIMENT_NAME)

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name='LogisticRegression_baseline', nested=True):
        logger.info("Memulai training Logistic Regression...")

        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver='lbfgs',
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec  = recall_score(y_test, y_pred, average='weighted')
        f1   = f1_score(y_test, y_pred, average='weighted')
        auc  = roc_auc_score(y_test, y_prob)

        logger.info(f"Accuracy : {acc:.4f}")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"Recall   : {rec:.4f}")
        logger.info(f"F1-Score : {f1:.4f}")
        logger.info(f"ROC-AUC  : {auc:.4f}")

        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred,
                                    target_names=['Non-HS', 'HS']))

    logger.info("Training selesai. Buka MLflow UI dengan: mlflow ui")


if __name__ == '__main__':
    train()
