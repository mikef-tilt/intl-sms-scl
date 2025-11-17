"""
Script to train machine learning models on SMS risk features.

This script:
1. Loads SMS risk features from Azure blob storage
2. Extracts the same train/test UserLoanIds from cached data
3. Trains XGBoost, Logistic Regression, and Neural Network models
4. Evaluates and logs results similar to apply_encoder_train_model.py
"""

from helpers import TableManager
from data_helper import load_data, query_spine
import polars as pl
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ============================================================================
# Configuration
# ============================================================================

# Azure blob configuration
BLOB_URL = "https://efwusdsdata01.blob.core.windows.net/mike/tmp/ensenada_sms.parquet"

# Feature columns to use
FEATURE_COLUMNS = [
    'sms_average_risk',
    'sms_weighted_risk', 
    'sms_risk_quantile_25',
    'sms_risk_quantile_75',
    'sms_risk_quantile_90'
]

# Cached train/test data to extract UserLoanIds
TRAIN_DATA_CACHE_FILE = "cache/train_data.parquet"
TEST_DATA_CACHE_FILE = "cache/test_data.parquet"

# Spine data configuration (to get UserId mapping)
CONTAINER_NAME = "mike"
DIRECTORY_PATH = "data-tmp/ensenada/raw"
TRAIN_START_DATE = "2025-08-01"
TRAIN_END_DATE = "2025-09-01"
TEST_START_DATE = "2025-09-01"
TEST_END_DATE = "2025-09-15"

# Model selection
MODEL_TYPE = 'all'  # Options: 'xgboost', 'logistic', 'neuralnet', 'all'

# XGBoost configuration (using best parameters from grid search)
XGBOOST_PARAMS = {
    'n_estimators': 3000,
    'max_depth': 5,
    'learning_rate': 0.008,
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    'min_child_weight': 10,
    'gamma': 1.0,
    'reg_alpha': 1.0,
    'reg_lambda': 10.0,
    'random_state': 42,
    'eval_metric': 'auc',
    'device': 'cuda:0',
    'scale_pos_weight': None,
    'early_stopping_rounds': 250
}

# Logistic Regression configuration
LOGISTIC_PARAMS = {
    'penalty': 'l2',
    'C': 1.0,
    'solver': 'lbfgs',
    'max_iter': 1000,
    'random_state': 42,
    'class_weight': 'balanced'
}

# Neural Network configuration
NEURALNET_PARAMS = {
    'hidden_layer_sizes': (64, 32),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.01,
    'batch_size': 256,
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.001,
    'max_iter': 200,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 20,
    'random_state': 42,
    'verbose': True
}


# ============================================================================
# Helper Functions
# ============================================================================

def load_sms_risk_data(blob_url: str) -> pl.DataFrame:
    """
    Load SMS risk features from Azure blob storage.

    Args:
        blob_url: URL to the parquet file in Azure blob storage

    Returns:
        Polars DataFrame with SMS risk features
    """
    from io import BytesIO

    print(f"\nLoading SMS risk data from Azure blob storage...")
    print(f"  URL: {blob_url}")

    # Extract container and path from URL
    # URL format: https://efwusdsdata01.blob.core.windows.net/mike/tmp/ensenada_sms.parquet
    parts = blob_url.replace("https://efwusdsdata01.blob.core.windows.net/", "").split("/", 1)
    container = parts[0]
    blob_path = parts[1]

    print(f"  Container: {container}")
    print(f"  Blob path: {blob_path}")

    # Use TableManager to read the file directly from Azure
    tm = TableManager(container_name=container, directory_path="tmp")

    # Get the file system client and read the parquet file
    file_system_client = tm.service_client.get_file_system_client(container)
    file_client = file_system_client.get_file_client(blob_path)

    print(f"  Downloading file...")

    # Download the file content
    download = file_client.download_file()
    file_content = download.readall()

    print(f"  Downloaded {len(file_content):,} bytes")
    print(f"  Reading parquet data...")

    # Try reading with pyarrow first (more forgiving), then convert to polars
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(BytesIO(file_content))
        df = pl.from_arrow(table)
    except Exception as e:
        print(f"  ⚠ PyArrow failed: {e}")
        print(f"  Trying with Polars directly...")
        df = pl.read_parquet(BytesIO(file_content))

    print(f"  ✓ Loaded {len(df):,} rows")
    print(f"  Columns: {df.columns}")

    return df


def prepare_data(df_sms: pl.DataFrame, df_train_spine: pl.DataFrame, df_test_spine: pl.DataFrame):
    """
    Prepare training and test data by joining SMS risk features with spine data on UserId and loan_created_at.

    Args:
        df_sms: DataFrame with SMS risk features (with user_id, loan_created_at, and loan_is_default_21d)
        df_train_spine: Training spine data with UserId, UserLoanId, CreatedAt
        df_test_spine: Test spine data with UserId, UserLoanId, CreatedAt

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    print(f"\nPreparing data...")

    print(f"  Train spine: {len(df_train_spine):,} rows")
    print(f"  Test spine: {len(df_test_spine):,} rows")

    print(f"\n  Processing SMS data...")
    print(f"    Total SMS rows: {len(df_sms):,}")

    # Rename columns in SMS data to match spine data format
    df_sms_renamed = df_sms.rename({
        'user_id': 'UserId',
        'loan_created_at': 'CreatedAt'
    })

    # Truncate CreatedAt to minute precision for both datasets
    print(f"  Truncating timestamps to minute precision...")
    df_sms_renamed = df_sms_renamed.with_columns(
        pl.col('CreatedAt').dt.truncate('1m').alias('CreatedAt')
    )

    # Select only the columns we need from SMS data
    df_sms_subset = df_sms_renamed.select(['UserId', 'CreatedAt', 'loan_is_default_21d'] + FEATURE_COLUMNS)

    print(f"  SMS data shape: {df_sms_subset.shape}")
    print(f"  SMS unique UserIds: {df_sms_subset['UserId'].n_unique():,}")
    print(f"  SMS unique CreatedAt: {df_sms_subset['CreatedAt'].n_unique():,}")

    # Prepare spine data with UserId and CreatedAt (truncated to minute)
    train_spine_subset = df_train_spine.select(['UserId', 'UserLoanId', 'CreatedAt', 'DefaultDPD21']).with_columns(
        pl.col('CreatedAt').dt.truncate('1m').alias('CreatedAt')
    )
    test_spine_subset = df_test_spine.select(['UserId', 'UserLoanId', 'CreatedAt', 'DefaultDPD21']).with_columns(
        pl.col('CreatedAt').dt.truncate('1m').alias('CreatedAt')
    )

    # Join SMS risk features with train/test splits on UserId and CreatedAt
    print(f"\n  Joining on UserId and CreatedAt (truncated to minute)...")
    df_train = train_spine_subset.join(df_sms_subset, on=['UserId', 'CreatedAt'], how='inner')
    df_test = test_spine_subset.join(df_sms_subset, on=['UserId', 'CreatedAt'], how='inner')

    print(f"  Train samples after join: {len(df_train):,}")
    print(f"  Test samples after join: {len(df_test):,}")

    # Check for missing values
    train_nulls = df_train.select(FEATURE_COLUMNS).null_count()
    test_nulls = df_test.select(FEATURE_COLUMNS).null_count()

    print(f"\n  Checking for missing values...")
    print(f"    Train nulls: {train_nulls}")
    print(f"    Test nulls: {test_nulls}")

    # Fill missing values with 0 (assuming missing SMS means no risk)
    for col in FEATURE_COLUMNS:
        df_train = df_train.with_columns(pl.col(col).fill_null(0))
        df_test = df_test.with_columns(pl.col(col).fill_null(0))

    # Extract features and labels
    X_train = df_train.select(FEATURE_COLUMNS).to_numpy()
    y_train = df_train.select('loan_is_default_21d').to_numpy().ravel()
    X_test = df_test.select(FEATURE_COLUMNS).to_numpy()
    y_test = df_test.select('loan_is_default_21d').to_numpy().ravel()

    print(f"\n  Final shapes:")
    print(f"    X_train: {X_train.shape}")
    print(f"    y_train: {y_train.shape}")
    print(f"    X_test: {X_test.shape}")
    print(f"    y_test: {y_test.shape}")

    print(f"\n  Label distribution:")
    print(f"    Train: {np.bincount(y_train)}")
    print(f"    Test: {np.bincount(y_test)}")

    return X_train, y_train, X_test, y_test


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  params: dict) -> XGBClassifier:
    """Train XGBoost classifier."""
    print(f"\n{'='*60}")
    print("Training XGBoost")
    print(f"{'='*60}")

    # Calculate scale_pos_weight
    n_negative = np.sum(y_train == 0)
    n_positive = np.sum(y_train == 1)
    scale_pos_weight = n_negative / n_positive

    print(f"  Class distribution:")
    print(f"    Negative: {n_negative:,} ({n_negative/len(y_train)*100:.2f}%)")
    print(f"    Positive: {n_positive:,} ({n_positive/len(y_train)*100:.2f}%)")
    print(f"    scale_pos_weight: {scale_pos_weight:.4f}")

    params_copy = params.copy()
    if params_copy.get('scale_pos_weight') is None:
        params_copy['scale_pos_weight'] = scale_pos_weight

    early_stopping_rounds = params_copy.pop('early_stopping_rounds', None)
    if early_stopping_rounds is not None:
        params_copy['early_stopping_rounds'] = early_stopping_rounds

    model = XGBClassifier(**params_copy)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50
    )

    print(f"✓ Training complete!")
    if hasattr(model, 'best_iteration'):
        print(f"  Best iteration: {model.best_iteration}")
        print(f"  Best score: {model.best_score:.4f}")

    return model


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              params: dict):
    """Train Logistic Regression classifier."""
    print(f"\n{'='*60}")
    print("Training Logistic Regression")
    print(f"{'='*60}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(**params)
    model.fit(X_train_scaled, y_train)

    print(f"✓ Training complete!")

    return model, scaler, X_train_scaled, X_test_scaled


def train_neural_network(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        params: dict):
    """Train Neural Network classifier."""
    print(f"\n{'='*60}")
    print("Training Neural Network")
    print(f"{'='*60}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(**params)
    model.fit(X_train_scaled, y_train)

    print(f"✓ Training complete!")
    print(f"  Training iterations: {model.n_iter_}")

    return model, scaler, X_train_scaled, X_test_scaled


def evaluate_model(model, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray, model_name: str):
    """Evaluate model and return metrics."""
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation - ROC AUC")
    print(f"{'='*60}")

    # Get predictions
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Calculate ROC AUC
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"\n📊 ROC AUC Scores:")
    print(f"  Train ROC AUC: {train_auc:.4f}")
    print(f"  Test ROC AUC:  {test_auc:.4f}")
    print(f"  Difference (Overfitting): {train_auc - test_auc:.4f}")

    return train_auc, test_auc, y_train_proba, y_test_proba


def plot_roc_curves(y_test: np.ndarray, results: dict, output_file: str = "cache/sms_risk_roc_curve.png"):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))

    for model_name, (test_auc, y_test_proba) in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {test_auc:.4f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5000)', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - SMS Risk Features', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_file, dpi=150)
    print(f"\n✓ ROC curve saved to: {output_file}")
    plt.close()


def save_results(results: dict, train_results: dict, filename: str = "cache/sms_risk_results.txt"):
    """Save model results to text file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL TRAINING RESULTS - SMS RISK FEATURES\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'='*80}\n\n")

        f.write("FEATURES USED\n")
        f.write("-"*80 + "\n")
        for feat in FEATURE_COLUMNS:
            f.write(f"  - {feat}\n")
        f.write("\n")

        # XGBoost Configuration
        f.write("XGBOOST CONFIGURATION\n")
        f.write("-"*80 + "\n")
        for param, value in XGBOOST_PARAMS.items():
            f.write(f"  {param:25s}: {value}\n")
        f.write("\n")

        # Logistic Regression Configuration
        f.write("LOGISTIC REGRESSION CONFIGURATION\n")
        f.write("-"*80 + "\n")
        for param, value in LOGISTIC_PARAMS.items():
            f.write(f"  {param:25s}: {value}\n")
        f.write("\n")

        # Neural Network Configuration
        f.write("NEURAL NETWORK CONFIGURATION\n")
        f.write("-"*80 + "\n")
        for param, value in NEURALNET_PARAMS.items():
            f.write(f"  {param:25s}: {value}\n")
        f.write("\n")

        # Performance Results
        f.write("="*80 + "\n")
        f.write("PERFORMANCE RESULTS (ROC AUC)\n")
        f.write("="*80 + "\n\n")

        sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)

        f.write(f"{'Model':<25s} {'Train AUC':>12s} {'Test AUC':>12s} {'Overfitting':>12s}\n")
        f.write("-"*80 + "\n")

        for model_name, (test_auc, _) in sorted_results:
            train_auc = train_results.get(model_name, 0.0)
            overfitting = train_auc - test_auc
            f.write(f"{model_name:<25s} {train_auc:>12.4f} {test_auc:>12.4f} {overfitting:>12.4f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("RANKING BY TEST AUC\n")
        f.write("="*80 + "\n")
        for i, (model_name, (test_auc, _)) in enumerate(sorted_results, 1):
            f.write(f"  {i}. {model_name:<25s}: {test_auc:.4f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("NOTES\n")
        f.write("="*80 + "\n")
        f.write("- Features: SMS risk aggregations (average, weighted, quantiles)\n")
        f.write("- No text preprocessing or TF-IDF encoding applied\n")
        f.write("- Direct modeling on pre-computed risk scores\n")
        f.write("\n")

    print(f"✓ Results saved to: {filename}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("="*60)
    print("SMS Risk Features - Model Training")
    print("="*60)

    # 1. Load spine data to get UserId and CreatedAt mapping
    print(f"\n1. Loading spine data...")
    tm = TableManager(container_name=CONTAINER_NAME, directory_path=DIRECTORY_PATH)
    df_train_spine, df_test_spine = query_spine(
        tm,
        train_start_date=TRAIN_START_DATE,
        train_end_date=TRAIN_END_DATE,
        test_start_date=TEST_START_DATE,
        test_end_date=TEST_END_DATE
    )

    print(f"  ✓ Train spine: {len(df_train_spine):,} rows")
    print(f"  ✓ Test spine: {len(df_test_spine):,} rows")
    print(f"  Spine columns: {df_train_spine.columns}")

    # 2. Load SMS risk features from Azure
    print(f"\n2. Loading SMS risk features from Azure...")
    df_sms = load_sms_risk_data(BLOB_URL)

    # 3. Prepare data
    print(f"\n3. Preparing training and test data...")
    X_train, y_train, X_test, y_test = prepare_data(df_sms, df_train_spine, df_test_spine)

    # 4. Train models
    results = {}
    train_results = {}

    print(f"\n4. Training models (MODEL_TYPE={MODEL_TYPE})...")

    if MODEL_TYPE in ['xgboost', 'all']:
        xgb_model = train_xgboost(X_train, y_train, X_test, y_test, XGBOOST_PARAMS)
        train_auc, test_auc, y_train_proba, y_test_proba = evaluate_model(
            xgb_model, X_train, y_train, X_test, y_test, "XGBoost"
        )
        results['XGBoost'] = (test_auc, y_test_proba)
        train_results['XGBoost'] = train_auc

    if MODEL_TYPE in ['logistic', 'all']:
        lr_model, lr_scaler, X_train_scaled, X_test_scaled = train_logistic_regression(
            X_train, y_train, X_test, y_test, LOGISTIC_PARAMS
        )
        train_auc, test_auc, y_train_proba, y_test_proba = evaluate_model(
            lr_model, X_train_scaled, y_train, X_test_scaled, y_test, "Logistic Regression"
        )
        results['Logistic Regression'] = (test_auc, y_test_proba)
        train_results['Logistic Regression'] = train_auc

    if MODEL_TYPE in ['neuralnet', 'all']:
        nn_model, nn_scaler, X_train_scaled, X_test_scaled = train_neural_network(
            X_train, y_train, X_test, y_test, NEURALNET_PARAMS
        )
        train_auc, test_auc, y_train_proba, y_test_proba = evaluate_model(
            nn_model, X_train_scaled, y_train, X_test_scaled, y_test, "Neural Network"
        )
        results['Neural Network'] = (test_auc, y_test_proba)
        train_results['Neural Network'] = train_auc

    # 5. Plot ROC curves
    print(f"\n5. Plotting ROC curves...")
    plot_roc_curves(y_test, results)

    # 6. Save results
    print(f"\n6. Saving results...")
    save_results(results, train_results)

    # Summary
    print(f"\n{'='*60}")
    print("Summary - Model Comparison")
    print(f"{'='*60}")
    print(f"✓ Models trained on SMS risk features")
    print(f"✓ Features used: {len(FEATURE_COLUMNS)}")
    print(f"\n📊 Test ROC AUC Scores:")
    for model_name, (test_auc, _) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
        print(f"  {model_name:20s}: {test_auc:.4f}")

    return results


if __name__ == "__main__":
    main()

