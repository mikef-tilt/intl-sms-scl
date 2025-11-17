"""
Script to evaluate model performance on first loans vs. repeat loans.

This script:
1. Loads spine data to identify first loans (CumulativeLoans==1) vs. repeat loans
2. Trains models on all loans using both TF-IDF and SMS risk features
3. Evaluates performance separately on first loans and repeat loans in the test set
4. Compares performance across both approaches and loan types
"""

from helpers import TableManager
from data_helper import load_data, load_encoder, query_spine
import polars as pl
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import os
from io import BytesIO


# ============================================================================
# Configuration
# ============================================================================

# Data configuration
CONTAINER_NAME = "mike"
DIRECTORY_PATH = "data-tmp/ensenada/raw"
TRAIN_START_DATE = "2025-08-01"
TRAIN_END_DATE = "2025-09-01"
TEST_START_DATE = "2025-09-01"
TEST_END_DATE = "2025-09-15"

# TF-IDF approach cache files
TFIDF_ENCODER_CACHE = "cache/tfidf_encoder.pkl"
TFIDF_TRAIN_DATA_CACHE = "cache/train_data.parquet"
TFIDF_TEST_DATA_CACHE = "cache/test_data.parquet"
TFIDF_TRAIN_EMBEDDINGS_CACHE = "cache/train_tfidf_features.npy"
TFIDF_TEST_EMBEDDINGS_CACHE = "cache/test_tfidf_features.npy"

# SMS risk features configuration
SMS_RISK_BLOB_URL = "https://efwusdsdata01.blob.core.windows.net/mike/tmp/ensenada_sms.parquet"
SMS_RISK_FEATURE_COLUMNS = [
    'sms_average_risk',
    'sms_weighted_risk',
    'sms_risk_quantile_25',
    'sms_risk_quantile_75',
    'sms_risk_quantile_90'
]

# Model parameters (using best parameters from previous experiments)
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
    'device': 'cpu',  # Use CPU for compatibility
    'early_stopping_rounds': 250
}

LOGISTIC_PARAMS = {
    'penalty': 'l2',
    'C': 1.0,
    'solver': 'lbfgs',
    'max_iter': 1000,
    'random_state': 42,
    'class_weight': 'balanced'
}

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
    'verbose': False
}

# Output configuration
OUTPUT_DIR = "cache/first_vs_repeat_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Helper Functions
# ============================================================================

def load_sms_risk_data(blob_url: str) -> pl.DataFrame:
    """Load SMS risk features from Azure blob storage."""
    from io import BytesIO
    
    print(f"\nLoading SMS risk data from Azure blob storage...")
    print(f"  URL: {blob_url}")
    
    # Extract container and path from URL
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


def prepare_sms_risk_data(df_sms: pl.DataFrame, df_train_spine: pl.DataFrame,
                          df_test_spine: pl.DataFrame):
    """Prepare SMS risk features data by joining with spine data."""
    print(f"\nPreparing SMS risk features data...")

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
    df_sms_subset = df_sms_renamed.select(['UserId', 'CreatedAt', 'loan_is_default_21d'] + SMS_RISK_FEATURE_COLUMNS)

    # Prepare spine data with UserId and CreatedAt (truncated to minute)
    train_spine_subset = df_train_spine.select(['UserId', 'UserLoanId', 'CreatedAt', 'DefaultDPD21', 'CumulativeLoans']).with_columns(
        pl.col('CreatedAt').dt.truncate('1m').alias('CreatedAt')
    )
    test_spine_subset = df_test_spine.select(['UserId', 'UserLoanId', 'CreatedAt', 'DefaultDPD21', 'CumulativeLoans']).with_columns(
        pl.col('CreatedAt').dt.truncate('1m').alias('CreatedAt')
    )

    # Join SMS risk features with train/test splits on UserId and CreatedAt
    print(f"  Joining on UserId and CreatedAt (truncated to minute)...")
    df_train = train_spine_subset.join(df_sms_subset, on=['UserId', 'CreatedAt'], how='inner')
    df_test = test_spine_subset.join(df_sms_subset, on=['UserId', 'CreatedAt'], how='inner')

    print(f"  Train samples after join: {len(df_train):,}")
    print(f"  Test samples after join: {len(df_test):,}")

    # Handle missing values
    print(f"\n  Handling missing values...")
    for col in SMS_RISK_FEATURE_COLUMNS:
        df_train = df_train.with_columns(pl.col(col).fill_null(0))
        df_test = df_test.with_columns(pl.col(col).fill_null(0))

    # Extract features and labels
    X_train = df_train.select(SMS_RISK_FEATURE_COLUMNS).to_numpy()
    y_train = df_train['DefaultDPD21'].to_numpy()
    X_test = df_test.select(SMS_RISK_FEATURE_COLUMNS).to_numpy()
    y_test = df_test['DefaultDPD21'].to_numpy()
    cumulative_loans_test = df_test['CumulativeLoans'].to_numpy()

    print(f"\n  Final shapes:")
    print(f"    X_train: {X_train.shape}")
    print(f"    X_test: {X_test.shape}")

    return X_train, y_train, X_test, y_test, cumulative_loans_test


def prepare_tfidf_data(encoder, df_train: pl.DataFrame, df_test: pl.DataFrame,
                       df_test_spine: pl.DataFrame):
    """Prepare TF-IDF features data."""
    print(f"\nPreparing TF-IDF features data...")

    # Load TF-IDF embeddings (use cached versions - no deduplication)
    print(f"  Loading cached TF-IDF embeddings...")
    X_train = np.load(TFIDF_TRAIN_EMBEDDINGS_CACHE)
    X_test = np.load(TFIDF_TEST_EMBEDDINGS_CACHE)
    print(f"    Train embeddings: {X_train.shape}")
    print(f"    Test embeddings: {X_test.shape}")

    # Get labels
    y_train = df_train['DefaultDPD21'].to_numpy()
    y_test = df_test['DefaultDPD21'].to_numpy()

    print(f"    y_train: {y_train.shape}")
    print(f"    y_test: {y_test.shape}")

    # Join test data with spine to get CumulativeLoans
    # Use UserLoanId as the primary key
    print(f"\n  Joining test data with spine to get CumulativeLoans...")
    print(f"    Test data rows: {len(df_test):,}")
    print(f"    Test spine rows: {len(df_test_spine):,}")

    # Create a mapping from UserLoanId to CumulativeLoans
    # Use the spine data which has the authoritative CumulativeLoans values
    spine_lookup = df_test_spine.select(['UserLoanId', 'CumulativeLoans']).unique(subset=['UserLoanId'], keep='first')

    # Join on UserLoanId only
    df_test_with_cumloans = df_test.join(
        spine_lookup,
        on='UserLoanId',
        how='left'
    )

    print(f"    Joined data rows: {len(df_test_with_cumloans):,}")

    # Extract CumulativeLoans
    cumulative_loans_test = df_test_with_cumloans['CumulativeLoans'].to_numpy()

    # Check for missing values
    n_missing = np.sum(np.isnan(cumulative_loans_test.astype(float)))
    if n_missing > 0:
        print(f"    ⚠ Warning: {n_missing} test samples have missing CumulativeLoans")

    print(f"\n  Final shapes:")
    print(f"    X_train: {X_train.shape}")
    print(f"    X_test: {X_test.shape}")
    print(f"    y_test: {y_test.shape}")
    print(f"    cumulative_loans_test: {cumulative_loans_test.shape}")

    return X_train, y_train, X_test, y_test, cumulative_loans_test


def train_xgboost(X_train, y_train, X_test, y_test, params):
    """Train XGBoost model."""
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

    # Update params with scale_pos_weight
    params_copy = params.copy()
    params_copy['scale_pos_weight'] = scale_pos_weight

    # Train model
    model = XGBClassifier(**params_copy)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50
    )

    print(f"✓ Training complete!")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best score: {model.best_score:.4f}")

    return model


def train_logistic_regression(X_train, y_train, X_test, y_test, params):
    """Train Logistic Regression model."""
    print(f"\n{'='*60}")
    print("Training Logistic Regression")
    print(f"{'='*60}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(**params)
    model.fit(X_train_scaled, y_train)

    print(f"✓ Training complete!")

    return model, scaler, X_train_scaled, X_test_scaled


def train_neural_network(X_train, y_train, X_test, y_test, params):
    """Train Neural Network model."""
    print(f"\n{'='*60}")
    print("Training Neural Network")
    print(f"{'='*60}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = MLPClassifier(**params)
    model.fit(X_train_scaled, y_train)

    print(f"✓ Training complete!")
    if hasattr(model, 'n_iter_'):
        print(f"  Training iterations: {model.n_iter_}")

    return model, scaler, X_train_scaled, X_test_scaled


def evaluate_by_loan_type(model, X_test, y_test, cumulative_loans_test,
                          model_name: str, X_test_scaled=None):
    """Evaluate model performance on first loans vs. repeat loans."""
    print(f"\n{'='*60}")
    print(f"{model_name} - Evaluation by Loan Type")
    print(f"{'='*60}")

    # Use scaled features if provided (for LR and NN)
    X_eval = X_test_scaled if X_test_scaled is not None else X_test

    # Get predictions for all test samples
    if hasattr(model, 'predict_proba'):
        y_proba_all = model.predict_proba(X_eval)[:, 1]
    else:
        y_proba_all = model.predict(X_eval)

    # Split by loan type
    first_loan_mask = cumulative_loans_test == 1
    repeat_loan_mask = cumulative_loans_test > 1

    # Calculate metrics for each group
    results = {}

    # All loans
    auc_all = roc_auc_score(y_test, y_proba_all)
    n_all = len(y_test)
    default_rate_all = np.mean(y_test)

    results['all'] = {
        'auc': auc_all,
        'n_samples': n_all,
        'default_rate': default_rate_all,
        'y_true': y_test,
        'y_proba': y_proba_all
    }

    # First loans
    if np.sum(first_loan_mask) > 0:
        y_test_first = y_test[first_loan_mask]
        y_proba_first = y_proba_all[first_loan_mask]

        if len(np.unique(y_test_first)) > 1:  # Need both classes for AUC
            auc_first = roc_auc_score(y_test_first, y_proba_first)
        else:
            auc_first = np.nan

        n_first = len(y_test_first)
        default_rate_first = np.mean(y_test_first)

        results['first'] = {
            'auc': auc_first,
            'n_samples': n_first,
            'default_rate': default_rate_first,
            'y_true': y_test_first,
            'y_proba': y_proba_first
        }

    # Repeat loans
    if np.sum(repeat_loan_mask) > 0:
        y_test_repeat = y_test[repeat_loan_mask]
        y_proba_repeat = y_proba_all[repeat_loan_mask]

        if len(np.unique(y_test_repeat)) > 1:  # Need both classes for AUC
            auc_repeat = roc_auc_score(y_test_repeat, y_proba_repeat)
        else:
            auc_repeat = np.nan

        n_repeat = len(y_test_repeat)
        default_rate_repeat = np.mean(y_test_repeat)

        results['repeat'] = {
            'auc': auc_repeat,
            'n_samples': n_repeat,
            'default_rate': default_rate_repeat,
            'y_true': y_test_repeat,
            'y_proba': y_proba_repeat
        }

    # Print results
    print(f"\n📊 Performance by Loan Type:")
    print(f"\n  All Loans:")
    print(f"    Samples: {results['all']['n_samples']:,}")
    print(f"    Default Rate: {results['all']['default_rate']:.2%}")
    print(f"    ROC AUC: {results['all']['auc']:.4f}")

    if 'first' in results:
        print(f"\n  First Loans (CumulativeLoans == 1):")
        print(f"    Samples: {results['first']['n_samples']:,}")
        print(f"    Default Rate: {results['first']['default_rate']:.2%}")
        print(f"    ROC AUC: {results['first']['auc']:.4f}")

    if 'repeat' in results:
        print(f"\n  Repeat Loans (CumulativeLoans > 1):")
        print(f"    Samples: {results['repeat']['n_samples']:,}")
        print(f"    Default Rate: {results['repeat']['default_rate']:.2%}")
        print(f"    ROC AUC: {results['repeat']['auc']:.4f}")

    if 'first' in results and 'repeat' in results:
        auc_diff = results['first']['auc'] - results['repeat']['auc']
        print(f"\n  AUC Difference (First - Repeat): {auc_diff:+.4f}")

    return results


def plot_comparison_charts(tfidf_results: dict, sms_risk_results: dict, output_dir: str):
    """Create comparison charts for first vs. repeat loans across both approaches."""
    print(f"\nCreating comparison charts...")

    # Prepare data for plotting
    approaches = ['TF-IDF', 'SMS Risk']
    loan_types = ['All', 'First', 'Repeat']

    # Extract AUC scores for each model
    models = ['XGBoost', 'Logistic Regression', 'Neural Network']

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance: First Loans vs. Repeat Loans', fontsize=16, fontweight='bold')

    # Plot 1: AUC comparison by loan type (XGBoost only)
    ax1 = axes[0, 0]
    model_name = 'XGBoost'

    tfidf_aucs = [
        tfidf_results[model_name]['all']['auc'],
        tfidf_results[model_name]['first']['auc'] if 'first' in tfidf_results[model_name] else np.nan,
        tfidf_results[model_name]['repeat']['auc'] if 'repeat' in tfidf_results[model_name] else np.nan
    ]
    sms_aucs = [
        sms_risk_results[model_name]['all']['auc'],
        sms_risk_results[model_name]['first']['auc'] if 'first' in sms_risk_results[model_name] else np.nan,
        sms_risk_results[model_name]['repeat']['auc'] if 'repeat' in sms_risk_results[model_name] else np.nan
    ]

    x = np.arange(len(loan_types))
    width = 0.35

    ax1.bar(x - width/2, tfidf_aucs, width, label='TF-IDF', alpha=0.8, color='steelblue')
    ax1.bar(x + width/2, sms_aucs, width, label='SMS Risk', alpha=0.8, color='coral')

    ax1.set_xlabel('Loan Type', fontsize=11)
    ax1.set_ylabel('ROC AUC', fontsize=11)
    ax1.set_title(f'{model_name} - AUC by Loan Type', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(loan_types)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0.5, 0.6])

    # Add value labels on bars
    for i, (tfidf_val, sms_val) in enumerate(zip(tfidf_aucs, sms_aucs)):
        if not np.isnan(tfidf_val):
            ax1.text(i - width/2, tfidf_val + 0.002, f'{tfidf_val:.4f}',
                    ha='center', va='bottom', fontsize=9)
        if not np.isnan(sms_val):
            ax1.text(i + width/2, sms_val + 0.002, f'{sms_val:.4f}',
                    ha='center', va='bottom', fontsize=9)

    # Plot 2: Sample sizes by loan type
    ax2 = axes[0, 1]

    tfidf_samples = [
        tfidf_results[model_name]['all']['n_samples'],
        tfidf_results[model_name]['first']['n_samples'] if 'first' in tfidf_results[model_name] else 0,
        tfidf_results[model_name]['repeat']['n_samples'] if 'repeat' in tfidf_results[model_name] else 0
    ]

    ax2.bar(x, tfidf_samples, alpha=0.8, color='steelblue')
    ax2.set_xlabel('Loan Type', fontsize=11)
    ax2.set_ylabel('Number of Samples', fontsize=11)
    ax2.set_title('Test Set Sample Sizes by Loan Type', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(loan_types)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, val in enumerate(tfidf_samples):
        ax2.text(i, val + max(tfidf_samples)*0.01, f'{val:,}',
                ha='center', va='bottom', fontsize=9)

    # Plot 3: Default rates by loan type
    ax3 = axes[1, 0]

    tfidf_default_rates = [
        tfidf_results[model_name]['all']['default_rate'] * 100,
        tfidf_results[model_name]['first']['default_rate'] * 100 if 'first' in tfidf_results[model_name] else 0,
        tfidf_results[model_name]['repeat']['default_rate'] * 100 if 'repeat' in tfidf_results[model_name] else 0
    ]

    ax3.bar(x, tfidf_default_rates, alpha=0.8, color='indianred')
    ax3.set_xlabel('Loan Type', fontsize=11)
    ax3.set_ylabel('Default Rate (%)', fontsize=11)
    ax3.set_title('Default Rates by Loan Type', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(loan_types)
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, val in enumerate(tfidf_default_rates):
        ax3.text(i, val + max(tfidf_default_rates)*0.01, f'{val:.2f}%',
                ha='center', va='bottom', fontsize=9)

    # Plot 4: Model comparison (all models, first loans only)
    ax4 = axes[1, 1]

    tfidf_first_aucs = []
    sms_first_aucs = []

    for model in models:
        tfidf_auc = tfidf_results[model]['first']['auc'] if 'first' in tfidf_results[model] else np.nan
        sms_auc = sms_risk_results[model]['first']['auc'] if 'first' in sms_risk_results[model] else np.nan
        tfidf_first_aucs.append(tfidf_auc)
        sms_first_aucs.append(sms_auc)

    x_models = np.arange(len(models))

    ax4.bar(x_models - width/2, tfidf_first_aucs, width, label='TF-IDF', alpha=0.8, color='steelblue')
    ax4.bar(x_models + width/2, sms_first_aucs, width, label='SMS Risk', alpha=0.8, color='coral')

    ax4.set_xlabel('Model', fontsize=11)
    ax4.set_ylabel('ROC AUC', fontsize=11)
    ax4.set_title('All Models - First Loans Performance', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_models)
    ax4.set_xticklabels(models, rotation=15, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim([0.5, 0.6])

    # Add value labels
    for i, (tfidf_val, sms_val) in enumerate(zip(tfidf_first_aucs, sms_first_aucs)):
        if not np.isnan(tfidf_val):
            ax4.text(i - width/2, tfidf_val + 0.002, f'{tfidf_val:.4f}',
                    ha='center', va='bottom', fontsize=9)
        if not np.isnan(sms_val):
            ax4.text(i + width/2, sms_val + 0.002, f'{sms_val:.4f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    output_file = os.path.join(output_dir, 'first_vs_repeat_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Comparison chart saved to: {output_file}")
    plt.close()


def save_detailed_results(tfidf_results: dict, sms_risk_results: dict, output_dir: str):
    """Save detailed results to text file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_file = os.path.join(output_dir, 'detailed_results.txt')

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FIRST LOANS VS. REPEAT LOANS - DETAILED ANALYSIS\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'='*80}\n\n")

        # TF-IDF Results
        f.write("="*80 + "\n")
        f.write("TF-IDF APPROACH RESULTS\n")
        f.write("="*80 + "\n\n")

        for model_name, results in tfidf_results.items():
            f.write(f"{model_name}\n")
            f.write("-" * 40 + "\n")

            for loan_type in ['all', 'first', 'repeat']:
                if loan_type in results:
                    r = results[loan_type]
                    f.write(f"\n  {loan_type.upper()} LOANS:\n")
                    f.write(f"    Samples: {r['n_samples']:,}\n")
                    f.write(f"    Default Rate: {r['default_rate']:.2%}\n")
                    f.write(f"    ROC AUC: {r['auc']:.4f}\n")

            if 'first' in results and 'repeat' in results:
                auc_diff = results['first']['auc'] - results['repeat']['auc']
                f.write(f"\n  AUC DIFFERENCE (First - Repeat): {auc_diff:+.4f}\n")

            f.write("\n")

        # SMS Risk Results
        f.write("\n" + "="*80 + "\n")
        f.write("SMS RISK FEATURES APPROACH RESULTS\n")
        f.write("="*80 + "\n\n")

        for model_name, results in sms_risk_results.items():
            f.write(f"{model_name}\n")
            f.write("-" * 40 + "\n")

            for loan_type in ['all', 'first', 'repeat']:
                if loan_type in results:
                    r = results[loan_type]
                    f.write(f"\n  {loan_type.upper()} LOANS:\n")
                    f.write(f"    Samples: {r['n_samples']:,}\n")
                    f.write(f"    Default Rate: {r['default_rate']:.2%}\n")
                    f.write(f"    ROC AUC: {r['auc']:.4f}\n")

            if 'first' in results and 'repeat' in results:
                auc_diff = results['first']['auc'] - results['repeat']['auc']
                f.write(f"\n  AUC DIFFERENCE (First - Repeat): {auc_diff:+.4f}\n")

            f.write("\n")

        # Comparison Summary
        f.write("\n" + "="*80 + "\n")
        f.write("COMPARISON SUMMARY - XGBOOST (BEST MODEL)\n")
        f.write("="*80 + "\n\n")

        model_name = 'XGBoost'

        f.write("TF-IDF vs. SMS Risk Features:\n")
        f.write("-" * 40 + "\n")

        for loan_type in ['all', 'first', 'repeat']:
            if loan_type in tfidf_results[model_name] and loan_type in sms_risk_results[model_name]:
                tfidf_auc = tfidf_results[model_name][loan_type]['auc']
                sms_auc = sms_risk_results[model_name][loan_type]['auc']
                diff = tfidf_auc - sms_auc

                f.write(f"\n{loan_type.upper()} LOANS:\n")
                f.write(f"  TF-IDF AUC:     {tfidf_auc:.4f}\n")
                f.write(f"  SMS Risk AUC:   {sms_auc:.4f}\n")
                f.write(f"  Difference:     {diff:+.4f} ({'TF-IDF better' if diff > 0 else 'SMS Risk better'})\n")

        f.write("\n" + "="*80 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("="*80 + "\n\n")

        # Calculate insights
        tfidf_first_auc = tfidf_results[model_name]['first']['auc'] if 'first' in tfidf_results[model_name] else np.nan
        tfidf_repeat_auc = tfidf_results[model_name]['repeat']['auc'] if 'repeat' in tfidf_results[model_name] else np.nan
        sms_first_auc = sms_risk_results[model_name]['first']['auc'] if 'first' in sms_risk_results[model_name] else np.nan
        sms_repeat_auc = sms_risk_results[model_name]['repeat']['auc'] if 'repeat' in sms_risk_results[model_name] else np.nan

        f.write("1. First Loans vs. Repeat Loans Performance:\n")
        if not np.isnan(tfidf_first_auc) and not np.isnan(tfidf_repeat_auc):
            if tfidf_first_auc > tfidf_repeat_auc:
                f.write(f"   - TF-IDF: First loans perform BETTER ({tfidf_first_auc:.4f} vs {tfidf_repeat_auc:.4f})\n")
            else:
                f.write(f"   - TF-IDF: Repeat loans perform BETTER ({tfidf_repeat_auc:.4f} vs {tfidf_first_auc:.4f})\n")

        if not np.isnan(sms_first_auc) and not np.isnan(sms_repeat_auc):
            if sms_first_auc > sms_repeat_auc:
                f.write(f"   - SMS Risk: First loans perform BETTER ({sms_first_auc:.4f} vs {sms_repeat_auc:.4f})\n")
            else:
                f.write(f"   - SMS Risk: Repeat loans perform BETTER ({sms_repeat_auc:.4f} vs {sms_first_auc:.4f})\n")

        f.write("\n2. Best Approach by Loan Type:\n")
        for loan_type in ['first', 'repeat']:
            if loan_type in tfidf_results[model_name] and loan_type in sms_risk_results[model_name]:
                tfidf_auc = tfidf_results[model_name][loan_type]['auc']
                sms_auc = sms_risk_results[model_name][loan_type]['auc']

                if tfidf_auc > sms_auc:
                    f.write(f"   - {loan_type.capitalize()} loans: TF-IDF is better ({tfidf_auc:.4f} vs {sms_auc:.4f})\n")
                else:
                    f.write(f"   - {loan_type.capitalize()} loans: SMS Risk is better ({sms_auc:.4f} vs {tfidf_auc:.4f})\n")

        f.write("\n")

    print(f"  ✓ Detailed results saved to: {output_file}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("FIRST LOANS VS. REPEAT LOANS - COMPREHENSIVE ANALYSIS")
    print("="*80)

    # Load spine data
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

    # Check CumulativeLoans distribution
    print(f"\n  CumulativeLoans distribution in test set:")
    cumulative_dist = df_test_spine['CumulativeLoans'].value_counts().sort('CumulativeLoans')
    for row in cumulative_dist.iter_rows():
        print(f"    CumulativeLoans={row[0]}: {row[1]:,} loans")

    # ========================================================================
    # TF-IDF Approach ONLY
    # ========================================================================
    print(f"\n{'='*80}")
    print("TF-IDF APPROACH")
    print(f"{'='*80}")

    # Load TF-IDF encoder and data
    print(f"\n2. Loading TF-IDF encoder and data...")
    encoder = load_encoder(TFIDF_ENCODER_CACHE)
    df_train_tfidf = load_data(TFIDF_TRAIN_DATA_CACHE)
    df_test_tfidf = load_data(TFIDF_TEST_DATA_CACHE)

    if encoder is None or df_train_tfidf is None or df_test_tfidf is None:
        print("  ⚠ TF-IDF cache files not found. Cannot proceed.")
        return None

    print(f"  ✓ Encoder loaded")
    print(f"  ✓ Train data: {len(df_train_tfidf):,} rows")
    print(f"  ✓ Test data: {len(df_test_tfidf):,} rows")

    # Prepare data
    X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf, cumulative_loans_test_tfidf = prepare_tfidf_data(
        encoder, df_train_tfidf, df_test_tfidf, df_test_spine
    )

    # Train and evaluate models
    tfidf_results = {}

    # XGBoost
    xgb_model = train_xgboost(X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf, XGBOOST_PARAMS)
    tfidf_results['XGBoost'] = evaluate_by_loan_type(
        xgb_model, X_test_tfidf, y_test_tfidf, cumulative_loans_test_tfidf, "XGBoost (TF-IDF)"
    )

    # Logistic Regression
    lr_model, lr_scaler, X_train_scaled, X_test_scaled = train_logistic_regression(
        X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf, LOGISTIC_PARAMS
    )
    tfidf_results['Logistic Regression'] = evaluate_by_loan_type(
        lr_model, X_test_tfidf, y_test_tfidf, cumulative_loans_test_tfidf,
        "Logistic Regression (TF-IDF)", X_test_scaled
    )

    # Neural Network
    nn_model, nn_scaler, X_train_scaled, X_test_scaled = train_neural_network(
        X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf, NEURALNET_PARAMS
    )
    tfidf_results['Neural Network'] = evaluate_by_loan_type(
        nn_model, X_test_tfidf, y_test_tfidf, cumulative_loans_test_tfidf,
        "Neural Network (TF-IDF)", X_test_scaled
    )

    # ========================================================================
    # Generate Reports
    # ========================================================================
    print(f"\n{'='*80}")
    print("GENERATING REPORTS")
    print(f"{'='*80}")

    # Save detailed results to text file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_file = os.path.join(OUTPUT_DIR, 'tfidf_first_vs_repeat_results.txt')

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TF-IDF APPROACH - FIRST LOANS VS. REPEAT LOANS ANALYSIS\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'='*80}\n\n")

        for model_name, results in tfidf_results.items():
            f.write(f"{model_name}\n")
            f.write("-" * 40 + "\n")

            for loan_type in ['all', 'first', 'repeat']:
                if loan_type in results:
                    r = results[loan_type]
                    f.write(f"\n  {loan_type.upper()} LOANS:\n")
                    f.write(f"    Samples: {r['n_samples']:,}\n")
                    f.write(f"    Default Rate: {r['default_rate']:.2%}\n")
                    f.write(f"    ROC AUC: {r['auc']:.4f}\n")

            if 'first' in results and 'repeat' in results:
                auc_diff = results['first']['auc'] - results['repeat']['auc']
                f.write(f"\n  AUC DIFFERENCE (First - Repeat): {auc_diff:+.4f}\n")

            f.write("\n")

    print(f"  ✓ Detailed results saved to: {output_file}")

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*80}")
    print("SUMMARY - TF-IDF APPROACH")
    print(f"{'='*80}")

    print(f"\n✓ Analysis complete!")
    print(f"✓ Results saved to: {OUTPUT_DIR}/")

    print(f"\n📊 Performance by Loan Type (XGBoost):")
    for loan_type in ['all', 'first', 'repeat']:
        if loan_type in tfidf_results['XGBoost']:
            r = tfidf_results['XGBoost'][loan_type]
            print(f"  {loan_type.capitalize():8s}: AUC={r['auc']:.4f}, N={r['n_samples']:,}, Default={r['default_rate']:.2%}")

    if 'first' in tfidf_results['XGBoost'] and 'repeat' in tfidf_results['XGBoost']:
        diff = tfidf_results['XGBoost']['first']['auc'] - tfidf_results['XGBoost']['repeat']['auc']
        print(f"\n  AUC Difference (First - Repeat): {diff:+.4f}")

    return tfidf_results


if __name__ == "__main__":
    tfidf_results = main()

