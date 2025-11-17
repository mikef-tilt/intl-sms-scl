"""
Script to apply the trained TF-IDF encoder and train classification models.

This script:
1. Loads the cached TF-IDF encoder and processed data
2. Applies the encoder to generate TF-IDF features for train and test sets
3. Trains classification models (XGBoost, Logistic Regression, Neural Network) on the features
4. Evaluates the models using ROC AUC score
"""

from data_helper import load_encoder, load_data
import polars as pl
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from datetime import datetime
import json


# ============================================================================
# Configuration
# ============================================================================

# Cache file paths
ENCODER_CACHE_FILE = "cache/tfidf_encoder.pkl"
TRAIN_DATA_CACHE_FILE = "cache/train_data.parquet"
TEST_DATA_CACHE_FILE = "cache/test_data.parquet"
TRAIN_EMBEDDINGS_CACHE_FILE = "cache/train_tfidf_features.npy"
TEST_EMBEDDINGS_CACHE_FILE = "cache/test_tfidf_features.npy"

# Feature generation configuration
USE_EMBEDDING_CACHE = True  # Whether to use cached TF-IDF features
CHUNK_SIZE = 50000  # Process in larger chunks for TF-IDF (no GPU memory constraints)

# Model selection - Choose which model to train
MODEL_TYPE = 'all'  # Options: 'xgboost', 'logistic', 'neuralnet', 'all'

# Grid Search Configuration
USE_GRID_SEARCH = True  # Whether to use grid search for XGBoost hyperparameter tuning
GRID_SEARCH_CV_FOLDS = 3  # Number of cross-validation folds for grid search
GRID_SEARCH_VERBOSE = 2  # Verbosity level for grid search (0=silent, 1=progress, 2=detailed)

# XGBoost Grid Search Parameter Grid (optimized for ~36 total fits with 3 CV folds = 12 combinations)
# Focus on more iterations since validation curve was still improving at 2000
XGBOOST_PARAM_GRID = {
    'n_estimators': [3000, 4000, 5000],  # Much more iterations (3 options)
    'max_depth': [4, 5],  # Deeper trees since 4 outperformed 3 (2 options)
    'learning_rate': [0.008, 0.01],  # Slower learning for more iterations (2 options)
    # Fixed parameters (not searched):
    # subsample: 0.5, colsample_bytree: 0.5, min_child_weight: 10
    # gamma: 1.0, reg_alpha: 1.0, reg_lambda: 10.0
}
# Total combinations: 3 * 2 * 2 = 12 combinations
# Total fits with 3-fold CV: 12 * 3 = 36 fits

# XGBoost Base Parameters (used if grid search is disabled or as base for grid search)
XGBOOST_BASE_PARAMS = {
    'subsample': 0.5,  # Use only 50% of samples per tree
    'colsample_bytree': 0.5,  # Use only 50% of features per tree
    'min_child_weight': 10,  # Require more samples in leaf nodes
    'gamma': 1.0,  # Higher minimum loss reduction
    'reg_alpha': 1.0,  # Stronger L1 regularization
    'reg_lambda': 10.0,  # Much stronger L2 regularization
    'random_state': 42,
    'eval_metric': 'auc',
    'device': 'cuda:0',  # Use GPU for training (XGBoost 3.1+ uses 'device' instead of 'gpu_id')
    'scale_pos_weight': None,  # Will be calculated based on class imbalance
    'early_stopping_rounds': 250  # More patience for slower learning with more iterations
}

# XGBoost Default Parameters (used when grid search is disabled)
XGBOOST_PARAMS = {
    'n_estimators': 2500,  # More iterations for better performance
    'max_depth': 3,  # Very shallow trees to prevent overfitting
    'learning_rate': 0.01,  # Slow learning rate
    **XGBOOST_BASE_PARAMS
}

# Logistic Regression configuration
LOGISTIC_PARAMS = {
    'penalty': 'l2',  # L2 regularization
    'C': 1.0,  # Inverse of regularization strength (smaller = stronger regularization)
    'solver': 'lbfgs',  # Good for small datasets
    'max_iter': 1000,
    'random_state': 42,
    'class_weight': 'balanced'  # Handle class imbalance
}

# Neural Network configuration - Simple 2-layer network
NEURALNET_PARAMS = {
    'hidden_layer_sizes': (64, 32),  # Two hidden layers: 64 and 32 neurons
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.01,  # L2 regularization
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

def save_embeddings(embeddings: np.ndarray, cache_file: str):
    """
    Save embeddings to a numpy file.

    Args:
        embeddings: Numpy array of embeddings
        cache_file: Path to cache file
    """
    import os
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.save(cache_file, embeddings)
    print(f"  ✓ Embeddings saved to cache: {cache_file}")


def load_embeddings(cache_file: str) -> np.ndarray:
    """
    Load embeddings from a numpy file.

    Args:
        cache_file: Path to cache file

    Returns:
        Numpy array of embeddings or None if file doesn't exist
    """
    import os
    if os.path.exists(cache_file):
        embeddings = np.load(cache_file)
        print(f"  ✓ Embeddings loaded from cache: {cache_file} - shape: {embeddings.shape}")
        return embeddings
    return None


def generate_embeddings(encoder, texts: list, batch_size: int = 256, chunk_size: int = 10000) -> np.ndarray:
    """
    Generate features for a list of texts using the encoder in chunks.

    Args:
        encoder: Trained encoder instance (TfidfSupDTEncoder or SbertSupConEncoder)
        texts: List of text strings
        batch_size: Batch size for encoding (default: 256, not used for TF-IDF)
        chunk_size: Number of samples to process per chunk (default: 10000)

    Returns:
        Numpy array of features
    """
    print(f"  Generating features for {len(texts):,} samples in chunks of {chunk_size:,}...")

    all_embeddings = []
    num_chunks = (len(texts) + chunk_size - 1) // chunk_size

    for i in range(0, len(texts), chunk_size):
        chunk_texts = texts[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        print(f"    Processing chunk {chunk_num}/{num_chunks} ({len(chunk_texts):,} samples)...")

        # Generate features for this chunk
        chunk_embeddings = encoder.transform(chunk_texts)
        all_embeddings.append(chunk_embeddings)

        print(f"    ✓ Chunk {chunk_num}/{num_chunks} complete - shape: {chunk_embeddings.shape}")

    # Concatenate all features
    embeddings = np.vstack(all_embeddings)
    print(f"  ✓ Total embeddings shape: {embeddings.shape}")
    return embeddings


def train_xgboost_with_grid_search(X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: np.ndarray, y_test: np.ndarray,
                                    param_grid: dict, base_params: dict) -> XGBClassifier:
    """
    Train an XGBoost classifier with grid search hyperparameter tuning.
    Uses manual cross-validation to avoid sklearn compatibility issues.

    Args:
        X_train: Training embeddings
        y_train: Training labels
        X_test: Test embeddings
        y_test: Test labels
        param_grid: Dictionary of parameters to search
        base_params: Base parameters for XGBoost

    Returns:
        Best trained XGBClassifier from grid search
    """
    from itertools import product

    print("\n" + "="*60)
    print("XGBoost Grid Search Hyperparameter Tuning")
    print("="*60)

    # Calculate scale_pos_weight to handle class imbalance
    n_negative = np.sum(y_train == 0)
    n_positive = np.sum(y_train == 1)
    scale_pos_weight = n_negative / n_positive

    print(f"  Class distribution:")
    print(f"    Negative (No Default): {n_negative:,} ({n_negative/len(y_train)*100:.2f}%)")
    print(f"    Positive (Default): {n_positive:,} ({n_positive/len(y_train)*100:.2f}%)")
    print(f"    Calculated scale_pos_weight: {scale_pos_weight:.4f}")

    # Update base params with calculated scale_pos_weight
    base_params_copy = base_params.copy()
    if base_params_copy.get('scale_pos_weight') is None:
        base_params_copy['scale_pos_weight'] = scale_pos_weight

    # Remove early_stopping_rounds from base params for grid search
    early_stopping_rounds = base_params_copy.pop('early_stopping_rounds', None)

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    print(f"\n  Grid Search Configuration:")
    print(f"    CV Folds: {GRID_SEARCH_CV_FOLDS}")
    print(f"    Total parameter combinations: {len(param_combinations)}")
    print(f"    Total fits: {len(param_combinations) * GRID_SEARCH_CV_FOLDS}")
    print(f"\n  Parameter grid:")
    for param, values in param_grid.items():
        print(f"    {param:20s}: {values}")

    # Setup cross-validation
    cv = StratifiedKFold(n_splits=GRID_SEARCH_CV_FOLDS, shuffle=True, random_state=42)

    print(f"\n  Starting grid search...")
    print(f"  This may take a while...\n")

    # Track best parameters
    best_score = -1
    best_params = None
    all_results = []

    # Iterate through all parameter combinations
    for idx, param_combo in enumerate(param_combinations, 1):
        # Create parameter dictionary
        params = dict(zip(param_names, param_combo))
        params.update(base_params_copy)

        # Cross-validation scores
        cv_scores = []

        # Perform cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_train_fold = X_train[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]

            # Train model
            model = XGBClassifier(**params)
            model.fit(X_train_fold, y_train_fold, verbose=0)

            # Evaluate on validation fold
            y_val_proba = model.predict_proba(X_val_fold)[:, 1]
            fold_score = roc_auc_score(y_val_fold, y_val_proba)
            cv_scores.append(fold_score)

        # Calculate mean CV score
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)

        all_results.append({
            'params': dict(zip(param_names, param_combo)),
            'mean_score': mean_cv_score,
            'std_score': std_cv_score
        })

        # Update best parameters
        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_params = dict(zip(param_names, param_combo))

        # Print progress
        if GRID_SEARCH_VERBOSE >= 1:
            print(f"  [{idx}/{len(param_combinations)}] CV AUC: {mean_cv_score:.4f} (+/- {std_cv_score:.4f}) - {dict(zip(param_names, param_combo))}")

    print(f"\n✓ Grid search complete!")
    print(f"\n  Best parameters:")
    for param, value in best_params.items():
        print(f"    {param:20s}: {value}")
    print(f"\n  Best CV ROC AUC: {best_score:.4f}")

    # Train final model with best parameters and early stopping
    print(f"\n  Training final model with best parameters...")

    final_params = best_params.copy()
    final_params.update(base_params_copy)

    if early_stopping_rounds is not None:
        final_params['early_stopping_rounds'] = early_stopping_rounds
        print(f"  Using early stopping with patience={early_stopping_rounds}")

    final_model = XGBClassifier(**final_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50
    )

    print(f"\n  ✓ Training complete!")
    if hasattr(final_model, 'best_iteration'):
        print(f"    Best iteration: {final_model.best_iteration}")
        print(f"    Best score: {final_model.best_score:.4f}")

    return final_model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  params: dict) -> XGBClassifier:
    """
    Train an XGBoost classifier with early stopping.

    Args:
        X_train: Training embeddings
        y_train: Training labels
        X_test: Test embeddings
        y_test: Test labels
        params: XGBoost parameters

    Returns:
        Trained XGBClassifier
    """
    print("\nTraining XGBoost classifier...")

    # Calculate scale_pos_weight to handle class imbalance
    # scale_pos_weight = count(negative class) / count(positive class)
    n_negative = np.sum(y_train == 0)
    n_positive = np.sum(y_train == 1)
    scale_pos_weight = n_negative / n_positive

    print(f"  Class distribution:")
    print(f"    Negative (No Default): {n_negative:,} ({n_negative/len(y_train)*100:.2f}%)")
    print(f"    Positive (Default): {n_positive:,} ({n_positive/len(y_train)*100:.2f}%)")
    print(f"    Calculated scale_pos_weight: {scale_pos_weight:.4f}")

    # Update params with calculated scale_pos_weight
    params_copy = params.copy()
    if params_copy.get('scale_pos_weight') is None:
        params_copy['scale_pos_weight'] = scale_pos_weight

    # Extract early_stopping_rounds if present (XGBoost 3.1+ uses callbacks)
    early_stopping_rounds = params_copy.pop('early_stopping_rounds', None)

    # Add early_stopping_rounds to model params if specified
    if early_stopping_rounds is not None:
        params_copy['early_stopping_rounds'] = early_stopping_rounds
        print(f"  Using early stopping with patience={early_stopping_rounds}")

    model = XGBClassifier(**params_copy)

    # Train with evaluation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50  # Print every 50 rounds instead of every round
    )

    print("✓ Training complete!")
    if early_stopping_rounds is not None and hasattr(model, 'best_iteration'):
        print(f"  Best iteration: {model.best_iteration}")
        print(f"  Best score: {model.best_score:.4f}")

    return model


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              params: dict):
    """
    Train a Logistic Regression classifier.

    Args:
        X_train: Training embeddings
        y_train: Training labels
        X_test: Test embeddings
        y_test: Test labels
        params: Logistic Regression parameters

    Returns:
        Trained LogisticRegression model and scaler
    """
    print("\nTraining Logistic Regression classifier...")

    # Standardize features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Calculate class distribution
    n_negative = np.sum(y_train == 0)
    n_positive = np.sum(y_train == 1)

    print(f"  Class distribution:")
    print(f"    Negative (No Default): {n_negative:,} ({n_negative/len(y_train)*100:.2f}%)")
    print(f"    Positive (Default): {n_positive:,} ({n_positive/len(y_train)*100:.2f}%)")
    print(f"  Using class_weight='balanced' to handle imbalance")

    model = LogisticRegression(**params)
    model.fit(X_train_scaled, y_train)

    print("✓ Training complete!")

    return model, scaler, X_train_scaled, X_test_scaled


def train_neural_network(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         params: dict):
    """
    Train a simple Neural Network classifier.

    Args:
        X_train: Training embeddings
        y_train: Training labels
        X_test: Test embeddings
        y_test: Test labels
        params: Neural Network parameters

    Returns:
        Trained MLPClassifier model and scaler
    """
    print("\nTraining Neural Network classifier...")

    # Standardize features (important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Calculate class distribution
    n_negative = np.sum(y_train == 0)
    n_positive = np.sum(y_train == 1)

    print(f"  Class distribution:")
    print(f"    Negative (No Default): {n_negative:,} ({n_negative/len(y_train)*100:.2f}%)")
    print(f"    Positive (Default): {n_positive:,} ({n_positive/len(y_train)*100:.2f}%)")
    print(f"  Network architecture: {params['hidden_layer_sizes']}")

    model = MLPClassifier(**params)
    model.fit(X_train_scaled, y_train)

    print("✓ Training complete!")
    if hasattr(model, 'best_loss_') and model.best_loss_ is not None:
        print(f"  Best validation loss: {model.best_loss_:.4f}")
    if hasattr(model, 'n_iter_'):
        print(f"  Training iterations: {model.n_iter_}")

    return model, scaler, X_train_scaled, X_test_scaled


def evaluate_model(model, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model"):
    """
    Evaluate the model focusing on ROC AUC.

    Args:
        model: Trained model
        X_train: Training embeddings (possibly scaled)
        y_train: Training labels
        X_test: Test embeddings (possibly scaled)
        y_test: Test labels
        model_name: Name of the model for display

    Returns:
        train_auc, test_auc, y_train_proba, y_test_proba
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation - ROC AUC")
    print(f"{'='*60}")

    # Prediction probabilities
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # ROC AUC scores
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"\n📊 ROC AUC Scores:")
    print(f"  Train ROC AUC: {train_auc:.4f}")
    print(f"  Test ROC AUC:  {test_auc:.4f}")
    print(f"  Difference (Overfitting): {train_auc - test_auc:.4f}")

    return train_auc, test_auc, y_train_proba, y_test_proba


def plot_roc_curves(y_test: np.ndarray, results: dict):
    """
    Plot ROC curves for all models.

    Args:
        y_test: True test labels
        results: Dictionary with model names as keys and (test_auc, y_test_proba) as values
    """
    plt.figure(figsize=(10, 8))

    colors = ['darkorange', 'green', 'purple', 'red']

    for i, (model_name, (test_auc, y_test_proba)) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f'{model_name} (AUC = {test_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison on SMS Embeddings', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('cache/roc_curve_comparison.png', dpi=150)
    print(f"\n✓ ROC curve comparison saved to: cache/roc_curve_comparison.png")
    plt.close()


def save_model_results(results: dict, train_results: dict, filename: str = "cache/model_results.txt"):
    """
    Save model parameters and performance metrics to a text file.

    Args:
        results: Dictionary with model names as keys and (test_auc, y_test_proba) as values
        train_results: Dictionary with model names as keys and train_auc as values
        filename: Path to save the results file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL TRAINING RESULTS - TF-IDF SMS CLASSIFICATION\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'='*80}\n\n")

        # XGBoost Configuration
        f.write("XGBOOST CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"  Grid Search Enabled     : {USE_GRID_SEARCH}\n")
        if USE_GRID_SEARCH:
            f.write(f"  CV Folds                : {GRID_SEARCH_CV_FOLDS}\n")
            f.write(f"\n  Parameter Grid:\n")
            for param, values in XGBOOST_PARAM_GRID.items():
                f.write(f"    {param:25s}: {values}\n")
            f.write(f"\n  Base Parameters:\n")
            for param, value in XGBOOST_BASE_PARAMS.items():
                f.write(f"    {param:25s}: {value}\n")
        else:
            f.write(f"\n  Parameters:\n")
            for param, value in XGBOOST_PARAMS.items():
                f.write(f"    {param:25s}: {value}\n")
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

        # Sort by test AUC (descending)
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
        f.write("- Overfitting = Train AUC - Test AUC\n")
        f.write("- Lower overfitting indicates better generalization\n")
        f.write("- TF-IDF features: 100 top features selected by Decision Tree\n")
        f.write("- Spanish text preprocessing: stopwords removed, lowercase, no punctuation\n")
        f.write("- Multiprocessing enabled for large datasets (20 parallel workers)\n")
        f.write("\n")

    print(f"\n✓ Model results saved to: {filename}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""

    print(f"{'='*60}")
    print("Apply Encoder & Train Models")
    print(f"{'='*60}")

    # Load cached encoder and data
    print("\n1. Loading cached encoder and data...")
    encoder = load_encoder(ENCODER_CACHE_FILE)
    df_train = load_data(TRAIN_DATA_CACHE_FILE)
    df_test = load_data(TEST_DATA_CACHE_FILE)

    if encoder is None or df_train is None or df_test is None:
        raise FileNotFoundError(
            "Cache files not found! Please run fit_encoder.py first to generate cached files."
        )

    # Prepare data
    print("\n2. Preparing data...")
    X_train_text = df_train['concatenated_body'].to_list()
    y_train = df_train['DefaultDPD21'].to_numpy()
    X_test_text = df_test['concatenated_body'].to_list()
    y_test = df_test['DefaultDPD21'].to_numpy()

    print(f"  Train samples: {len(X_train_text):,}")
    print(f"  Test samples: {len(X_test_text):,}")
    print(f"  Train label distribution: {np.bincount(y_train)}")
    print(f"  Test label distribution: {np.bincount(y_test)}")

    # Generate or load embeddings
    print(f"\n3. Generating/loading embeddings...")

    # Try to load cached embeddings
    X_train_embeddings = None
    X_test_embeddings = None

    if USE_EMBEDDING_CACHE:
        print("  Checking for cached embeddings...")
        X_train_embeddings = load_embeddings(TRAIN_EMBEDDINGS_CACHE_FILE)
        X_test_embeddings = load_embeddings(TEST_EMBEDDINGS_CACHE_FILE)

    # Generate features if not cached
    if X_train_embeddings is None:
        print("  Training set:")
        X_train_embeddings = generate_embeddings(encoder, X_train_text,
                                                 batch_size=256,
                                                 chunk_size=CHUNK_SIZE)
        if USE_EMBEDDING_CACHE:
            save_embeddings(X_train_embeddings, TRAIN_EMBEDDINGS_CACHE_FILE)

    if X_test_embeddings is None:
        print("  Test set:")
        X_test_embeddings = generate_embeddings(encoder, X_test_text,
                                               batch_size=256,
                                               chunk_size=CHUNK_SIZE)
        if USE_EMBEDDING_CACHE:
            save_embeddings(X_test_embeddings, TEST_EMBEDDINGS_CACHE_FILE)

    # Dictionary to store results
    results = {}
    train_results = {}  # Store train AUC scores

    # Train models based on MODEL_TYPE
    print(f"\n4. Training models (MODEL_TYPE={MODEL_TYPE})...")

    if MODEL_TYPE in ['xgboost', 'all']:
        print(f"\n{'='*60}")
        print("Training XGBoost")
        print(f"{'='*60}")

        if USE_GRID_SEARCH:
            print(f"  Grid search enabled: YES")
            xgb_model = train_xgboost_with_grid_search(
                X_train_embeddings, y_train,
                X_test_embeddings, y_test,
                XGBOOST_PARAM_GRID,
                XGBOOST_BASE_PARAMS
            )
        else:
            print(f"  Grid search enabled: NO")
            xgb_model = train_xgboost(
                X_train_embeddings, y_train,
                X_test_embeddings, y_test,
                XGBOOST_PARAMS
            )

        train_auc, test_auc, y_train_proba, y_test_proba = evaluate_model(
            xgb_model,
            X_train_embeddings, y_train,
            X_test_embeddings, y_test,
            model_name="XGBoost"
        )
        results['XGBoost'] = (test_auc, y_test_proba)
        train_results['XGBoost'] = train_auc

    if MODEL_TYPE in ['logistic', 'all']:
        print(f"\n{'='*60}")
        print("Training Logistic Regression")
        print(f"{'='*60}")
        lr_model, lr_scaler, X_train_scaled, X_test_scaled = train_logistic_regression(
            X_train_embeddings, y_train,
            X_test_embeddings, y_test,
            LOGISTIC_PARAMS
        )
        train_auc, test_auc, y_train_proba, y_test_proba = evaluate_model(
            lr_model,
            X_train_scaled, y_train,
            X_test_scaled, y_test,
            model_name="Logistic Regression"
        )
        results['Logistic Regression'] = (test_auc, y_test_proba)
        train_results['Logistic Regression'] = train_auc

    if MODEL_TYPE in ['neuralnet', 'all']:
        print(f"\n{'='*60}")
        print("Training Neural Network")
        print(f"{'='*60}")
        nn_model, nn_scaler, X_train_scaled, X_test_scaled = train_neural_network(
            X_train_embeddings, y_train,
            X_test_embeddings, y_test,
            NEURALNET_PARAMS
        )
        train_auc, test_auc, y_train_proba, y_test_proba = evaluate_model(
            nn_model,
            X_train_scaled, y_train,
            X_test_scaled, y_test,
            model_name="Neural Network"
        )
        results['Neural Network'] = (test_auc, y_test_proba)
        train_results['Neural Network'] = train_auc

    # Plot ROC curves
    print(f"\n5. Plotting ROC curves...")
    plot_roc_curves(y_test, results)

    # Save model results to file
    print(f"\n6. Saving model results...")
    save_model_results(results, train_results)

    # Summary
    print(f"\n{'='*60}")
    print("Summary - Model Comparison")
    print(f"{'='*60}")
    print(f"✓ Encoder applied successfully")
    print(f"✓ Models trained: {', '.join(results.keys())}")
    print(f"\n📊 Test ROC AUC Scores:")
    for model_name, (test_auc, _) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
        print(f"  {model_name:20s}: {test_auc:.4f}")
    print(f"\n✓ ROC curve comparison saved to cache/roc_curve_comparison.png")

    return results


if __name__ == "__main__":
    model, train_auc, test_auc = main()

