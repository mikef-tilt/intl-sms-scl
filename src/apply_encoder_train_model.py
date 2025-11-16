"""
Script to apply the trained encoder and train an XGBoost model.

This script:
1. Loads the cached encoder and processed data
2. Applies the encoder to generate embeddings for train and test sets
3. Trains an XGBoost classifier on the embeddings
4. Evaluates the model using ROC AUC score
"""

from data_helper import load_encoder, load_data
import polars as pl
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# ============================================================================
# Configuration
# ============================================================================

# Cache file paths
ENCODER_CACHE_FILE = "cache/sbert_encoder.pkl"
TRAIN_DATA_CACHE_FILE = "cache/train_data.parquet"
TEST_DATA_CACHE_FILE = "cache/test_data.parquet"
TRAIN_EMBEDDINGS_CACHE_FILE = "cache/train_embeddings.npy"
TEST_EMBEDDINGS_CACHE_FILE = "cache/test_embeddings.npy"

# Embedding generation configuration
EMBEDDING_BATCH_SIZE = 256  # Batch size for encoding
EMBEDDING_CHUNK_SIZE = 10000  # Number of samples to process per chunk
USE_EMBEDDING_CACHE = True  # Whether to use cached embeddings

# Model selection - Choose which model to train
MODEL_TYPE = 'all'  # Options: 'xgboost', 'logistic', 'neuralnet', 'all'

# XGBoost configuration - Simple model with strong regularization
XGBOOST_PARAMS = {
    'n_estimators': 1000,  # More trees but with strong regularization
    'max_depth': 3,  # Very shallow trees to prevent overfitting (reduced from 8)
    'learning_rate': 0.01,  # Very slow learning rate (reduced from 0.05)
    'subsample': 0.5,  # Use only 50% of samples per tree (reduced from 0.8)
    'colsample_bytree': 0.5,  # Use only 50% of features per tree (reduced from 0.8)
    'min_child_weight': 10,  # Require more samples in leaf nodes (increased from 1)
    'gamma': 1.0,  # Higher minimum loss reduction (increased from 0.1)
    'reg_alpha': 1.0,  # Stronger L1 regularization (increased from 0.1)
    'reg_lambda': 10.0,  # Much stronger L2 regularization (increased from 1.0)
    'random_state': 42,
    'eval_metric': 'auc',
    'device': 'cuda:0',  # Use GPU for training (XGBoost 3.1+ uses 'device' instead of 'gpu_id')
    'scale_pos_weight': None,  # Will be calculated based on class imbalance
    'early_stopping_rounds': 100  # More patience for slower learning (increased from 50)
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
    Generate embeddings for a list of texts using the encoder in chunks.

    Args:
        encoder: Trained SbertSupConEncoder instance
        texts: List of text strings
        batch_size: Batch size for encoding (default: 256)
        chunk_size: Number of samples to process per chunk (default: 10000)

    Returns:
        Numpy array of embeddings
    """
    print(f"  Generating embeddings for {len(texts):,} samples in chunks of {chunk_size:,}...")

    all_embeddings = []
    num_chunks = (len(texts) + chunk_size - 1) // chunk_size

    for i in range(0, len(texts), chunk_size):
        chunk_texts = texts[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        print(f"    Processing chunk {chunk_num}/{num_chunks} ({len(chunk_texts):,} samples)...")

        # Generate embeddings for this chunk
        chunk_embeddings = encoder.transform(chunk_texts)
        all_embeddings.append(chunk_embeddings)

        print(f"    ✓ Chunk {chunk_num}/{num_chunks} complete - shape: {chunk_embeddings.shape}")

    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)
    print(f"  ✓ Total embeddings shape: {embeddings.shape}")
    return embeddings


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

    # Generate embeddings if not cached
    if X_train_embeddings is None:
        print(f"  Using batch_size={EMBEDDING_BATCH_SIZE}, chunk_size={EMBEDDING_CHUNK_SIZE}")
        print("  Training set:")
        X_train_embeddings = generate_embeddings(encoder, X_train_text,
                                                 batch_size=EMBEDDING_BATCH_SIZE,
                                                 chunk_size=EMBEDDING_CHUNK_SIZE)
        if USE_EMBEDDING_CACHE:
            save_embeddings(X_train_embeddings, TRAIN_EMBEDDINGS_CACHE_FILE)

    if X_test_embeddings is None:
        print(f"  Using batch_size={EMBEDDING_BATCH_SIZE}, chunk_size={EMBEDDING_CHUNK_SIZE}")
        print("  Test set:")
        X_test_embeddings = generate_embeddings(encoder, X_test_text,
                                               batch_size=EMBEDDING_BATCH_SIZE,
                                               chunk_size=EMBEDDING_CHUNK_SIZE)
        if USE_EMBEDDING_CACHE:
            save_embeddings(X_test_embeddings, TEST_EMBEDDINGS_CACHE_FILE)

    # Dictionary to store results
    results = {}

    # Train models based on MODEL_TYPE
    print(f"\n4. Training models (MODEL_TYPE={MODEL_TYPE})...")

    if MODEL_TYPE in ['xgboost', 'all']:
        print(f"\n{'='*60}")
        print("Training XGBoost")
        print(f"{'='*60}")
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

    # Plot ROC curves
    print(f"\n5. Plotting ROC curves...")
    plot_roc_curves(y_test, results)

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

