"""
Evaluate the trained encoder with XGBoost and Logistic Regression.
"""

import pickle
import numpy as np
import polars as pl
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb

# Configuration
ENCODER_PATH = "cache/sbert_encoder.pkl"
TRAIN_DATA_PATH = "cache/train_data.parquet"
TEST_DATA_PATH = "cache/test_data.parquet"
TRAIN_EMBEDDINGS_CACHE = "cache/train_embeddings.npy"
TEST_EMBEDDINGS_CACHE = "cache/test_embeddings.npy"

def main():
    print("\n" + "="*60)
    print("Evaluating Trained Encoder")
    print("="*60)
    
    # Load encoder
    print(f"\nLoading encoder from {ENCODER_PATH}...")
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    print(f"✓ Encoder loaded")
    print(f"  Device: {encoder.device_}")
    print(f"  Batch size: {encoder.batch_size}")
    
    # Load data
    print(f"\nLoading data...")
    train_df = pl.read_parquet(TRAIN_DATA_PATH)
    test_df = pl.read_parquet(TEST_DATA_PATH)
    
    X_train = train_df['concatenated_body'].to_numpy()
    y_train = train_df['DefaultDPD21'].to_numpy()
    X_test = test_df['concatenated_body'].to_numpy()
    y_test = test_df['DefaultDPD21'].to_numpy()
    
    print(f"✓ Train: {len(X_train):,} samples")
    print(f"✓ Test: {len(X_test):,} samples")
    
    # Generate or load embeddings
    if Path(TRAIN_EMBEDDINGS_CACHE).exists():
        print(f"\n✓ Loading cached train embeddings from {TRAIN_EMBEDDINGS_CACHE}")
        train_embeddings = np.load(TRAIN_EMBEDDINGS_CACHE)
    else:
        print(f"\nEncoding training data (using {encoder.device_})...")
        train_embeddings = encoder.transform(X_train, show_progress_bar=True)
        np.save(TRAIN_EMBEDDINGS_CACHE, train_embeddings)
        print(f"✓ Cached train embeddings to {TRAIN_EMBEDDINGS_CACHE}")
    
    if Path(TEST_EMBEDDINGS_CACHE).exists():
        print(f"✓ Loading cached test embeddings from {TEST_EMBEDDINGS_CACHE}")
        test_embeddings = np.load(TEST_EMBEDDINGS_CACHE)
    else:
        print(f"\nEncoding test data (using {encoder.device_})...")
        test_embeddings = encoder.transform(X_test, show_progress_bar=True)
        np.save(TEST_EMBEDDINGS_CACHE, test_embeddings)
        print(f"✓ Cached test embeddings to {TEST_EMBEDDINGS_CACHE}")
    
    print(f"\n✓ Train embeddings shape: {train_embeddings.shape}")
    print(f"✓ Test embeddings shape: {test_embeddings.shape}")
    
    # Train and evaluate Logistic Regression
    print("\n" + "="*60)
    print("Logistic Regression")
    print("="*60)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(train_embeddings, y_train)

    # Training set evaluation
    y_pred_lr_train = lr.predict(train_embeddings)
    y_pred_proba_lr_train = lr.predict_proba(train_embeddings)[:, 1]
    roc_auc_lr_train = roc_auc_score(y_train, y_pred_proba_lr_train)

    # Test set evaluation
    y_pred_lr = lr.predict(test_embeddings)
    y_pred_proba_lr = lr.predict_proba(test_embeddings)[:, 1]
    roc_auc_lr = roc_auc_score(y_test, y_pred_proba_lr)

    print(f"\n✓ Train ROC AUC: {roc_auc_lr_train:.4f}")
    print(f"✓ Test ROC AUC:  {roc_auc_lr:.4f}")
    print(f"✓ Difference (Overfitting): {roc_auc_lr_train - roc_auc_lr:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred_lr))
    print("\nTest Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr))
    
    # Train and evaluate XGBoost
    print("\n" + "="*60)
    print("XGBoost")
    print("="*60)

    # Calculate scale_pos_weight for class imbalance
    n_negative = np.sum(y_train == 0)
    n_positive = np.sum(y_train == 1)
    scale_pos_weight = n_negative / n_positive

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    xgb_model.fit(train_embeddings, y_train)

    # Training set evaluation
    y_pred_xgb_train = xgb_model.predict(train_embeddings)
    y_pred_proba_xgb_train = xgb_model.predict_proba(train_embeddings)[:, 1]
    roc_auc_xgb_train = roc_auc_score(y_train, y_pred_proba_xgb_train)

    # Test set evaluation
    y_pred_xgb = xgb_model.predict(test_embeddings)
    y_pred_proba_xgb = xgb_model.predict_proba(test_embeddings)[:, 1]
    roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

    print(f"\n✓ Train ROC AUC: {roc_auc_xgb_train:.4f}")
    print(f"✓ Test ROC AUC:  {roc_auc_xgb:.4f}")
    print(f"✓ Difference (Overfitting): {roc_auc_xgb_train - roc_auc_xgb:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred_xgb))
    print("\nTest Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_xgb))

    # Train and evaluate MLP (Neural Network)
    print("\n" + "="*60)
    print("MLP Neural Network")
    print("="*60)

    # Standardize features for neural network
    scaler = StandardScaler()
    train_embeddings_scaled = scaler.fit_transform(train_embeddings)
    test_embeddings_scaled = scaler.transform(test_embeddings)

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),  # Two hidden layers
        activation='relu',
        solver='adam',
        alpha=0.01,  # L2 regularization
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=True
    )

    print("\nTraining MLP...")
    mlp_model.fit(train_embeddings_scaled, y_train)

    # Training set evaluation
    y_pred_mlp_train = mlp_model.predict(train_embeddings_scaled)
    y_pred_proba_mlp_train = mlp_model.predict_proba(train_embeddings_scaled)[:, 1]
    roc_auc_mlp_train = roc_auc_score(y_train, y_pred_proba_mlp_train)

    # Test set evaluation
    y_pred_mlp = mlp_model.predict(test_embeddings_scaled)
    y_pred_proba_mlp = mlp_model.predict_proba(test_embeddings_scaled)[:, 1]
    roc_auc_mlp = roc_auc_score(y_test, y_pred_proba_mlp)

    print(f"\n✓ Train ROC AUC: {roc_auc_mlp_train:.4f}")
    print(f"✓ Test ROC AUC:  {roc_auc_mlp:.4f}")
    print(f"✓ Difference (Overfitting): {roc_auc_mlp_train - roc_auc_mlp:.4f}")
    if hasattr(mlp_model, 'best_loss_') and mlp_model.best_loss_ is not None:
        print(f"✓ Best validation loss: {mlp_model.best_loss_:.4f}")
    if hasattr(mlp_model, 'n_iter_'):
        print(f"✓ Training iterations: {mlp_model.n_iter_}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred_mlp))
    print("\nTest Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_mlp))

    # Summary
    print("\n" + "="*60)
    print("Summary - Train vs Test Performance")
    print("="*60)
    print(f"\n{'Model':<25} {'Train AUC':>12} {'Test AUC':>12} {'Overfit':>12}")
    print("-" * 60)
    print(f"{'Logistic Regression':<25} {roc_auc_lr_train:>12.4f} {roc_auc_lr:>12.4f} {roc_auc_lr_train - roc_auc_lr:>12.4f}")
    print(f"{'XGBoost':<25} {roc_auc_xgb_train:>12.4f} {roc_auc_xgb:>12.4f} {roc_auc_xgb_train - roc_auc_xgb:>12.4f}")
    print(f"{'MLP Neural Network':<25} {roc_auc_mlp_train:>12.4f} {roc_auc_mlp:>12.4f} {roc_auc_mlp_train - roc_auc_mlp:>12.4f}")
    print("="*60)

if __name__ == "__main__":
    main()

