"""
Evaluate a checkpoint on the test set using multiple classifiers.

Usage:
    python evaluate_checkpoint.py --checkpoint cache/checkpoints/checkpoint-400
"""

import argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from data_helper import load_encoder_from_checkpoint, load_data
import warnings
import hashlib
import pickle
warnings.filterwarnings('ignore')

def get_cache_key(checkpoint_path, data_type):
    """Generate a cache key based on checkpoint path and data type."""
    checkpoint_name = Path(checkpoint_path).name
    return f"cache/embeddings_{checkpoint_name}_{data_type}.npy"

def evaluate_checkpoint(checkpoint_path, embedding_dim=32, batch_size=8, use_cache=True):
    """
    Load a checkpoint and evaluate it on the test set.

    Args:
        checkpoint_path: Path to checkpoint directory
        embedding_dim: Embedding dimension used during training
        batch_size: Batch size for encoding
        use_cache: Whether to use cached embeddings
    """
    print(f"\n{'='*60}")
    print("Checkpoint Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load encoder from checkpoint
    print(f"\n{'='*60}")
    print("Loading Encoder from Checkpoint")
    print(f"{'='*60}")
    encoder = load_encoder_from_checkpoint(
        checkpoint_path=checkpoint_path,
        embedding_dim=embedding_dim,
        batch_size=batch_size
    )

    # Verify CUDA is being used
    print(f"✓ Encoder device: {encoder.device_}")
    print(f"✓ Batch size: {encoder.batch_size}")

    # Load data
    print(f"\n{'='*60}")
    print("Loading Data")
    print(f"{'='*60}")

    df_train = load_data('cache/train_data.parquet')
    df_test = load_data('cache/test_data.parquet')

    if df_train is None or df_test is None:
        print("❌ Could not load data")
        return

    X_train = df_train['concatenated_body'].to_list()
    y_train = df_train['DefaultDPD21'].to_numpy()
    X_test = df_test['concatenated_body'].to_list()
    y_test = df_test['DefaultDPD21'].to_numpy()

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Train label distribution: {np.bincount(y_train)}")
    print(f"Test label distribution: {np.bincount(y_test)}")

    # Generate embeddings with caching
    print(f"\n{'='*60}")
    print("Generating Embeddings")
    print(f"{'='*60}")

    train_cache_file = get_cache_key(checkpoint_path, 'train')
    test_cache_file = get_cache_key(checkpoint_path, 'test')

    # Train embeddings
    if use_cache and Path(train_cache_file).exists():
        print(f"✓ Loading cached train embeddings from {train_cache_file}")
        train_embeddings = np.load(train_cache_file)
        print(f"✓ Train embeddings shape: {train_embeddings.shape}")
    else:
        print(f"Encoding training data (using {encoder.device_})...")
        train_embeddings = encoder.model_.encode(
            X_train,
            show_progress_bar=True,
            device=encoder.device_,
            batch_size=encoder.batch_size,
            convert_to_numpy=True
        )
        print(f"✓ Train embeddings shape: {train_embeddings.shape}")
        # Cache the embeddings
        np.save(train_cache_file, train_embeddings)
        print(f"✓ Cached train embeddings to {train_cache_file}")

    # Test embeddings
    if use_cache and Path(test_cache_file).exists():
        print(f"✓ Loading cached test embeddings from {test_cache_file}")
        test_embeddings = np.load(test_cache_file)
        print(f"✓ Test embeddings shape: {test_embeddings.shape}")
    else:
        print(f"Encoding test data (using {encoder.device_})...")
        test_embeddings = encoder.model_.encode(
            X_test,
            show_progress_bar=True,
            device=encoder.device_,
            batch_size=encoder.batch_size,
            convert_to_numpy=True
        )
        print(f"✓ Test embeddings shape: {test_embeddings.shape}")
        # Cache the embeddings
        np.save(test_cache_file, test_embeddings)
        print(f"✓ Cached test embeddings to {test_cache_file}")
    
    # Train and evaluate classifiers
    print(f"\n{'='*60}")
    print("Training and Evaluating Classifiers")
    print(f"{'='*60}")
    
    results = {}
    
    # 1. Logistic Regression
    print("\n1. Logistic Regression")
    print("-" * 40)
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(train_embeddings, y_train)
    
    y_pred_proba = lr.predict_proba(test_embeddings)[:, 1]
    y_pred = lr.predict(test_embeddings)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    results['Logistic Regression'] = auc
    
    print(f"ROC AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 2. XGBoost
    print("\n2. XGBoost")
    print("-" * 40)
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc'
    )
    xgb.fit(train_embeddings, y_train)
    
    y_pred_proba = xgb.predict_proba(test_embeddings)[:, 1]
    y_pred = xgb.predict(test_embeddings)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    results['XGBoost'] = auc
    
    print(f"ROC AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print("\nROC AUC Scores:")
    for model_name, score in results.items():
        print(f"  {model_name:20s}: {score:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate a training checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint directory')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for encoding')
    parser.add_argument('--no-cache', action='store_true', help='Disable embedding caching')

    args = parser.parse_args()

    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print(f"\nAvailable checkpoints:")
        checkpoint_dir = Path("cache/checkpoints")
        if checkpoint_dir.exists():
            for cp in sorted(checkpoint_dir.glob("checkpoint-*")):
                print(f"  - {cp}")
        else:
            print(f"  No checkpoints found in {checkpoint_dir}")
        return

    evaluate_checkpoint(args.checkpoint, args.embedding_dim, args.batch_size, use_cache=not args.no_cache)

if __name__ == "__main__":
    main()

