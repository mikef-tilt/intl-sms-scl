"""
Utility script to load and use a checkpoint for encoding.

Usage:
    python use_checkpoint.py --checkpoint cache/checkpoints/checkpoint-3114
"""

import argparse
from pathlib import Path
from data_helper import load_encoder_from_checkpoint, load_data
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Use a training checkpoint for encoding')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint directory')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for encoding')
    parser.add_argument('--test_data', type=str, default='cache/test_data.parquet', help='Test data file')
    parser.add_argument('--output', type=str, default='cache/checkpoint_embeddings.npy', help='Output file for embeddings')
    
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
    
    # Load encoder from checkpoint
    print(f"\n{'='*60}")
    print("Loading Encoder from Checkpoint")
    print(f"{'='*60}")
    encoder = load_encoder_from_checkpoint(
        checkpoint_path=args.checkpoint,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size
    )
    
    # Load test data
    print(f"\n{'='*60}")
    print("Loading Test Data")
    print(f"{'='*60}")
    df_test = load_data(args.test_data)
    if df_test is None:
        print(f"❌ Could not load test data from {args.test_data}")
        return
    
    X_test = df_test['concatenated_body'].to_list()
    print(f"Test samples: {len(X_test)}")
    
    # Generate embeddings
    print(f"\n{'='*60}")
    print("Generating Embeddings")
    print(f"{'='*60}")
    embeddings = encoder.transform(X_test)
    
    # Save embeddings
    np.save(args.output, embeddings)
    print(f"\n✓ Embeddings saved to: {args.output}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    
    print(f"\n{'='*60}")
    print("Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

