"""
Script to train SbertSupConEncoder on SMS data with labels.

This script:
1. Loads training/test spine data with labels
2. Queries SMS data in chunks with time-window aggregation
3. Joins SMS data with labels
4. Truncates text to fit within 4096 tokens (BGE-M3 with optimized speed)
5. Trains the SbertSupConEncoder model
"""

from helpers import TableManager
from data_helper import *
from transformer import SbertSupConEncoder
from transformers import AutoTokenizer
import polars as pl
import pickle
import os
from pathlib import Path


# ============================================================================
# Configuration
# ============================================================================

# Data configuration
CONTAINER_NAME = "mike"
DIRECTORY_PATH = "data-tmp/ensenada/raw"

# Date ranges
TRAIN_START_DATE = "2025-07-16"
TRAIN_END_DATE = "2025-09-01"
TEST_START_DATE = "2025-09-01"
TEST_END_DATE = "2025-09-15"

# SMS query configuration
DAYS_PRIOR = 14
SMS_SEPARATOR = '---'
MAX_TOKENS = 4096  # Reduced from 8192 to 4096 for faster training (mean was ~3800)
CHUNK_SIZE = 20000  # Number of UserLoanIds to process per chunk

# Model configuration
# Model options:
# - 'BAAI/bge-m3': 8192 tokens, 1024 dims, multilingual, state-of-the-art (~4-5 it/s)
# - 'jaimevera1107/all-MiniLM-L6-v2-similarity-es': 256 tokens, 384 dims, faster training (~6 it/s)
# - 'hiiamsid/sentence_similarity_spanish_es': 512 tokens, 768 dims, slower training (~2 it/s)
MODEL_NAME = 'BAAI/bge-m3'
# MODEL_NAME = 'jaimevera1107/all-MiniLM-L6-v2-similarity-es'  # Uncomment for smaller, faster model
# MODEL_NAME = 'hiiamsid/sentence_similarity_spanish_es'  # Uncomment for larger Spanish-specific model
EMBEDDING_DIM = 32  # Reduced to 32 for faster training and better generalization
N_EPOCHS = 1  # Keep at 1 for faster training
BATCH_SIZE = 8  # Increased to 8 with smaller tokens (4096) and embedding dims (32)
LEARNING_RATE = 1e-4  # Reduced from 2e-4 for larger model stability
GRADIENT_CHECKPOINTING = True  # Enable gradient checkpointing to save GPU memory
TEMPERATURE = 0.07
DEVICE = 'cuda'
RANDOM_STATE = 42

# Cache configuration
CACHE_DIR = "cache"
ENCODER_CACHE_FILE = "cache/sbert_encoder.pkl"
TRAIN_DATA_CACHE_FILE = "cache/train_data.parquet"
TEST_DATA_CACHE_FILE = "cache/test_data.parquet"
USE_CACHE = True  # Set to False to force retraining and reprocessing

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""

    # Check if cached data and encoder exist
    if USE_CACHE:
        print(f"Checking for cached files...")
        print(f"  Encoder: {ENCODER_CACHE_FILE}")
        print(f"  Train data: {TRAIN_DATA_CACHE_FILE}")
        print(f"  Test data: {TEST_DATA_CACHE_FILE}")

        encoder = load_encoder(ENCODER_CACHE_FILE)
        df_train_cached = load_data(TRAIN_DATA_CACHE_FILE)
        df_test_cached = load_data(TEST_DATA_CACHE_FILE)

        if encoder is not None and df_train_cached is not None and df_test_cached is not None:
            print("\n✓ Using cached encoder and data (set USE_CACHE=False to reprocess)")
            return encoder, df_train_cached, df_test_cached
        elif encoder is not None or df_train_cached is not None or df_test_cached is not None:
            print("\n⚠ Partial cache found, will regenerate all files for consistency")
        else:
            print("\n⚠ No cached files found, will process from scratch")
    else:
        print("\n⚠ Cache disabled (USE_CACHE=False), processing from scratch")

    # Initialize TableManager
    print("\nInitializing TableManager...")
    tm = TableManager(
        container_name=CONTAINER_NAME,
        directory_path=DIRECTORY_PATH,
    )

    # Load spine data
    print(f"\nLoading spine data...")
    df_train_spine, df_test_spine = query_spine(
        tm,
        train_start_date=TRAIN_START_DATE,
        train_end_date=TRAIN_END_DATE,
        test_start_date=TEST_START_DATE,
        test_end_date=TEST_END_DATE
    )
    print(f"  Train spine: {len(df_train_spine):,} rows")
    print(f"  Test spine: {len(df_test_spine):,} rows")

    # Check if processed data already exists
    print(f"\nChecking for cached processed data...")
    df_train_cached = load_data(TRAIN_DATA_CACHE_FILE)
    df_test_cached = load_data(TEST_DATA_CACHE_FILE)

    if df_train_cached is not None and df_test_cached is not None:
        print("\n✓ Using cached processed data (delete cache files to reprocess)")
        df_train = df_train_cached
        df_test = df_test_cached
    else:
        if df_train_cached is not None or df_test_cached is not None:
            print("\n⚠ Partial data cache found, will reprocess both datasets")
        else:
            print("\n⚠ No cached processed data found, will process from scratch")

        # Load tokenizer
        print(f"\nLoading tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Process training data
        print(f"\n{'='*60}")
        print("Processing Training Data")
        print(f"{'='*60}")
        df_train_labels = df_train_spine.select(['UserLoanId', 'CreatedAt', 'DefaultDPD21'])

        print(f"Processing {len(df_train_spine):,} training UserLoanIds in chunks...")
        print(f"  Lookback window: {DAYS_PRIOR} days")
        print(f"  Max tokens: {MAX_TOKENS}")

        train_chunks = []
        train_user_loan_ids = [str(uid) for uid in df_train_spine['UserLoanId']]

        for i, df_sms in enumerate(query_sms(
            tm,
            train_user_loan_ids,
            chunk_size=CHUNK_SIZE,
            days_prior=DAYS_PRIOR,
            separator=SMS_SEPARATOR
        ), start=1):
            df_chunk = process_sms_chunk(df_sms, df_train_labels, tokenizer, max_tokens=MAX_TOKENS)
            print_chunk_stats(i, df_sms, df_chunk)
            train_chunks.append(df_chunk)

        df_train = pl.concat(train_chunks)
        print_final_stats(df_train, "Training")

        # Process test data
        print(f"\n{'='*60}")
        print("Processing Test Data")
        print(f"{'='*60}")
        df_test_labels = df_test_spine.select(['UserLoanId', 'CreatedAt', 'DefaultDPD21'])

        print(f"Processing {len(df_test_spine):,} test UserLoanIds in chunks...")
        print(f"  Lookback window: {DAYS_PRIOR} days")
        print(f"  Max tokens: {MAX_TOKENS}")

        test_chunks = []
        test_user_loan_ids = [str(uid) for uid in df_test_spine['UserLoanId']]

        for i, df_sms in enumerate(query_sms(
            tm,
            test_user_loan_ids,
            chunk_size=CHUNK_SIZE,
            days_prior=DAYS_PRIOR,
            separator=SMS_SEPARATOR
        ), start=1):
            df_chunk = process_sms_chunk(df_sms, df_test_labels, tokenizer, max_tokens=MAX_TOKENS)
            print_chunk_stats(i, df_sms, df_chunk)
            test_chunks.append(df_chunk)

        df_test = pl.concat(test_chunks)
        print_final_stats(df_test, "Test")

        # Save processed data to cache
        print(f"\nSaving processed data to cache...")
        save_data(df_train, TRAIN_DATA_CACHE_FILE)
        save_data(df_test, TEST_DATA_CACHE_FILE)

    # Prepare data for training
    X_train = df_train['concatenated_body'].to_list()
    y_train = df_train['DefaultDPD21'].to_numpy()

    print(f"\n{'='*60}")
    print("Ready for Training")
    print(f"{'='*60}")
    print(f"X_train: {len(X_train):,} samples")
    print(f"y_train: {len(y_train):,} labels")
    print(f"Unique labels: {sorted(set(y_train))}")

    # Initialize encoder
    print(f"\nInitializing SbertSupConEncoder...")
    encoder = SbertSupConEncoder(
        base_model_name=MODEL_NAME,
        embedding_dim=EMBEDDING_DIM,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        temperature=TEMPERATURE,
        device=DEVICE,
        random_state=RANDOM_STATE,
        gradient_checkpointing=GRADIENT_CHECKPOINTING
    )

    # Train the model
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    encoder.fit(X_train, y_train)

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")

    # Save encoder and data to cache
    print(f"\nSaving to cache...")
    save_encoder(encoder, ENCODER_CACHE_FILE)

    return encoder, df_train, df_test


if __name__ == "__main__":
    encoder, df_train, df_test = main()










