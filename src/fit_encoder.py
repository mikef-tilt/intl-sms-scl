"""
Script to train TfidfSupDTEncoder on SMS data with labels.

This script:
1. Loads training/test spine data with labels
2. Queries SMS data in chunks with time-window aggregation
3. Joins SMS data with labels
4. Trains the TfidfSupDTEncoder model using supervised TF-IDF with Decision Tree feature selection
"""

from helpers import TableManager
from data_helper import *
from tfidf_transformer import TfidfSupDTEncoder
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
TRAIN_START_DATE = "2025-08-01"
TRAIN_END_DATE = "2025-09-01"
TEST_START_DATE = "2025-09-01"
TEST_END_DATE = "2025-09-15"

# SMS query configuration
DAYS_PRIOR = 28
SMS_SEPARATOR = '---'
CHUNK_SIZE = 20000  # Number of UserLoanIds to process per chunk

# TF-IDF Model configuration
N_FEATURES = 100  # Number of top features to select based on decision tree importance
MAX_FEATURES = 10000  # Maximum number of features for initial TF-IDF vectorization
MIN_DF = 2  # Minimum document frequency
MAX_DF = 0.95  # Maximum document frequency
NGRAM_RANGE = (1, 2)  # Unigrams and bigrams
DT_MAX_DEPTH = 10  # Maximum depth of decision tree for feature selection
DT_MIN_SAMPLES_SPLIT = 100  # Minimum samples to split in decision tree
N_JOBS = -1  # Number of parallel jobs for preprocessing (-1 = use all CPUs)
RANDOM_STATE = 42

# Cache configuration
CACHE_DIR = "cache"
ENCODER_CACHE_FILE = "cache/tfidf_encoder.pkl"
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

        # Process training data
        print(f"\n{'='*60}")
        print("Processing Training Data")
        print(f"{'='*60}")
        df_train_labels = df_train_spine.select(['UserLoanId', 'CreatedAt', 'DefaultDPD21'])

        print(f"Processing {len(df_train_spine):,} training UserLoanIds in chunks...")
        print(f"  Lookback window: {DAYS_PRIOR} days")

        train_chunks = []
        train_user_loan_ids = [str(uid) for uid in df_train_spine['UserLoanId']]

        for i, df_sms in enumerate(query_sms(
            tm,
            train_user_loan_ids,
            chunk_size=CHUNK_SIZE,
            days_prior=DAYS_PRIOR,
            separator=SMS_SEPARATOR
        ), start=1):
            # Simple join with labels (no tokenization needed for TF-IDF)
            df_chunk = df_sms.join(df_train_labels, on=['UserLoanId', 'CreatedAt'], how='inner')
            print(f"\nChunk {i}:")
            print(f"  SMS rows before join: {len(df_sms):,}")
            print(f"  Rows after join: {len(df_chunk):,}")
            train_chunks.append(df_chunk)

        df_train = pl.concat(train_chunks)
        print(f"\n{'='*60}")
        print(f"Training Data Summary")
        print(f"{'='*60}")
        print(f"  Total rows: {len(df_train):,}")
        print(f"  Label distribution: {df_train['DefaultDPD21'].value_counts().sort('DefaultDPD21')}")

        # Process test data
        print(f"\n{'='*60}")
        print("Processing Test Data")
        print(f"{'='*60}")
        df_test_labels = df_test_spine.select(['UserLoanId', 'CreatedAt', 'DefaultDPD21'])

        print(f"Processing {len(df_test_spine):,} test UserLoanIds in chunks...")
        print(f"  Lookback window: {DAYS_PRIOR} days")

        test_chunks = []
        test_user_loan_ids = [str(uid) for uid in df_test_spine['UserLoanId']]

        for i, df_sms in enumerate(query_sms(
            tm,
            test_user_loan_ids,
            chunk_size=CHUNK_SIZE,
            days_prior=DAYS_PRIOR,
            separator=SMS_SEPARATOR
        ), start=1):
            # Simple join with labels (no tokenization needed for TF-IDF)
            df_chunk = df_sms.join(df_test_labels, on=['UserLoanId', 'CreatedAt'], how='inner')
            print(f"\nChunk {i}:")
            print(f"  SMS rows before join: {len(df_sms):,}")
            print(f"  Rows after join: {len(df_chunk):,}")
            test_chunks.append(df_chunk)

        df_test = pl.concat(test_chunks)
        print(f"\n{'='*60}")
        print(f"Test Data Summary")
        print(f"{'='*60}")
        print(f"  Total rows: {len(df_test):,}")
        print(f"  Label distribution: {df_test['DefaultDPD21'].value_counts().sort('DefaultDPD21')}")

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

    # Initialize TF-IDF encoder with Decision Tree feature selection
    print(f"\nInitializing TfidfSupDTEncoder...")
    print(f"  Top features to select: {N_FEATURES}")
    print(f"  Max features for initial TF-IDF: {MAX_FEATURES}")
    print(f"  N-gram range: {NGRAM_RANGE}")
    print(f"  Decision Tree max depth: {DT_MAX_DEPTH}")
    print(f"  Parallel jobs: {N_JOBS} ({'all CPUs' if N_JOBS == -1 else f'{N_JOBS} CPUs'})")

    encoder = TfidfSupDTEncoder(
        n_features=N_FEATURES,
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE,
        dt_max_depth=DT_MAX_DEPTH,
        dt_min_samples_split=DT_MIN_SAMPLES_SPLIT,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE
    )

    # Train the model
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    encoder.fit(X_train, y_train)

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")

    # Display top features
    print(f"\nTop 20 Most Important Features:")
    feature_importances = encoder.get_feature_importances()
    for i, (feature, importance) in enumerate(list(feature_importances.items())[:20], 1):
        print(f"  {i:2d}. {feature:30s} - {importance:.6f}")

    # Save encoder and data to cache
    print(f"\nSaving to cache...")
    save_encoder(encoder, ENCODER_CACHE_FILE)

    return encoder, df_train, df_test


if __name__ == "__main__":
    encoder, df_train, df_test = main()










