from helpers import TableManager
import polars as pl
from datetime import datetime
import random
import pickle
import os
from pathlib import Path



def query_spine(
        tm: TableManager,
        train_start_date: str,
        train_end_date: str,
        test_start_date: str,
        test_end_date: str
    ) -> pl.DataFrame:
    df_spine = tm.read_table('label_spine')

    # Convert date strings (format: 'YYYY-MM-DD') to datetime objects
    train_start = datetime.strptime(train_start_date, '%Y-%m-%d')
    train_end = datetime.strptime(train_end_date, '%Y-%m-%d')
    test_start = datetime.strptime(test_start_date, '%Y-%m-%d')
    test_end = datetime.strptime(test_end_date, '%Y-%m-%d')

    df_train_spine = df_spine.filter(
        pl.col('CreatedAt') >= train_start,
        pl.col('CreatedAt') < train_end
    )
    df_test_spine = df_spine.filter(
        pl.col('CreatedAt') >= test_start,
        pl.col('CreatedAt') < test_end
    )

    return df_train_spine, df_test_spine
    


def query_sms(
    tm: TableManager,
    user_loan_ids: list[str],
    chunk_size: int = 100000,
    randomize: bool = True,
    random_seed: int = None,
    days_prior: int = None,
    separator: str = "\n---\n"
    ):
    """
    Query SMS data in chunks for a list of UserLoanIds.

    Args:
        tm: TableManager instance
        user_loan_ids: List of UserLoanId strings to query
        chunk_size: Number of IDs to query per chunk (default: 10000)
        randomize: Whether to randomize the order of chunks (default: True)
        random_seed: Optional random seed for reproducibility (default: None)
        days_prior: If specified, aggregates messages by UserLoanId with a lookback window.
                   Returns concatenated unique messages within the time window. (default: None)
        separator: String separator for concatenated messages when days_prior is set (default: "\n---\n")

    Yields:
        DataFrame containing SMS data for each chunk of UserLoanIds.
        If days_prior is None: returns raw SMS data
        If days_prior is set: returns aggregated data with columns [UserLoanId, CreatedAt, concatenated_body]
    """
    # Create a copy to avoid modifying the original list
    ids_copy = user_loan_ids.copy()

    # Randomize the IDs if requested
    if randomize:
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(ids_copy)

    # Create chunks
    user_loan_ids_chunks = [ids_copy[i:i + chunk_size] for i in range(0, len(ids_copy), chunk_size)]

    # Query each chunk
    for chunk in user_loan_ids_chunks:
        if days_prior is None:
            # Simple query without aggregation
            query = f"SELECT * FROM <table> WHERE UserLoanId IN ({', '.join(chunk)})"
        else:
            # Aggregated query with time window and deduplication
            # Escape single quotes in separator for SQL
            sql_separator = separator.replace("'", "''")

            query = f"""
                WITH sms_data AS (
                    SELECT DISTINCT
                        UserLoanId,
                        CreatedAt,
                        body
                    FROM <table>
                    WHERE UserLoanId IN ({', '.join(chunk)})
                        AND body IS NOT NULL
                        AND TRIM(body) != ''
                ),
                sms_with_window AS (
                    SELECT
                        s1.UserLoanId,
                        s1.CreatedAt,
                        s2.body
                    FROM sms_data s1
                    JOIN sms_data s2
                        ON s1.UserLoanId = s2.UserLoanId
                        AND s2.CreatedAt <= s1.CreatedAt
                        AND s2.CreatedAt >= s1.CreatedAt - INTERVAL '{days_prior} days'
                )
                SELECT
                    UserLoanId,
                    CreatedAt,
                    STRING_AGG(DISTINCT body, '{sql_separator}' ORDER BY body) as concatenated_body
                FROM sms_with_window
                GROUP BY UserLoanId, CreatedAt
                ORDER BY UserLoanId, CreatedAt
            """

        df_sms = tm.query_table(table_name="android_sms", query=query)
        yield df_sms



def truncate_to_max_tokens(text: str, tokenizer, max_tokens: int = 512) -> str:
    """
    Truncate text to fit within max_tokens.

    Args:
        text: Text to truncate
        tokenizer: HuggingFace tokenizer instance
        max_tokens: Maximum number of tokens (default: 512)

    Returns:
        Truncated text
    """
    if text is None or text == '':
        return ''

    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # If already within limit, return original
    if len(tokens) <= max_tokens:
        return text

    # Truncate tokens and decode back to text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using the provided tokenizer."""
    if text is None or text == '':
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def process_sms_chunk(df_sms: pl.DataFrame, df_labels: pl.DataFrame, tokenizer, max_tokens: int = 512) -> pl.DataFrame:
    """
    Process a single SMS chunk: join with labels, truncate, and add token counts.

    Args:
        df_sms: SMS data chunk
        df_labels: Labels DataFrame
        tokenizer: Tokenizer for counting tokens
        max_tokens: Maximum tokens per text (default: 512)

    Returns:
        Processed DataFrame with labels and truncated text
    """
    # Join with labels
    df_chunk = df_sms.join(df_labels, on=['UserLoanId', 'CreatedAt'], how='inner')

    # Truncate concatenated_body to max tokens
    df_chunk = df_chunk.with_columns(
        pl.col('concatenated_body')
          .map_elements(lambda x: truncate_to_max_tokens(x, tokenizer, max_tokens=max_tokens), return_dtype=pl.Utf8)
          .alias('concatenated_body')
    )

    # Add token count for verification
    df_chunk = df_chunk.with_columns(
        pl.col('concatenated_body')
          .map_elements(lambda x: count_tokens(x, tokenizer), return_dtype=pl.Int64)
          .alias('token_count')
    )

    return df_chunk


def print_chunk_stats(chunk_num: int, df_sms: pl.DataFrame, df_chunk: pl.DataFrame):
    """Print statistics for a processed chunk."""
    print(f"\nChunk {chunk_num}:")
    print(f"  SMS rows before join: {len(df_sms):,}")
    print(f"  Rows after join: {len(df_chunk):,}")
    print(f"  Token count stats: min={df_chunk['token_count'].min()}, "
          f"max={df_chunk['token_count'].max()}, "
          f"mean={df_chunk['token_count'].mean():.1f}")


def print_final_stats(df: pl.DataFrame, dataset_name: str = "Training"):
    """Print final data statistics."""
    print(f"\nFinal {dataset_name} Data Summary:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Columns: {df.columns}")
    if 'DefaultDPD21' in df.columns:
        print(f"  Label distribution:")
        for row in df['DefaultDPD21'].value_counts().sort('DefaultDPD21').iter_rows():
            print(f"    {row[0]}: {row[1]:,}")


def save_encoder(encoder, cache_file: str):
    """
    Save the trained encoder to a cache file.

    Args:
        encoder: Trained SbertSupConEncoder instance
        cache_file: Path to cache file
    """
    # Create cache directory if it doesn't exist
    cache_dir = os.path.dirname(cache_file)
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Save encoder
    with open(cache_file, 'wb') as f:
        pickle.dump(encoder, f)

    print(f"\n✓ Encoder saved to: {cache_file}")


def load_encoder_from_checkpoint(checkpoint_path: str, embedding_dim: int = 32, batch_size: int = 8):
    """
    Load a trained encoder from a HuggingFace checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory (e.g., 'cache/checkpoints/checkpoint-3114')
        embedding_dim: Embedding dimension used during training
        batch_size: Batch size for encoding

    Returns:
        SbertSupConEncoder instance loaded from checkpoint
    """
    from transformer import SbertSupConEncoder
    from sentence_transformers import SentenceTransformer
    import torch

    print(f"\nLoading encoder from checkpoint: {checkpoint_path}")

    # Load the SentenceTransformer model from checkpoint
    model = SentenceTransformer(checkpoint_path)

    # Create encoder wrapper
    encoder = SbertSupConEncoder(
        base_model_name=checkpoint_path,  # Will be ignored since we set model directly
        embedding_dim=embedding_dim,
        batch_size=batch_size
    )

    # Set the loaded model and mark as fitted
    encoder.model_ = model
    encoder.is_fitted_ = True
    encoder.device_ = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.model_.to(encoder.device_)

    print(f"✓ Encoder loaded from checkpoint")
    print(f"  Device: {encoder.device_}")

    return encoder


def load_encoder(cache_file: str):
    """
    Load a trained encoder from cache file.

    Args:
        cache_file: Path to cache file

    Returns:
        Loaded SbertSupConEncoder instance or None if file doesn't exist
    """
    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, 'rb') as f:
            encoder = pickle.load(f)
        print(f"  ✓ Encoder loaded from cache: {cache_file}")
        return encoder
    except Exception as e:
        print(f"  ⚠ Warning: Failed to load encoder from cache: {e}")
        return None


def save_data(df: pl.DataFrame, cache_file: str):
    """
    Save a Polars DataFrame to a parquet file.

    Args:
        df: Polars DataFrame to save
        cache_file: Path to cache file
    """
    # Create cache directory if it doesn't exist
    cache_dir = os.path.dirname(cache_file)
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Save DataFrame as parquet
    df.write_parquet(cache_file)

    print(f"  ✓ Data saved to: {cache_file} ({len(df):,} rows)")


def load_data(cache_file: str) -> pl.DataFrame:
    """
    Load a Polars DataFrame from a parquet file.

    Args:
        cache_file: Path to cache file

    Returns:
        Loaded Polars DataFrame or None if file doesn't exist
    """
    if not os.path.exists(cache_file):
        return None

    try:
        df = pl.read_parquet(cache_file)
        print(f"  ✓ Data loaded from cache: {cache_file} ({len(df):,} rows)")
        return df
    except Exception as e:
        print(f"  ⚠ Warning: Failed to load data from cache: {e}")
        return None

