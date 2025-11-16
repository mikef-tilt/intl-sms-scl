# International SMS Supervised Contrastive Learning (intl-sms-scl)

A machine learning pipeline for training supervised contrastive learning models on SMS data to predict loan defaults. This project uses sentence transformers with supervised contrastive loss to create embeddings from SMS messages, which are then used to train classification models.

## Overview

This project implements a two-stage machine learning pipeline:

1. **Stage 1: Encoder Training** - Fine-tunes a sentence transformer model using supervised contrastive learning on SMS data
2. **Stage 2: Classifier Training** - Trains classification models (XGBoost, Logistic Regression, Neural Network) on the learned embeddings

The pipeline is designed to predict loan defaults (DefaultDPD21) based on SMS message patterns from borrowers.

## Project Structure

```
intl-sms-scl/
├── src/
│   ├── __init__.py
│   ├── fit_encoder.py              # Stage 1: Train the supervised contrastive encoder
│   ├── apply_encoder_train_model.py # Stage 2: Apply encoder and train classifiers
│   ├── transformer.py              # SbertSupConEncoder implementation
│   ├── data_helper.py              # Data loading and processing utilities
│   └── helpers.py                  # Azure Data Lake and database connection utilities
├── requirements.txt                # Python dependencies
└── README.MD                       # This file
```

## Features

### Encoder Training (`fit_encoder.py`)
- Loads SMS data from Azure Data Lake Storage
- Aggregates SMS messages with configurable time windows (default: 14 days prior)
- Truncates text to fit within token limits (4096 tokens for BGE-M3)
- Trains SentenceTransformer with supervised contrastive loss
- Supports multiple base models:
  - **BAAI/bge-m3** (default): 8192 tokens, 1024 dims, multilingual
  - **jaimevera1107/all-MiniLM-L6-v2-similarity-es**: 256 tokens, 384 dims, faster
  - **hiiamsid/sentence_similarity_spanish_es**: 512 tokens, 768 dims, Spanish-specific
- Caches processed data and trained encoder for reuse

### Classifier Training (`apply_encoder_train_model.py`)
- Loads cached encoder and generates embeddings
- Supports multiple classification models:
  - **XGBoost**: Gradient boosting with GPU acceleration
  - **Logistic Regression**: Linear baseline with L2 regularization
  - **Neural Network**: Multi-layer perceptron (64→32 neurons)
- Handles class imbalance with appropriate weighting
- Evaluates models using ROC AUC scores
- Generates comparative ROC curve visualizations

### Data Management (`data_helper.py`)
- Azure Data Lake integration for reading parquet files
- Time-window based SMS aggregation
- Token-aware text truncation
- Efficient caching system for processed data

### Infrastructure (`helpers.py`)
- Azure Data Lake Service Client integration
- SQL Server connection utilities (Azure SQL, Synapse, Fabric)
- DuckDB-based querying for efficient parquet processing
- Support for both incremental and full table types

## Requirements

- Python 3.8+
- CUDA 13.0 (for GPU acceleration)
- Azure credentials for Data Lake access
- See `requirements.txt` for complete dependency list

Key dependencies:
- `torch==2.5.1` - PyTorch for deep learning
- `sentence-transformers` - For transformer-based embeddings
- `xgboost==2.1.3` - Gradient boosting classifier
- `lightgbm==4.3.0` - Alternative gradient boosting
- `polars==1.4.1` - Fast DataFrame operations
- `duckdb==1.0.0` - SQL queries on parquet files
- `azure-storage-file-datalake==12.22.0` - Azure Data Lake integration
- `scikit-learn==1.6.1` - Machine learning utilities

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd intl-sms-scl

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file with your Azure credentials:

```bash
DATALAKE_ACCOUNT=<your-datalake-account-name>
DATALAKE_KEY=<your-datalake-access-key>
```

### Training Configuration

Edit the configuration variables in `src/fit_encoder.py`:

```python
# Data configuration
CONTAINER_NAME = "mike"
DIRECTORY_PATH = "data-tmp/ensenada/raw"

# Date ranges
TRAIN_START_DATE = "2025-07-16"
TRAIN_END_DATE = "2025-09-01"
TEST_START_DATE = "2025-09-01"
TEST_END_DATE = "2025-09-15"

# SMS query configuration
DAYS_PRIOR = 14  # Lookback window for SMS aggregation
MAX_TOKENS = 4096  # Maximum tokens per sample

# Model configuration
MODEL_NAME = 'BAAI/bge-m3'
EMBEDDING_DIM = 32
N_EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
```

## Usage

### Stage 1: Train the Encoder

```bash
python src/fit_encoder.py
```

This will:
1. Load spine data (labels) from Azure Data Lake
2. Query and aggregate SMS messages with time windows
3. Process and truncate text to fit token limits
4. Train the supervised contrastive encoder
5. Cache the encoder and processed data to `cache/`

### Stage 2: Train Classifiers

```bash
python src/apply_encoder_train_model.py
```

This will:
1. Load the cached encoder and data
2. Generate embeddings for train/test sets
3. Train selected models (XGBoost, Logistic Regression, Neural Network)
4. Evaluate models using ROC AUC
5. Generate ROC curve comparison plot

### Model Selection

Edit `MODEL_TYPE` in `apply_encoder_train_model.py`:

```python
MODEL_TYPE = 'all'  # Options: 'xgboost', 'logistic', 'neuralnet', 'all'
```

## Output

The pipeline generates the following cached files:

```
cache/
├── sbert_encoder.pkl           # Trained encoder
├── train_data.parquet          # Processed training data
├── test_data.parquet           # Processed test data
├── train_embeddings.npy        # Training embeddings
├── test_embeddings.npy         # Test embeddings
└── roc_curve_comparison.png    # ROC curve visualization
```

## Model Architecture

### SbertSupConEncoder

The encoder uses a 4-layer architecture:

1. **Transformer Layer**: Pre-trained sentence transformer (e.g., BGE-M3)
2. **Pooling Layer**: Mean pooling over token embeddings
3. **Projection Layer**: Dense layer projecting to `embedding_dim` (default: 32)
4. **Normalization Layer**: L2 normalization for contrastive learning

Training uses:
- **Loss**: BatchHardSoftMarginTripletLoss (supervised contrastive)
- **Optimizer**: AdamW with weight decay (0.02)
- **Scheduler**: Cosine annealing with warmup (10% of first epoch)
- **Mixed Precision**: Enabled for faster training
- **Gradient Checkpointing**: Optional for memory efficiency

## Performance Considerations

- **GPU Memory**: Gradient checkpointing reduces memory usage
- **Batch Processing**: Embeddings generated in chunks (default: 10,000 samples)
- **Caching**: Processed data and embeddings cached to avoid recomputation
- **Token Limits**: Text truncated to 4096 tokens (mean ~3800) for optimal speed

## Hardware Requirements

- **GPU**: CUDA-capable GPU recommended (CUDA 13.0 used for development)
- **Memory**: 16GB+ RAM recommended for large datasets
- **Storage**: Sufficient space for cached embeddings and models

## License

[Add your license information here]

## Contributors

[Add contributor information here]

## Notes

- CUDA 13.0 was used to train this model
- The pipeline supports both CPU and GPU training (GPU strongly recommended)
- Azure Data Lake credentials required for data access
- Supports incremental and full table types in Azure Data Lake