# TF-IDF Supervised Feature Selection Methodology

## Overview

This document describes the new supervised TF-IDF approach implemented as an alternative to the embedding-based methodology. The new approach uses TF-IDF vectorization with Decision Tree-based feature selection to identify the most important tokens for predicting loan defaults from Spanish SMS messages.

## Key Changes

### 1. New TF-IDF Encoder (`src/tfidf_transformer.py`)

The `TfidfSupDTEncoder` class implements a supervised TF-IDF approach with the following features:

#### Text Preprocessing
- **Lowercase conversion**: All text is converted to lowercase for consistency
- **Punctuation removal**: All punctuation marks are removed
- **Spanish stopwords removal**: Common Spanish words that don't carry predictive value are filtered out
- **Whitespace normalization**: Extra whitespace is removed

#### Spanish Stopwords
The encoder includes a comprehensive list of 100+ Spanish stopwords including:
- Articles: el, la, los, las, un, una
- Prepositions: de, en, con, por, para
- Pronouns: yo, tú, él, ella, nosotros
- Common verbs: ser, estar, haber, tener, hacer
- And many more common Spanish words

#### Feature Extraction
1. **TF-IDF Vectorization**: Converts preprocessed text into TF-IDF features
   - Supports unigrams and bigrams (configurable)
   - Configurable min/max document frequency
   - Initial vocabulary size: up to 10,000 features

2. **Supervised Feature Selection**: Uses a Decision Tree classifier to identify important features
   - Trains a Decision Tree on the TF-IDF features
   - Ranks features by their importance scores
   - Selects top N features (default: 100)

3. **Reduced Feature Space**: Creates a final TF-IDF vectorizer using only the selected features

### 2. Updated Training Script (`src/fit_encoder.py`)

The training script has been updated to:
- Use `TfidfSupDTEncoder` instead of `SbertSupConEncoder`
- Remove dependency on HuggingFace tokenizers
- Simplify data processing (no token truncation needed)
- Display top 20 most important features after training

#### Configuration Parameters
```python
N_FEATURES = 100              # Number of top features to select
MAX_FEATURES = 10000          # Maximum features for initial TF-IDF
MIN_DF = 2                    # Minimum document frequency
MAX_DF = 0.95                 # Maximum document frequency
NGRAM_RANGE = (1, 2)          # Unigrams and bigrams
DT_MAX_DEPTH = 10             # Decision tree max depth
DT_MIN_SAMPLES_SPLIT = 100    # Min samples to split in tree
```

### 3. Updated Application Script (`src/apply_encoder_train_model.py`)

The application script has been updated to:
- Load the TF-IDF encoder instead of the sentence transformer encoder
- Use TF-IDF features instead of embeddings
- Update cache file names to reflect TF-IDF features

## Advantages of TF-IDF Approach

1. **Interpretability**: The top features are human-readable tokens that can be easily understood
2. **No GPU Required**: TF-IDF doesn't require GPU acceleration, making it faster on CPU
3. **Faster Training**: No need to fine-tune large transformer models
4. **Language-Specific**: Custom Spanish stopwords ensure better feature quality
5. **Supervised Selection**: Decision Tree identifies features that are actually predictive of the target
6. **Smaller Model Size**: The final model is much smaller than transformer-based models

## Disadvantages Compared to Embeddings

1. **No Semantic Understanding**: TF-IDF treats words as independent tokens, missing semantic relationships
2. **Vocabulary Limitation**: Can only handle tokens seen during training
3. **No Context**: Doesn't capture word order or context like transformers do
4. **Sparse Representations**: TF-IDF vectors are sparse, which may be less effective for some models

## Usage

### Training the Encoder

```bash
python src/fit_encoder.py
```

This will:
1. Load SMS data from Azure Data Lake
2. Preprocess the text (lowercase, remove punctuation, remove stopwords)
3. Fit TF-IDF vectorizer
4. Train Decision Tree for feature selection
5. Select top 100 features
6. Save the encoder to `cache/tfidf_encoder.pkl`

### Applying the Encoder

```bash
python src/apply_encoder_train_model.py
```

This will:
1. Load the trained TF-IDF encoder
2. Transform train/test data into TF-IDF features
3. Train classification models (XGBoost, Logistic Regression, Neural Network)
4. Evaluate models using ROC AUC

## Testing

A test script is provided to verify the encoder works correctly:

```bash
python test_tfidf_encoder.py
```

This runs a simple test with sample Spanish SMS messages and displays the top features selected.

## Output Files

The pipeline generates the following cached files:

```
cache/
├── tfidf_encoder.pkl           # Trained TF-IDF encoder
├── train_data.parquet          # Processed training data
├── test_data.parquet           # Processed test data
├── train_tfidf_features.npy    # Training TF-IDF features
├── test_tfidf_features.npy     # Test TF-IDF features
└── roc_curve_comparison.png    # ROC curve visualization
```

## Next Steps

1. Run the training pipeline on the full dataset
2. Compare performance with the embedding-based approach
3. Analyze the top features to gain insights into default prediction
4. Consider tuning hyperparameters (n_features, dt_max_depth, etc.)
5. Experiment with different n-gram ranges or feature selection methods

