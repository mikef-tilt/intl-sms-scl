"""
TF-IDF based encoder with supervised feature selection using Decision Trees.

This module provides a supervised TF-IDF approach that:
1. Preprocesses Spanish SMS text (lowercase, remove punctuation, remove stopwords)
2. Applies TF-IDF vectorization
3. Uses a Decision Tree classifier to identify the top N most important tokens
4. Creates a reduced feature space based on the most important tokens
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted, check_array
import numpy as np
import pandas as pd
import re
import string
from multiprocessing import Pool, cpu_count
from functools import partial
import os


class TfidfSupDTEncoder(BaseEstimator, TransformerMixin):
    """
    An sklearn-compatible TF-IDF encoder with supervised feature selection
    using Decision Tree feature importances.
    
    This encoder:
    1. Preprocesses Spanish text (lowercase, remove punctuation, remove stopwords)
    2. Fits a TF-IDF vectorizer on the training data
    3. Trains a Decision Tree classifier to identify important features
    4. Selects the top N tokens based on feature importance
    5. Transforms text into a reduced TF-IDF representation
    
    Parameters
    ----------
    n_features : int, default=100
        Number of top features to select based on decision tree importance
    max_features : int, default=10000
        Maximum number of features for initial TF-IDF vectorization
    min_df : int or float, default=2
        Minimum document frequency for TF-IDF
    max_df : float, default=0.95
        Maximum document frequency for TF-IDF
    ngram_range : tuple, default=(1, 2)
        N-gram range for TF-IDF (unigrams and bigrams by default)
    dt_max_depth : int, default=10
        Maximum depth of the decision tree for feature selection
    dt_min_samples_split : int, default=100
        Minimum samples required to split a node in the decision tree
    n_jobs : int, default=-1
        Number of parallel jobs for preprocessing. -1 means use all CPUs
    random_state : int, default=None
        Random state for reproducibility
    """
    
    # Spanish stopwords - common words to remove
    SPANISH_STOPWORDS = {
        'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'haber',
        'por', 'con', 'su', 'para', 'como', 'estar', 'tener', 'le', 'lo', 'todo',
        'pero', 'más', 'hacer', 'o', 'poder', 'decir', 'este', 'ir', 'otro', 'ese',
        'la', 'si', 'me', 'ya', 'ver', 'porque', 'dar', 'cuando', 'él', 'muy',
        'sin', 'vez', 'mucho', 'saber', 'qué', 'sobre', 'mi', 'alguno', 'mismo',
        'yo', 'también', 'hasta', 'año', 'dos', 'querer', 'entre', 'así', 'primero',
        'desde', 'grande', 'eso', 'ni', 'nos', 'llegar', 'pasar', 'tiempo', 'ella',
        'sí', 'día', 'uno', 'bien', 'poco', 'deber', 'entonces', 'poner', 'cosa',
        'tanto', 'hombre', 'parecer', 'nuestro', 'tan', 'donde', 'ahora', 'parte',
        'después', 'vida', 'quedar', 'siempre', 'creer', 'hablar', 'llevar', 'dejar',
        'nada', 'cada', 'seguir', 'menos', 'nuevo', 'encontrar', 'algo', 'solo',
        'decir', 'mundo', 'país', 'fin', 'bajo', 'cómo', 'durante', 'además',
        'unos', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué',
        'unos', 'yo', 'del', 'voy', 'muy', 'fue', 'ha', 'soy', 'es', 'eres',
        'somos', 'sois', 'son', 'he', 'has', 'hemos', 'habéis', 'han', 'había',
        'te', 'les', 'nos', 'os', 'tu', 'tus', 'sus', 'nuestros', 'vuestros'
    }
    
    def __init__(
        self,
        n_features=100,
        max_features=10000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        dt_max_depth=10,
        dt_min_samples_split=100,
        n_jobs=-1,
        random_state=None
    ):
        self.n_features = n_features
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.dt_max_depth = dt_max_depth
        self.dt_min_samples_split = dt_min_samples_split
        self.n_jobs = n_jobs
        self.random_state = random_state
    
    @staticmethod
    def preprocess_spanish_text(text):
        """
        Preprocess Spanish SMS text.

        Steps:
        1. Convert to lowercase
        2. Remove punctuation
        3. Remove extra whitespace
        4. Remove stopwords

        Parameters
        ----------
        text : str
            Input text to preprocess

        Returns
        -------
        str
            Preprocessed text
        """
        if not isinstance(text, str) or not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove stopwords
        words = text.split()
        words = [w for w in words if w not in TfidfSupDTEncoder.SPANISH_STOPWORDS]

        return ' '.join(words)

    def _preprocess_texts_parallel(self, texts):
        """
        Preprocess texts in parallel using multiprocessing.

        Parameters
        ----------
        texts : array-like of shape (n_samples,)
            Input texts to preprocess

        Returns
        -------
        list
            List of preprocessed texts
        """
        n_jobs = self.n_jobs if self.n_jobs > 0 else cpu_count()

        # For small datasets, don't use multiprocessing (overhead not worth it)
        # Threshold is higher because preprocessing is very fast
        if len(texts) < 50000 or n_jobs == 1:
            return [self.preprocess_spanish_text(text) for text in texts]

        # Use multiprocessing for large datasets (50k+ samples)
        print(f"  Using {n_jobs} parallel workers for preprocessing...")
        chunk_size = max(1, len(texts) // (n_jobs * 4))  # 4 chunks per worker

        with Pool(processes=n_jobs) as pool:
            processed_texts = pool.map(
                self.preprocess_spanish_text,
                texts,
                chunksize=chunk_size
            )

        return processed_texts

    def fit(self, X, y):
        """
        Fit the TF-IDF encoder with supervised feature selection.

        Steps:
        1. Preprocess all texts
        2. Fit TF-IDF vectorizer
        3. Train Decision Tree classifier
        4. Select top N features based on feature importance
        5. Create reduced TF-IDF vectorizer with selected features

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Training text data
        y : array-like of shape (n_samples,)
            Target labels

        Returns
        -------
        self
            Fitted encoder
        """
        # Set random state
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Preprocess texts in parallel
        print(f"Preprocessing {len(X):,} texts...")
        X_processed = self._preprocess_texts_parallel(X)

        # Fit initial TF-IDF vectorizer
        print(f"Fitting TF-IDF vectorizer (max_features={self.max_features})...")
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            lowercase=False,  # Already lowercased in preprocessing
            token_pattern=r'\b\w+\b'
        )

        X_tfidf = self.vectorizer_.fit_transform(X_processed)
        print(f"  Initial TF-IDF shape: {X_tfidf.shape}")

        # Train Decision Tree for feature selection
        print(f"Training Decision Tree for feature selection...")
        self.dt_classifier_ = DecisionTreeClassifier(
            max_depth=self.dt_max_depth,
            min_samples_split=self.dt_min_samples_split,
            random_state=self.random_state
        )
        self.dt_classifier_.fit(X_tfidf, y)

        # Get feature importances
        feature_importances = self.dt_classifier_.feature_importances_
        feature_names = self.vectorizer_.get_feature_names_out()

        # Select top N features
        top_indices = np.argsort(feature_importances)[-self.n_features:]
        self.selected_features_ = feature_names[top_indices]
        self.selected_indices_ = top_indices

        print(f"  Selected top {self.n_features} features")
        print(f"  Top 10 features: {list(self.selected_features_[-10:])}")

        # Create vocabulary for reduced vectorizer
        self.vocabulary_ = {feature: idx for idx, feature in enumerate(self.selected_features_)}

        # Create reduced TF-IDF vectorizer with selected features
        self.reduced_vectorizer_ = TfidfVectorizer(
            vocabulary=self.vocabulary_,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            lowercase=False,
            token_pattern=r'\b\w+\b'
        )

        # Fit the reduced vectorizer
        self.reduced_vectorizer_.fit(X_processed)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Transform text data into reduced TF-IDF representation.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Text data to transform

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            TF-IDF representation using selected features
        """
        check_is_fitted(self, 'is_fitted_')

        # Preprocess texts in parallel
        X_processed = self._preprocess_texts_parallel(X)

        # Transform using reduced vectorizer
        X_tfidf = self.reduced_vectorizer_.transform(X_processed)

        return X_tfidf.toarray()

    def fit_transform(self, X, y=None):
        """
        Fit the encoder and transform the data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Training text data
        y : array-like of shape (n_samples,)
            Target labels

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            TF-IDF representation using selected features
        """
        return self.fit(X, y).transform(X)

    def get_feature_names(self):
        """
        Get the selected feature names.

        Returns
        -------
        feature_names : ndarray
            Array of selected feature names
        """
        check_is_fitted(self, 'is_fitted_')
        return self.selected_features_

    def get_feature_importances(self):
        """
        Get the feature importances from the decision tree.

        Returns
        -------
        importances : dict
            Dictionary mapping feature names to their importances
        """
        check_is_fitted(self, 'is_fitted_')

        feature_names = self.vectorizer_.get_feature_names_out()
        importances = self.dt_classifier_.feature_importances_

        # Create dictionary for selected features only
        selected_importances = {
            feature_names[idx]: importances[idx]
            for idx in self.selected_indices_
        }

        # Sort by importance
        return dict(sorted(selected_importances.items(), key=lambda x: x[1], reverse=True))

