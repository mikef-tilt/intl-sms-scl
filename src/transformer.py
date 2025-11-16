from sklearn.utils.validation import check_is_fitted, check_array
from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd

class SbertSupConEncoder:
    """
    An sklearn-compatible wrapper for fine-tuning a SentenceTransformer
    model using Supervised Contrastive (SupCon) Loss.

    Supports `warm_start` for continual or incremental training.
    """
    def __init__(self,
                 base_model_name='jaimevera1107/all‑MiniLM‑L6‑v2‑similarity‑es',
                 embedding_dim=128,
                 n_epochs=3,
                 batch_size=16,
                 lr=2e-5,
                 temperature=0.07,
                 device=None,
                 random_state=None,
                 warm_start=False,
                 gradient_checkpointing=False):

        self.base_model_name = base_model_name
        self.embedding_dim = embedding_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.temperature = temperature
        self.device = device
        self.random_state = random_state
        self.warm_start = warm_start # <-- Store parameter
        self.gradient_checkpointing = gradient_checkpointing

    def fit(self, X, y):
        """
        Fine-tunes the SentenceTransformer model.

        If `warm_start=True` and the model is already fitted,
        it continues training on the existing model. Otherwise,
        it initializes a new model.
        """

        if self.random_state:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        if self.device is None:
            self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_ = self.device

        # Log GPU availability and usage
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  Using device: {self.device_}")
            if self.device_ == "cuda":
                print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print(f"⚠ No GPU available, using CPU")
            print(f"  Device: {self.device_}")
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        
        train_examples = []
        for text, label in zip(X, y):
            train_examples.append(InputExample(texts=[text], label=int(label)))

        is_first_fit = not (hasattr(self, 'is_fitted_') and self.is_fitted_)

        if is_first_fit or not self.warm_start:
            # Initialize model from scratch
            print(f"Initializing new model from {self.base_model_name}...")
            base_model = models.Transformer(self.base_model_name)
            pooling_model = models.Pooling(
                base_model.get_word_embedding_dimension()
            )
            projection_model = models.Dense(
                in_features=pooling_model.get_sentence_embedding_dimension(),
                out_features=self.embedding_dim
            )
            normalization_model = models.Normalize()

            self.model_ = SentenceTransformer(
                modules=[base_model, pooling_model, projection_model, normalization_model],
                device=self.device_
            )

            # Enable gradient checkpointing if requested (saves GPU memory)
            if self.gradient_checkpointing:
                print("Enabling gradient checkpointing to save GPU memory...")
                if hasattr(self.model_[0].auto_model, 'gradient_checkpointing_enable'):
                    self.model_[0].auto_model.gradient_checkpointing_enable()
        else:
            print("Continuing training on existing model...")
            self.model_.to(self.device_)

        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=self.batch_size
        )

        # Use BatchHardSoftMarginTripletLoss for supervised contrastive learning
        # This loss works with single sentences + labels and doesn't require setting a margin
        # It uses soft margin which adapts automatically during training
        train_loss = losses.BatchHardSoftMarginTripletLoss(
            model=self.model_
        )

        print(f"Training for {self.n_epochs} epochs on {len(train_examples)} new samples...")
        print(f"Using BatchHardSoftMarginTripletLoss (no margin required, adapts automatically)")

        # Configure AdamW optimizer with better parameters for faster convergence
        optimizer_params = {
            'lr': self.lr,
            'eps': 1e-8,  # Smaller epsilon for better numerical stability
            'weight_decay': 0.02,  # Increased weight decay for better regularization
            'betas': (0.9, 0.999)  # Standard Adam betas for momentum
        }

        print(f"Using AdamW optimizer with lr={self.lr}, eps=1e-8, weight_decay=0.02, betas=(0.9, 0.999)")

        # Calculate warmup steps (10% of first epoch for faster initial learning)
        warmup_steps = int(len(train_dataloader) * 0.1)

        self.model_.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.n_epochs,
            optimizer_params=optimizer_params,
            show_progress_bar=True,
            warmup_steps=warmup_steps,
            use_amp=True,  # Enable automatic mixed precision for faster training
            scheduler='warmupcosine',  # Use cosine annealing with warmup for better convergence
            weight_decay=0.02,  # Increased L2 regularization to match optimizer
            max_grad_norm=0.5  # Tighter gradient clipping for more stability
        )

        print(f"Training completed with {warmup_steps} warmup steps, cosine annealing scheduler, and tighter gradient clipping (max_norm=0.5)")

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Transforms new text data into the learned embedding space.
        """
        check_is_fitted(self, 'is_fitted_')
        check_array(X, dtype=object, ensure_2d=False)
        self.model_.eval()
        embeddings = self.model_.encode(
            X,
            show_progress_bar=False,
            device=self.device_,
            batch_size=self.batch_size,  # Use batch processing for GPU efficiency
            convert_to_numpy=True
        )
        return embeddings

    
    def fit_transform(self, X: pd.Series, y=None):
        return self.fit(X, y).transform(X)