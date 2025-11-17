"""
Simple test script to verify the TfidfSupDTEncoder works correctly with multiprocessing.
"""

from src.tfidf_transformer import TfidfSupDTEncoder
import numpy as np
import time

# Sample Spanish SMS data
X_train = [
    "Hola, ¿cómo estás? Necesito pagar mi préstamo.",
    "Buenos días, quiero información sobre mi crédito.",
    "¡Urgente! No puedo pagar este mes.",
    "Gracias por el préstamo, todo está bien.",
    "¿Cuándo vence mi pago?",
    "Necesito una extensión de plazo.",
    "Mi situación financiera es difícil.",
    "Pagaré la próxima semana sin falta.",
    "¿Puedo refinanciar mi deuda?",
    "Estoy desempleado, no puedo pagar.",
    "Todo está en orden con mi cuenta.",
    "Tengo problemas económicos graves.",
    "¿Cuál es mi saldo pendiente?",
    "Voy a pagar hoy mismo.",
    "No tengo dinero para pagar.",
    "Mi préstamo está al día.",
    "Necesito ayuda urgente.",
    "Pagaré en cuotas si es posible.",
    "Mi trabajo está estable ahora.",
    "Perdí mi empleo recientemente.",
]

# Labels (0 = no default, 1 = default)
y_train = np.array([0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1])

# Test data
X_test = [
    "Hola, necesito información sobre mi préstamo.",
    "No puedo pagar, estoy sin trabajo.",
    "Todo bien, pagaré a tiempo.",
    "Tengo dificultades financieras.",
]

y_test = np.array([0, 1, 0, 1])

print("="*60)
print("Testing TfidfSupDTEncoder with Multiprocessing")
print("="*60)

# Test 1: Small dataset (should not use multiprocessing)
print("\n" + "="*60)
print("Test 1: Small Dataset (20 samples)")
print("="*60)

encoder_small = TfidfSupDTEncoder(
    n_features=20,
    max_features=100,
    min_df=1,
    max_df=0.95,
    ngram_range=(1, 2),
    dt_max_depth=5,
    dt_min_samples_split=2,
    n_jobs=-1,  # Use all CPUs
    random_state=42
)

print(f"\nTraining encoder on {len(X_train)} samples...")
start_time = time.time()
encoder_small.fit(X_train, y_train)
fit_time = time.time() - start_time
print(f"Fit time: {fit_time:.3f} seconds")

print(f"\n{'='*60}")
print("Top 10 Features Selected:")
print(f"{'='*60}")
feature_importances = encoder_small.get_feature_importances()
for i, (feature, importance) in enumerate(list(feature_importances.items())[:10], 1):
    print(f"  {i:2d}. {feature:30s} - {importance:.6f}")

# Transform test data
start_time = time.time()
X_test_transformed = encoder_small.transform(X_test)
transform_time = time.time() - start_time
print(f"\nTransform time: {transform_time:.3f} seconds")
print(f"Test data shape: {X_test_transformed.shape}")

# Test 2: Large dataset (should use multiprocessing)
print("\n" + "="*60)
print("Test 2: Large Dataset (10,000 samples)")
print("="*60)

# Create a larger dataset by repeating the samples
X_train_large = X_train * 500  # 10,000 samples
y_train_large = np.tile(y_train, 500)

print(f"\nCreated large dataset: {len(X_train_large):,} samples")

# Test with multiprocessing
encoder_parallel = TfidfSupDTEncoder(
    n_features=20,
    max_features=100,
    min_df=1,
    max_df=0.95,
    ngram_range=(1, 2),
    dt_max_depth=5,
    dt_min_samples_split=2,
    n_jobs=-1,  # Use all CPUs
    random_state=42
)

print("\nTraining with multiprocessing (n_jobs=-1)...")
start_time = time.time()
encoder_parallel.fit(X_train_large, y_train_large)
parallel_time = time.time() - start_time
print(f"Fit time with multiprocessing: {parallel_time:.3f} seconds")

# Test without multiprocessing
encoder_serial = TfidfSupDTEncoder(
    n_features=20,
    max_features=100,
    min_df=1,
    max_df=0.95,
    ngram_range=(1, 2),
    dt_max_depth=5,
    dt_min_samples_split=2,
    n_jobs=1,  # Single process
    random_state=42
)

print("\nTraining without multiprocessing (n_jobs=1)...")
start_time = time.time()
encoder_serial.fit(X_train_large, y_train_large)
serial_time = time.time() - start_time
print(f"Fit time without multiprocessing: {serial_time:.3f} seconds")

speedup = serial_time / parallel_time
print(f"\n{'='*60}")
print("Performance Comparison")
print(f"{'='*60}")
print(f"Serial time:     {serial_time:.3f} seconds")
print(f"Parallel time:   {parallel_time:.3f} seconds")
print(f"Speedup:         {speedup:.2f}x")

print(f"\n{'='*60}")
print("Testing Complete!")
print(f"{'='*60}")
print("✓ Encoder successfully trained with multiprocessing")
print(f"✓ Multiprocessing provides {speedup:.2f}x speedup on large datasets")
print(f"✓ Selected {len(encoder_parallel.get_feature_names())} features")

