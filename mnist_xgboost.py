"""
MNIST XGBoost Classifier with Feature Importance Visualization

This script trains an XGBoost model on the MNIST dataset
and visualizes feature importance for classification.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_mnist_data():
    """Load and preprocess MNIST dataset."""
    print("Loading MNIST dataset...")

    try:
        # Try using keras/tensorflow
        from tensorflow import keras
        (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = keras.datasets.mnist.load_data()

        # Flatten the images
        X_train_raw = X_train_raw.reshape(X_train_raw.shape[0], -1)
        X_test_raw = X_test_raw.reshape(X_test_raw.shape[0], -1)

        # Combine train and test for our own split later
        X = np.vstack([X_train_raw, X_test_raw])
        y = np.concatenate([y_train_raw, y_test_raw])

        print("Loaded MNIST using TensorFlow/Keras")

    except ImportError:
        # Fallback to sklearn
        print("TensorFlow not available, trying sklearn...")
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X, y = mnist.data, mnist.target
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

    # Convert to numpy arrays and ensure correct dtypes
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # Normalize pixel values to [0, 1]
    X = X / 255.0

    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Unique labels: {np.unique(y)}")

    return X, y

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model for multi-class classification."""
    print("\nTraining XGBoost model...")

    # Create XGBoost classifier
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=10,
        max_depth=8,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',
        eval_metric='mlogloss'
    )

    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, y_pred

def visualize_feature_importance(model, save_path='xgboost_feature_importance.png'):
    """Visualize feature importance as a heatmap."""
    print("\nVisualizing feature importance...")

    # Get feature importance
    importance = model.feature_importances_

    # Reshape to 28x28 image
    importance_image = importance.reshape(28, 28)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('XGBoost Feature Importance for MNIST Classification',
                 fontsize=16, fontweight='bold')

    # Heatmap visualization
    im1 = ax1.imshow(importance_image, cmap='hot', interpolation='nearest')
    ax1.set_title('Feature Importance Heatmap', fontsize=14)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Smooth version
    im2 = ax2.imshow(importance_image, cmap='viridis', interpolation='bilinear')
    ax2.set_title('Feature Importance (Smoothed)', fontsize=14)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance saved to {save_path}")
    plt.show()

    # Print statistics
    print(f"\nFeature Importance Statistics:")
    print(f"  Mean: {importance.mean():.6f}")
    print(f"  Std: {importance.std():.6f}")
    print(f"  Min: {importance.min():.6f}")
    print(f"  Max: {importance.max():.6f}")

    # Find most important pixels
    top_n = 20
    top_indices = np.argsort(importance)[-top_n:][::-1]
    print(f"\nTop {top_n} most important pixels:")
    for rank, idx in enumerate(top_indices, 1):
        row = idx // 28
        col = idx % 28
        print(f"  {rank}. Pixel ({row}, {col}): {importance[idx]:.6f}")

def analyze_confusion_matrix(y_test, y_pred, save_path='xgboost_confusion_matrix.png'):
    """Create and analyze confusion matrix."""
    print("\nGenerating confusion matrix...")

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Visualize confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of predictions', rotation=270, labelpad=20)

    # Set labels
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - MNIST XGBoost', fontsize=14, fontweight='bold')

    # Set ticks
    tick_marks = np.arange(10)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.show()

    # Overall distribution of incorrect guesses
    print("\n" + "="*60)
    print("OVERALL DISTRIBUTION OF INCORRECT GUESSES")
    print("="*60)

    # Calculate total errors
    total_predictions = cm.sum()
    total_correct = np.trace(cm)
    total_errors = total_predictions - total_correct

    print(f"\nTotal predictions: {total_predictions}")
    print(f"Correct predictions: {total_correct} ({100*total_correct/total_predictions:.2f}%)")
    print(f"Incorrect predictions: {total_errors} ({100*total_errors/total_predictions:.2f}%)")

    # Find all misclassification pairs
    error_pairs = []
    for true_digit in range(10):
        for pred_digit in range(10):
            if true_digit != pred_digit and cm[true_digit, pred_digit] > 0:
                error_pairs.append((true_digit, pred_digit, cm[true_digit, pred_digit]))

    # Sort by frequency
    error_pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTop 20 most common misclassification pairs:")
    print(f"{'Rank':<6}{'True':<6}{'→':<4}{'Pred':<6}{'Count':<8}{'% of all errors':<20}")
    print("-" * 60)

    for rank, (true_digit, pred_digit, count) in enumerate(error_pairs[:20], 1):
        percentage = 100 * count / total_errors if total_errors > 0 else 0
        print(f"{rank:<6}{true_digit:<6}{'→':<4}{pred_digit:<6}{count:<8}{percentage:.2f}%")

    # Which true digits were most often misclassified?
    print(f"\nWhich true digits were most often misclassified?")
    print(f"{'True Digit':<20}{'Errors':<10}{'% of all errors':<20}{'Error Rate':<20}")
    print("-" * 70)

    digit_errors = []
    for true_digit in range(10):
        total_actual = cm[true_digit, :].sum()
        errors = total_actual - cm[true_digit, true_digit]
        error_rate = 100 * errors / total_actual if total_actual > 0 else 0
        pct_of_all_errors = 100 * errors / total_errors if total_errors > 0 else 0
        digit_errors.append((true_digit, errors, pct_of_all_errors, error_rate))

    # Sort by number of errors
    digit_errors.sort(key=lambda x: x[1], reverse=True)

    for true_digit, errors, pct_of_all, error_rate in digit_errors:
        print(f"{true_digit:<20}{errors:<10}{pct_of_all:.2f}%{' ':<14}{error_rate:.2f}%")

    return cm

def main():
    """Main execution function."""
    # Load data
    X, y = load_mnist_data()

    # Split into train and test sets
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Train model
    model, y_pred = train_xgboost(X_train, y_train, X_test, y_test)

    # Analyze confusion matrix
    analyze_confusion_matrix(y_test, y_pred)

    # Visualize feature importance
    visualize_feature_importance(model)

    print("\n" + "="*50)
    print("Model training and visualization complete!")
    print("="*50)

if __name__ == "__main__":
    main()
