"""
MNIST Logistic Regression with Weight Visualization

This script trains a logistic regression model on the MNIST dataset
and visualizes the weights for each one-vs-rest classifier.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train logistic regression model with one-vs-rest approach."""
    print("\nTraining Logistic Regression model...")

    # Create and train the model
    # Using one-vs-rest (OvR) multi-class strategy
    model = LogisticRegression(
        multi_class='ovr',  # One-vs-Rest
        solver='lbfgs',
        max_iter=100,
        random_state=42,
        verbose=1
    )

    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, y_pred

def visualize_weights(model, save_path='mnist_weights.png'):
    """Visualize the weights for each one-vs-rest classifier."""
    print("\nVisualizing weights for each classifier...")

    # Get the weights (coefficients) for each classifier
    # Shape: (n_classes, n_features) = (10, 784)
    weights = model.coef_

    # Create a figure with 2 rows and 5 columns for 10 digits
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Logistic Regression Weights for MNIST One-vs-Rest Classifiers',
                 fontsize=16, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        # Reshape weights to 28x28 image
        weight_image = weights[i].reshape(28, 28)

        # Display the weight matrix
        im = ax.imshow(weight_image, cmap='RdBu', interpolation='nearest')
        ax.set_title(f'Digit {i}', fontsize=12, fontweight='bold')
        ax.axis('off')

        # Add colorbar for each subplot
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.show()

    # Also create a separate detailed visualization
    fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))
    fig2.suptitle('Detailed Weight Patterns for Each Digit Classifier',
                  fontsize=16, fontweight='bold')

    for i, ax in enumerate(axes2.flat):
        weight_image = weights[i].reshape(28, 28)

        im = ax.imshow(weight_image, cmap='seismic', interpolation='bilinear')
        ax.set_title(f'Digit {i}\nMax: {weight_image.max():.3f}, Min: {weight_image.min():.3f}',
                    fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('mnist_weights_detailed.png', dpi=300, bbox_inches='tight')
    print(f"Detailed visualization saved to mnist_weights_detailed.png")
    plt.show()

def analyze_confusion_matrix(y_test, y_pred, save_path='confusion_matrix.png'):
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
    ax.set_title('Confusion Matrix - MNIST Logistic Regression', fontsize=14, fontweight='bold')

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

    # Analyze misclassifications for digit 5
    print("\n" + "="*60)
    print("MISCLASSIFICATION ANALYSIS FOR DIGIT 5")
    print("="*60)

    digit = 5
    total_actual_5s = cm[digit, :].sum()
    correct_5s = cm[digit, digit]
    incorrect_5s = total_actual_5s - correct_5s

    print(f"\nTotal actual 5s in test set: {total_actual_5s}")
    print(f"Correctly predicted as 5: {correct_5s} ({100*correct_5s/total_actual_5s:.1f}%)")
    print(f"Incorrectly predicted: {incorrect_5s} ({100*incorrect_5s/total_actual_5s:.1f}%)")

    print(f"\nWhen the model got a 5 wrong, it predicted:")
    for predicted_digit in range(10):
        if predicted_digit != digit:
            count = cm[digit, predicted_digit]
            if count > 0:
                percentage = 100 * count / incorrect_5s
                print(f"  Digit {predicted_digit}: {count} times ({percentage:.1f}% of errors)")

    # Analysis for all digits
    print("\n" + "="*60)
    print("MISCLASSIFICATION ANALYSIS FOR ALL DIGITS")
    print("="*60)

    for digit in range(10):
        total_actual = cm[digit, :].sum()
        correct = cm[digit, digit]
        incorrect = total_actual - correct

        if incorrect > 0:
            print(f"\nDigit {digit}: {incorrect} misclassifications out of {total_actual} ({100*incorrect/total_actual:.1f}%)")
            print(f"  Most commonly confused with:")

            # Get top 3 confusion targets
            confusion_counts = [(i, cm[digit, i]) for i in range(10) if i != digit and cm[digit, i] > 0]
            confusion_counts.sort(key=lambda x: x[1], reverse=True)

            for predicted_digit, count in confusion_counts[:3]:
                percentage = 100 * count / incorrect
                print(f"    {predicted_digit}: {count} times ({percentage:.1f}% of errors)")

    # Overall distribution of incorrect guesses
    print("\n" + "="*60)
    print("OVERALL DISTRIBUTION OF INCORRECT GUESSES")
    print("="*60)

    # Calculate total errors
    total_predictions = cm.sum()
    total_correct = np.trace(cm)  # Sum of diagonal
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
        percentage = 100 * count / total_errors
        print(f"{rank:<6}{true_digit:<6}{'→':<4}{pred_digit:<6}{count:<8}{percentage:.2f}%")

    # Distribution by predicted digit (for all errors)
    print(f"\nWhen the model made errors, what did it predict?")
    print(f"{'Predicted Digit':<20}{'Count':<10}{'% of all errors':<20}")
    print("-" * 50)

    for pred_digit in range(10):
        # Count all misclassifications that resulted in this prediction
        count = sum(cm[true_digit, pred_digit] for true_digit in range(10) if true_digit != pred_digit)
        if count > 0:
            percentage = 100 * count / total_errors
            print(f"{pred_digit:<20}{count:<10}{percentage:.2f}%")

    # Distribution by true digit (for all errors)
    print(f"\nWhich true digits were most often misclassified?")
    print(f"{'True Digit':<20}{'Errors':<10}{'% of all errors':<20}{'Error Rate':<20}")
    print("-" * 70)

    digit_errors = []
    for true_digit in range(10):
        total_actual = cm[true_digit, :].sum()
        errors = total_actual - cm[true_digit, true_digit]
        error_rate = 100 * errors / total_actual
        pct_of_all_errors = 100 * errors / total_errors
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
    model, y_pred = train_logistic_regression(X_train, y_train, X_test, y_test)

    # Analyze confusion matrix
    analyze_confusion_matrix(y_test, y_pred)

    # Visualize weights
    visualize_weights(model)

    print("\n" + "="*50)
    print("Model training and visualization complete!")
    print("="*50)

    # Print weight statistics
    print("\nWeight Statistics:")
    print(f"Weight matrix shape: {model.coef_.shape}")
    print(f"Number of classifiers: {model.coef_.shape[0]}")
    print(f"Number of features per classifier: {model.coef_.shape[1]}")

    for i in range(10):
        weights = model.coef_[i]
        print(f"\nDigit {i}:")
        print(f"  Mean weight: {weights.mean():.6f}")
        print(f"  Std weight: {weights.std():.6f}")
        print(f"  Min weight: {weights.min():.6f}")
        print(f"  Max weight: {weights.max():.6f}")

if __name__ == "__main__":
    main()
