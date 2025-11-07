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
from sklearn.metrics import accuracy_score, classification_report
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

    return model

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
    model = train_logistic_regression(X_train, y_train, X_test, y_test)

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
