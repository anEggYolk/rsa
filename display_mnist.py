"""
Visualize MNIST Dataset Samples

This script loads and displays sample images from the MNIST dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

def display_mnist_samples(num_samples=25):
    """Display sample images from MNIST dataset."""
    print("Loading MNIST dataset...")

    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    print(f"Training set: {X_train.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    print(f"Image shape: {X_train.shape[1]}x{X_train.shape[2]}")

    # Calculate grid size
    grid_size = int(np.sqrt(num_samples))

    # Display random samples from training set
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle('Random MNIST Training Samples', fontsize=16, fontweight='bold')

    # Select random indices
    random_indices = np.random.choice(len(X_train), num_samples, replace=False)

    for i, ax in enumerate(axes.flat):
        idx = random_indices[i]
        ax.imshow(X_train[idx], cmap='gray')
        ax.set_title(f'Label: {y_train[idx]}', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=300, bbox_inches='tight')
    print("\nSample visualization saved to mnist_samples.png")
    plt.show()

    # Display one sample from each digit (0-9)
    fig2, axes2 = plt.subplots(2, 5, figsize=(15, 6))
    fig2.suptitle('One Sample of Each Digit (0-9)', fontsize=16, fontweight='bold')

    for digit in range(10):
        # Find first occurrence of this digit
        idx = np.where(y_train == digit)[0][0]
        row = digit // 5
        col = digit % 5

        axes2[row, col].imshow(X_train[idx], cmap='gray')
        axes2[row, col].set_title(f'Digit: {digit}', fontsize=12, fontweight='bold')
        axes2[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('mnist_digits_0_to_9.png', dpi=300, bbox_inches='tight')
    print("Digit samples (0-9) saved to mnist_digits_0_to_9.png")
    plt.show()

    # Print dataset statistics
    print("\n" + "="*50)
    print("MNIST Dataset Statistics")
    print("="*50)
    print(f"\nTraining set size: {X_train.shape[0]:,}")
    print(f"Test set size: {X_test.shape[0]:,}")
    print(f"Total images: {X_train.shape[0] + X_test.shape[0]:,}")
    print(f"Image dimensions: {X_train.shape[1]} x {X_train.shape[2]} pixels")
    print(f"Pixel value range: {X_train.min()} to {X_train.max()}")

    print("\nClass distribution in training set:")
    for digit in range(10):
        count = np.sum(y_train == digit)
        percentage = 100 * count / len(y_train)
        print(f"  Digit {digit}: {count:,} images ({percentage:.2f}%)")

    print("\nClass distribution in test set:")
    for digit in range(10):
        count = np.sum(y_test == digit)
        percentage = 100 * count / len(y_test)
        print(f"  Digit {digit}: {count:,} images ({percentage:.2f}%)")

if __name__ == "__main__":
    display_mnist_samples(num_samples=25)
