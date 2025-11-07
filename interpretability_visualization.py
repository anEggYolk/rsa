"""
MNIST Model Interpretability Visualizations

This script provides comprehensive interpretability visualizations for both
Logistic Regression and XGBoost models on MNIST dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

def load_mnist_data():
    """Load and preprocess MNIST dataset."""
    print("Loading MNIST dataset...")

    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = keras.datasets.mnist.load_data()

    # Flatten the images
    X_train_raw = X_train_raw.reshape(X_train_raw.shape[0], -1)
    X_test_raw = X_test_raw.reshape(X_test_raw.shape[0], -1)

    # Combine train and test for our own split later
    X = np.vstack([X_train_raw, X_test_raw])
    y = np.concatenate([y_train_raw, y_test_raw])

    # Convert to numpy arrays and ensure correct dtypes
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # Normalize pixel values to [0, 1]
    X = X / 255.0

    print(f"Dataset loaded: {X.shape}")
    return X, y

def train_models(X_train, y_train, X_test, y_test):
    """Train both Logistic Regression and XGBoost models."""
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(
        multi_class='ovr',
        solver='lbfgs',
        max_iter=100,
        random_state=42,
        verbose=0
    )
    lr_model.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=10,
        max_depth=8,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        tree_method='hist',
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
    print(f"XGBoost Accuracy: {xgb_acc:.4f}")

    return lr_model, xgb_model

def visualize_model_comparison(lr_model, xgb_model):
    """Compare weight/importance visualizations side by side."""
    print("\nGenerating model comparison visualization...")

    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    fig.suptitle('Model Interpretability: Logistic Regression Weights vs XGBoost Feature Importance',
                 fontsize=16, fontweight='bold', y=1.02)

    # Logistic Regression weights for each digit
    lr_weights = lr_model.coef_

    # XGBoost feature importance
    xgb_importance = xgb_model.feature_importances_

    for digit in range(10):
        # Plot Logistic Regression weights
        ax_lr = axes[0, digit]
        weight_image = lr_weights[digit].reshape(28, 28)
        im_lr = ax_lr.imshow(weight_image, cmap='RdBu', interpolation='nearest')
        ax_lr.set_title(f'LR: {digit}', fontsize=10, fontweight='bold')
        ax_lr.axis('off')

        # Plot XGBoost importance (same for all digits, so we'll show it scaled)
        ax_xgb = axes[1, digit]
        if digit == 0:
            # Show actual feature importance for first digit
            importance_image = xgb_importance.reshape(28, 28)
            im_xgb = ax_xgb.imshow(importance_image, cmap='viridis', interpolation='nearest')
            ax_xgb.set_title(f'XGB: All', fontsize=10, fontweight='bold')
        else:
            # For other digits, show empty or same importance
            importance_image = xgb_importance.reshape(28, 28)
            im_xgb = ax_xgb.imshow(importance_image, cmap='viridis', interpolation='nearest')
            ax_xgb.set_title(f'XGB: All', fontsize=10, fontweight='bold')
        ax_xgb.axis('off')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved model_comparison.png")
    plt.show()

def visualize_prediction_heatmaps(lr_model, xgb_model, X_test, y_test):
    """Visualize how models make predictions on specific examples."""
    print("\nGenerating prediction heatmap visualizations...")

    # Select one example of each digit
    examples = []
    for digit in range(10):
        idx = np.where(y_test == digit)[0][0]
        examples.append((X_test[idx], y_test[idx], idx))

    fig, axes = plt.subplots(3, 10, figsize=(20, 7))
    fig.suptitle('Prediction Interpretability: Original Image, LR Activation, XGB Contribution',
                 fontsize=16, fontweight='bold', y=0.98)

    for col, (x, true_label, idx) in enumerate(examples):
        # Original image
        ax_orig = axes[0, col]
        ax_orig.imshow(x.reshape(28, 28), cmap='gray')
        ax_orig.set_title(f'True: {true_label}', fontsize=10, fontweight='bold')
        ax_orig.axis('off')

        # Logistic Regression activation map
        ax_lr = axes[1, col]
        lr_pred = lr_model.predict([x])[0]
        lr_weights = lr_model.coef_[true_label]
        activation = x * lr_weights  # Element-wise multiplication
        activation_image = activation.reshape(28, 28)
        im_lr = ax_lr.imshow(activation_image, cmap='coolwarm', interpolation='bilinear')
        ax_lr.set_title(f'LR→{lr_pred}', fontsize=10)
        ax_lr.axis('off')

        # XGBoost contribution (approximation using feature importance)
        ax_xgb = axes[2, col]
        xgb_pred = xgb_model.predict([x])[0]
        xgb_importance = xgb_model.feature_importances_
        contribution = x * xgb_importance  # Approximate contribution
        contribution_image = contribution.reshape(28, 28)
        im_xgb = ax_xgb.imshow(contribution_image, cmap='coolwarm', interpolation='bilinear')
        ax_xgb.set_title(f'XGB→{xgb_pred}', fontsize=10)
        ax_xgb.axis('off')

    plt.tight_layout()
    plt.savefig('prediction_heatmaps.png', dpi=300, bbox_inches='tight')
    print("Saved prediction_heatmaps.png")
    plt.show()

def visualize_decision_boundaries(lr_model, xgb_model, X_test, y_test):
    """Visualize prediction confidence for both models."""
    print("\nGenerating confidence visualization...")

    # Get prediction probabilities
    lr_proba = lr_model.predict_proba(X_test[:100])
    xgb_proba = xgb_model.predict_proba(X_test[:100])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Prediction Confidence Distribution (First 100 Test Samples)',
                 fontsize=16, fontweight='bold')

    # Logistic Regression confidence
    lr_confidence = np.max(lr_proba, axis=1)
    lr_pred = np.argmax(lr_proba, axis=1)
    lr_correct = (lr_pred == y_test[:100])

    ax1.scatter(range(100), lr_confidence, c=lr_correct, cmap='RdYlGn',
                alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0.9, color='red', linestyle='--', label='90% confidence')
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Prediction Confidence', fontsize=12)
    ax1.set_title(f'Logistic Regression\nAvg Confidence: {lr_confidence.mean():.3f}',
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # XGBoost confidence
    xgb_confidence = np.max(xgb_proba, axis=1)
    xgb_pred = np.argmax(xgb_proba, axis=1)
    xgb_correct = (xgb_pred == y_test[:100])

    ax2.scatter(range(100), xgb_confidence, c=xgb_correct, cmap='RdYlGn',
                alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0.9, color='red', linestyle='--', label='90% confidence')
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_ylabel('Prediction Confidence', fontsize=12)
    ax2.set_title(f'XGBoost\nAvg Confidence: {xgb_confidence.mean():.3f}',
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('confidence_visualization.png', dpi=300, bbox_inches='tight')
    print("Saved confidence_visualization.png")
    plt.show()

def visualize_misclassified_examples(lr_model, xgb_model, X_test, y_test):
    """Show examples where models disagree or both fail."""
    print("\nGenerating misclassification analysis...")

    lr_pred = lr_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    # Find interesting cases
    lr_wrong = lr_pred != y_test
    xgb_wrong = xgb_pred != y_test
    both_wrong = lr_wrong & xgb_wrong
    disagree = (lr_pred != xgb_pred) & (~both_wrong)

    print(f"LR wrong: {lr_wrong.sum()}")
    print(f"XGB wrong: {xgb_wrong.sum()}")
    print(f"Both wrong: {both_wrong.sum()}")
    print(f"Models disagree (but at least one correct): {disagree.sum()}")

    fig, axes = plt.subplots(3, 10, figsize=(20, 7))
    fig.suptitle('Misclassification Analysis', fontsize=16, fontweight='bold')

    # Row 1: Both models wrong
    both_wrong_indices = np.where(both_wrong)[0][:10]
    for i, idx in enumerate(both_wrong_indices):
        if i < 10:
            ax = axes[0, i]
            ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            ax.set_title(f'True:{y_test[idx]}\nLR:{lr_pred[idx]} XGB:{xgb_pred[idx]}',
                        fontsize=8)
            ax.axis('off')
    axes[0, 0].text(-5, 14, 'Both Wrong:', fontsize=12, fontweight='bold',
                    rotation=90, verticalalignment='center')

    # Row 2: LR wrong, XGB correct
    lr_wrong_xgb_right = lr_wrong & (~xgb_wrong)
    lr_wrong_indices = np.where(lr_wrong_xgb_right)[0][:10]
    for i, idx in enumerate(lr_wrong_indices):
        if i < 10:
            ax = axes[1, i]
            ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            ax.set_title(f'True:{y_test[idx]}\nLR:{lr_pred[idx]} XGB:{xgb_pred[idx]}',
                        fontsize=8)
            ax.axis('off')
    axes[1, 0].text(-5, 14, 'LR Wrong\nXGB Right:', fontsize=12, fontweight='bold',
                    rotation=90, verticalalignment='center')

    # Row 3: XGB wrong, LR correct
    xgb_wrong_lr_right = xgb_wrong & (~lr_wrong)
    xgb_wrong_indices = np.where(xgb_wrong_lr_right)[0][:10]
    for i, idx in enumerate(xgb_wrong_indices):
        if i < 10:
            ax = axes[2, i]
            ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            ax.set_title(f'True:{y_test[idx]}\nLR:{lr_pred[idx]} XGB:{xgb_pred[idx]}',
                        fontsize=8)
            ax.axis('off')
    axes[2, 0].text(-5, 14, 'XGB Wrong\nLR Right:', fontsize=12, fontweight='bold',
                    rotation=90, verticalalignment='center')

    plt.tight_layout()
    plt.savefig('misclassification_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved misclassification_analysis.png")
    plt.show()

def visualize_confidence_by_digit(lr_model, xgb_model, X_test, y_test):
    """Analyze confidence per digit class."""
    print("\nGenerating per-digit confidence analysis...")

    lr_proba = lr_model.predict_proba(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)

    lr_confidence = np.max(lr_proba, axis=1)
    xgb_confidence = np.max(xgb_proba, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Average Prediction Confidence by Digit Class',
                 fontsize=16, fontweight='bold')

    # Calculate average confidence per digit
    lr_conf_by_digit = []
    xgb_conf_by_digit = []

    for digit in range(10):
        mask = y_test == digit
        lr_conf_by_digit.append(lr_confidence[mask].mean())
        xgb_conf_by_digit.append(xgb_confidence[mask].mean())

    x = np.arange(10)
    width = 0.35

    # Logistic Regression
    bars1 = ax1.bar(x, lr_conf_by_digit, width, label='LR Confidence',
                    color='steelblue', alpha=0.8, edgecolor='black')
    ax1.axhline(y=0.9, color='red', linestyle='--', label='90% threshold')
    ax1.set_xlabel('Digit', fontsize=12)
    ax1.set_ylabel('Average Confidence', fontsize=12)
    ax1.set_title('Logistic Regression', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, conf in zip(bars1, lr_conf_by_digit):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{conf:.3f}', ha='center', va='bottom', fontsize=9)

    # XGBoost
    bars2 = ax2.bar(x, xgb_conf_by_digit, width, label='XGB Confidence',
                    color='forestgreen', alpha=0.8, edgecolor='black')
    ax2.axhline(y=0.9, color='red', linestyle='--', label='90% threshold')
    ax2.set_xlabel('Digit', fontsize=12)
    ax2.set_ylabel('Average Confidence', fontsize=12)
    ax2.set_title('XGBoost', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, conf in zip(bars2, xgb_conf_by_digit):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{conf:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('confidence_by_digit.png', dpi=300, bbox_inches='tight')
    print("Saved confidence_by_digit.png")
    plt.show()

def main():
    """Main execution function."""
    # Load data
    X, y = load_mnist_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]}")
    print(f"Test set: {X_test.shape[0]}")

    # Train models
    lr_model, xgb_model = train_models(X_train, y_train, X_test, y_test)

    # Generate visualizations
    visualize_model_comparison(lr_model, xgb_model)
    visualize_prediction_heatmaps(lr_model, xgb_model, X_test, y_test)
    visualize_decision_boundaries(lr_model, xgb_model, X_test, y_test)
    visualize_confidence_by_digit(lr_model, xgb_model, X_test, y_test)
    visualize_misclassified_examples(lr_model, xgb_model, X_test, y_test)

    print("\n" + "="*60)
    print("Interpretability visualization complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - model_comparison.png")
    print("  - prediction_heatmaps.png")
    print("  - confidence_visualization.png")
    print("  - confidence_by_digit.png")
    print("  - misclassification_analysis.png")

if __name__ == "__main__":
    main()
