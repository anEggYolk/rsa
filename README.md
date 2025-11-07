# MNIST Logistic Regression with Weight Visualization

This project trains a logistic regression model on the MNIST dataset and visualizes the learned weights for each one-vs-rest classifier.

## Overview

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels. Logistic regression uses a one-vs-rest (OvR) approach for multi-class classification, creating 10 binary classifiers (one for each digit).

## Features

- Loads and preprocesses the MNIST dataset
- Trains a logistic regression model using one-vs-rest strategy
- Evaluates model performance with accuracy and classification report
- Visualizes the learned weights for each digit classifier
- Provides detailed weight statistics

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the script:

```bash
python mnist_logistic_regression.py
```

## Output

The script generates:
- `mnist_weights.png`: Visualization of weights for all 10 classifiers
- `mnist_weights_detailed.png`: Detailed visualization with weight statistics
- Console output showing:
  - Training progress
  - Model accuracy
  - Classification report
  - Weight statistics for each classifier

## Understanding the Weights

Each weight visualization shows a 28x28 matrix corresponding to the 784 input features (pixels). The weights indicate:
- **Positive weights (red)**: Pixel patterns that increase the probability of a digit
- **Negative weights (blue)**: Pixel patterns that decrease the probability of a digit
- **Near-zero weights**: Pixels that don't contribute much to the classification

The weight patterns often resemble the actual digit shapes, as the model learns which pixels are most discriminative for each digit.

## Model Details

- **Algorithm**: Logistic Regression with L2 regularization
- **Multi-class strategy**: One-vs-Rest (OvR)
- **Solver**: lbfgs
- **Training/Test split**: 80/20
- **Data normalization**: Pixel values scaled to [0, 1]
