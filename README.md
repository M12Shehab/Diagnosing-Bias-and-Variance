# Machine Learning Algorithms Repository

## Hardware

The machine learning models in this repository were trained using NVIDIA GPUs GTX-970 and A100 in the cloud.

For detailed instructions on running the code and reproducing the results, please refer to the documentation in the repository.

## Overview

This repository contains implementations and examples of various machine-learning algorithms and concepts. The focus is on understanding and applying these algorithms in a practical setting.

## Packages

The repository uses the following Python packages:

- NumPy
- pandas
- scikit-learn
- TensorFlow

## Polynomial Regression

The repository includes an implementation of Polynomial Regression, a form of linear regression where the relationship between the independent variable x and the dependent variable y is modeled as an nth degree polynomial.

## Bias and Variance

The concept of Bias and Variance is explained and demonstrated using examples in the repository. Bias is the error introduced by approximating a real-world problem, which may be complex, by a simple model. Variance is the error introduced by using a model that is too sensitive to small fluctuations in the training data.

## Tuning Regularization

Regularization techniques such as L1 and L2 regularization are implemented and explained in the repository. Regularization is used to prevent overfitting by adding a penalty term to the loss function.

## Evaluating a Deep Learning Algorithm (Neural Network)

### Dataset

The repository includes a dataset for training and testing the neural network. The dataset is preprocessed and split into training and testing sets.

### Model Complexity

The neural network is trained with different levels of complexity (number of layers, number of neurons per layer) to understand the impact of model complexity on performance.

### Complex Model Results

Results of training the neural network with a complex architecture are provided, including classification error metric.

### Simple Model Results

Results of training the neural network with a simple architecture are provided, compared to the complex model results.

### Regularized-Model Results

Results of training the neural network with regularization (L1, L2) are provided, compared to the results of the complex and simple models.

## Compared Results of All Three Models

The results of all three models (complex, simple, and regularized) are compared regarding performance metrics and computational efficiency.

## Source code:
[Jupyter Notebook](ApplyingMachineLearning.ipynb)
