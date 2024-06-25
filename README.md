# ANN

# Neural Network Implementation in Python

This repository contains Python scripts that demonstrate the implementation of a simple neural network from scratch using basic concepts such as the sigmoid activation function, neuron class, mean squared error (MSE) loss function, and training using gradient descent.

---

## Sigmoid Activation Function

The `sigmoid` function implements the sigmoid activation function, which is defined as:

```python
import numpy as np

def sigmoid(x):
    """Sigmoid activation function: f(x) = 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-x))
