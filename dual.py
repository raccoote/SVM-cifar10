import os
import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.utils import shuffle
#from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import random
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def load_filtered_data(data_dir):
    X_train_data = []
    y_train_labels = []

    for i in range(1, 6):  # 5 data batches in CIFAR-10
        file_path = os.path.join(data_dir, f'data_batch_{i}')
        batch = unpickle(file_path)

        for index, label in enumerate(batch['labels']):
            if label == 0 or label == 5:  # Filter for airplane (0) and dogs (5)
                X_train_data.append(batch['data'][index])
                y_train_labels.append(1 if label == 0 else -1)  # Assign 1 for airplane, -1 for dog

    X_train_data = np.array(X_train_data, dtype=np.float32)
    y_train_labels = np.array(y_train_labels, dtype=np.int64)
    return X_train_data, y_train_labels


class SVM:
    def __init__(self, kernel='linear', degree=2, gamma=None, coef0=0.0, learning_rate=0.001, lambda_param=0.01, epochs=200, C=0.1):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.support_vectors = []
        self.C = C
        self.w = None
        self.b = 0
        
    def _apply_kernel(self, xi, xj):
        if self.kernel == 'linear':
            return np.dot(xi, xj)
        elif self.kernel == 'polynomial_homogeneous':
            return (np.dot(xi, xj)) ** self.degree
        elif self.kernel == 'polynomial_inhomogeneous':
            return (np.dot(xi, xj) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(xi - xj) ** 2)
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * np.dot(xi, xj) + self.coef0)

    def fit(self, X, y):  # dual, working
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)  # Lagrange multipliers

        # Kernel matrix computation
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self._apply_kernel(X[i], X[j])

        for epoch in range(self.epochs):
            print(epoch)
            for idx in range(n_samples):
                decision_function = np.dot(self.alpha * y, kernel_matrix[idx]) - self.b

                # Compute the gradient
                if y[idx] * decision_function < 1:
                    gradient = -y[idx] * kernel_matrix[:, idx]  # Update gradient
                    self.alpha += self.learning_rate * (gradient + self.lambda_param * self.alpha * n_samples)

        # Calculate weights and bias after training
        support_vector_indices = np.where(self.alpha > 0)[0]
        self.support_vectors = X[support_vector_indices]
        self.w = np.dot(self.alpha * y, X)
        self.b = np.mean(y - np.dot(self.w, X.T))

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

data_dir = r'C:\Users\User\Desktop\ai\cifar-10-batches-py'
X, y = load_filtered_data(data_dir)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

start_time = time.time()
svm = SVM(kernel='polynomial_homogeneous')
svm.fit(X_train, y_train)
elapsed_time = time.time() - start_time
print(f"Training time: {elapsed_time} seconds")

# Test predictions and accuracy
print(f"Testing Accuracy: {svm.accuracy(X_test, y_test) * 100:.2f}%")
num_airplanes = np.sum(y_test == 1)
num_dogs = np.sum(y_test == -1)

print("Number of airplanes (label 1):", num_airplanes)
print("Number of dogs (label -1):", num_dogs)
