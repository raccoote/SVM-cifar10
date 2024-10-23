import os
import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import numpy as np



kernel='sigmoid'
optim='primal'
learning_rate=0.01
coef0=1.0
gamma=0.5
lambda_param=0.01
epochs=10

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
            if label == 0 or label == 5:  # filter for airplane (0) and dogs (5)
                X_train_data.append(batch['data'][index])
                y_train_labels.append(1 if label == 0 else -1)  # 1 for airplane, -1 for dog

    X_train_data = np.array(X_train_data, dtype=np.float32)
    y_train_labels = np.array(y_train_labels, dtype=np.int64)
    return X_train_data, y_train_labels


class SVM:
    def __init__(self, kernel='linear', degree=3, gamma=0.5, coef0=1.0, learning_rate=0.001, lambda_param=0.01, epochs=3, C=0.1):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.C = C
        self.w = None
        self.b = 0
        
    def apply_kernel(self, xi, xj):
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

    def hinge_loss(self, X, y):
        return np.maximum(0, 1 - y * (np.dot(X, self.w) - self.b)).mean() + self.lambda_param * np.dot(self.w, self.w)

    def fit(self, X, y):   # primal non linear kernels (sgd)
        self.X_train = X
        num_samples, _ = X.shape
        self.w = np.zeros(num_samples)
        self.b = 0
        for epoch in range(self.epochs):   
            print(epoch)
            for idx in range(num_samples):
                kernel_values = [self.apply_kernel(X[idx], sample) for sample in X]
                decision = np.dot(self.w, kernel_values) - self.b
                condition = y[idx] * decision >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    current_kernel_values = np.array(kernel_values)
                    self.w -= self.learning_rate * (
                        2 * self.lambda_param * self.w - y[idx] * current_kernel_values
                    )
                    self.b -= self.learning_rate * y[idx]
                       

    def predict(self, X):
        predictions = []
        for idx in range(X.shape[0]):
            kernel_values = [self.apply_kernel(X[idx], sample) for sample in self.X_train]
            decision = np.dot(self.w, kernel_values) - self.b
            predictions.append(np.sign(decision))
        return np.array(predictions)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    

    

data_dir = r'C:\Users\User\Desktop\ai\cifar-10-batches-py'
X, y = load_filtered_data(data_dir)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

start_time = time.time()
svm = SVM(kernel='sigmoid')
svm.fit(X_train, y_train)
elapsed_time = time.time() - start_time
print(f"Training time: {elapsed_time} seconds")
print(f"Training Accuracy: {svm.accuracy(X_train, y_train) * 100:.2f}%")
print(f"Testing Accuracy: {svm.accuracy(X_test, y_test) * 100:.2f}%")
num_airplanes = np.sum(y_test == 1)
num_dogs = np.sum(y_test == -1)

print("Number of airplanes (label 1):", num_airplanes)
print("Number of dogs (label -1):", num_dogs)
