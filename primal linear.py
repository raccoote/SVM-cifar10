import os
import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random




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
            if label == 0 or label == 5:  # filter for airplane (0) and dog (5)
                X_train_data.append(batch['data'][index])
                y_train_labels.append(1 if label == 0 else -1)  # label 1 for airplane, label -1 for dog

    X_train_data = np.array(X_train_data, dtype=np.float32)
    y_train_labels = np.array(y_train_labels, dtype=np.int64)
    return X_train_data, y_train_labels


class MySVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=100, C=1.0):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.C = C
        self.w = None
        self.b = 0

    def hinge_loss(self, X, y):
        return np.maximum(0, 1 - y * (np.dot(X, self.w) - self.b)).mean() + self.lambda_param * np.dot(self.w, self.w)
                    
    def fit(self, X, y):  # linear primal(sgd) -> 80%
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y[idx])
                    )
                    self.b -= self.learning_rate * y[idx]

        # training accuracy 
        correct_predictions = 0
        for idx, x_i in enumerate(X):
            prediction = np.dot(x_i, self.w) - self.b
            if prediction >= 0 and y[idx] == 1:
                correct_predictions += 1
            elif prediction < 0 and y[idx] == -1:
                correct_predictions += 1

        accuracy = correct_predictions / num_samples
        print(f"Training Accuracy: {accuracy * 100:.2f}%")

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
svm = MySVM()
svm.fit(X_train, y_train)
elapsed_time = time.time() - start_time
print(f"Training time: {elapsed_time} seconds")

print(f"Testing Accuracy: {svm.accuracy(X_test, y_test) * 100:.2f}%")
num_airplanes = np.sum(y_test == 1)
num_dogs = np.sum(y_test == -1)

print("Number of airplanes (label 1):", num_airplanes)
print("Number of dogs (label -1):", num_dogs)
