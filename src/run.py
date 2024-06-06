import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class MultilayerPerceptron:
    def __init__(self, layers, learning_rate=0.01, activation_function=sigmoid):
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.weights = [np.random.randn(y, x + 1) for x, y in zip(layers[:-1], layers[1:])]

    def add_bias(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def feedforward(self, X):
        activations = [X]
        for weights in self.weights:
            X = self.add_bias(X)
            X = self.activation_function(np.dot(X, weights.T))
            activations.append(X)
        return activations

    def backpropagation(self, activations, y):
        errors = [y - activations[-1]]
        deltas = [errors[-1] * sigmoid_derivative(activations[-1])]
        for i in range(len(activations) - 2, 0, -1):
            error = deltas[-1].dot(self.weights[i][:, :-1])
            delta = error * sigmoid_derivative(activations[i])
            errors.append(error)
            deltas.append(delta)
        deltas.reverse()
        for i in range(len(self.weights)):
            layer_input = self.add_bias(activations[i])
            self.weights[i] += self.learning_rate * deltas[i].T.dot(layer_input)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            activations = self.feedforward(X)
            self.backpropagation(activations, y)

    def predict(self, X):
        return self.feedforward(X)[-1]

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            mlp = pickle.load(f)
        self.__dict__.update(mlp.__dict__)

    def loadTrainDataFromCsv(self, path):
        train_data = pd.read_csv(path)
        X = train_data.iloc[:, 1:].values / 255.0
        y = train_data.iloc[:, 0].values
        y = np.eye(10)[y]
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=0.05)
        self.x_train = normalize(self.x_train)
        self.x_valid = normalize(self.x_valid)

    def loadTestDataFromCsv(self, path):
        test_data = pd.read_csv(path)
        X = test_data.iloc[:, 1:].values / 255.0
        y = test_data.iloc[:, 0].values
        y = np.eye(10)[y]
        self.x_test, self.y_test = X, y


# Создание экземпляра перцептрона с 2 скрытыми слоями по 128 и 64 нейронов
mlp = MultilayerPerceptron(layers=[784, 128, 64, 10], learning_rate=0.01, activation_function=sigmoid)

# Загрузка обучающих данных
mlp.loadTrainDataFromCsv('data/mnist_train.csv')

# Обучение модели
mlp.train(mlp.x_train, mlp.y_train, epochs=10)

# Загрузка тестовых данных
mlp.loadTestDataFromCsv('data/mnist_test.csv')

# Получение предсказаний для тестовых данных
predictions = mlp.predict(mlp.x_test)

# Подсчет точности
accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(mlp.y_test, axis=1))
print(f"Точность модели: {accuracy * 100:.2f}%")

# Сохранение модели
mlp.save('trained_mlp_model.pkl')
