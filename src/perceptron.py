import numpy as np
import pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class MultilayerPerceptron:
    def __init__(self, layers, learning_rate=0.01, activation_function=sigmoid, activation_derivative=sigmoid_derivative):
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation = activation_function
        self.activation_derivative = activation_derivative
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def feedforward(self, inputs):
        activations = [inputs]
        for weight in self.weights:
            net_inputs = np.dot(weight, activations[-1])
            activations.append(self.activation(net_inputs))
        return activations

    def backpropagate(self, activations, targets):
        # Ошибка на выходном слое
        error = targets - activations[-1]
        # Дельта для выходного слоя (произведение поэлементно)
        delta = error * self.activation_derivative(activations[-1])
        # Инициализация списка дельт, начиная с выходного слоя
        deltas = [delta]

        # Обратное распространение для остальных слоев
        for i in range(len(self.weights) - 1, 0, -1):
            delta = deltas[-1].dot(self.weights[i].T) * self.activation_derivative(activations[i])
            deltas.append(delta)

        # Обновление весов
        deltas.reverse()  # Переворачиваем список, чтобы идти от входа к выходу
        for i in range(len(self.weights)):
            layer_input = np.atleast_2d(activations[i])
            delta = np.atleast_2d(deltas[i])
            # Обновление весов
            self.weights[i] += self.learning_rate * delta.T.dot(layer_input).T


    def train(self, inputs, targets, epochs=100):
        for epoch in range(epochs):
            for input_vector, target_vector in zip(inputs, targets):
                activations = self.feedforward(input_vector)
                self.backpropagate(activations, target_vector)

    def predict(self, inputs):
        return self.feedforward(inputs)[-1]

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
