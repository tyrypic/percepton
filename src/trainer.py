import numpy as np

def one_hot_encode(labels, num_classes):
    # Убедимся, что метки являются целочисленными и подходят для индексации
    labels = labels.astype(int)
    return np.eye(num_classes)[labels]

def train_perceptron(model, X_train, y_train, epochs=100):
    # Проверим, не в one-hot ли уже формате y_train
    if y_train.ndim == 1 or y_train.shape[1] == 1:
        num_classes = np.max(y_train) + 1
        y_train_encoded = one_hot_encode(y_train.flatten(), num_classes)
    else:
        y_train_encoded = y_train  # Если уже в one-hot, используем как есть
    
    model.train(X_train, y_train_encoded, epochs)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)  # Если y_test в one-hot, преобразуем для сравнения
    accuracy = np.mean(predicted_labels == y_test)
    return accuracy

