import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(train_path, test_path):
    # Загрузка данных
    train_data = pd.read_csv(train_path, header=None)
    test_data = pd.read_csv(test_path, header=None)

    # Обработка данных
    # Пиксели нормализуются для обучающего и тестового наборов
    X_train = train_data.iloc[:, 1:].values / 255.0
    X_test = test_data.iloc[:, 1:].values / 255.0

    # Метки преобразуются в one-hot кодировку
    y_train = pd.get_dummies(train_data.iloc[:, 0]).values
    y_test = pd.get_dummies(test_data.iloc[:, 0]).values

    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled
