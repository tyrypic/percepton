multilayer_perceptron_project/
│
├── data/                 # Директория для хранения данных
│   ├── mnist_train.csv   # Обучающий набор данных MNIST
│   ├── mnist_test.csv    # Тестовый набор данных MNIST
│   └── custom_data/      # Собственный набор данных
│       ├── initial_set/  # Исходный набор из 100 изображений
│       └── augmented_set/ # Расширенный набор данных
│
├── models/               # Директория для сохраненных моделей
│   └── trained_model.pkl # Сериализованная обученная модель
│
├── src/                  # Исходный код проекта
│   ├── __init__.py       # Инициализация пакета
│   ├── perceptron.py     # Определение класса многослойного перцептрона
│   ├── data_loader.py    # Скрипты для загрузки и предобработки данных
│   ├── trainer.py        # Методы тренировки и тестирования модели
│   └── utils.py          # Вспомогательные функции и утилиты
│
├── notebooks/            # Jupyter notebooks для экспериментов и демонстраций
│   ├── Experimentation.ipynb
│   └── Results_Analysis.ipynb
│
├── docs/                 # Документация проекта
│   └── project_documentation.md
│
├── requirements.txt      # Зависимости проекта для pip
└── README.md             # Основная информация и инструкции по проекту
