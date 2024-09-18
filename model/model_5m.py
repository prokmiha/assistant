import os
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
import time

class Model5m:
    def __init__(self):
        self.model = None
        self.model_path = "./models/model_5m.keras"  # Используем формат .keras
        self.ticker_list = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]
        
        if not os.path.exists('./model/models'):
            os.makedirs('./models')
        
        if not os.path.exists('./model/results'):
            os.makedirs('./results')        
            
        if not os.path.exists('./model/results'):
            os.makedirs('./results')

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(60, 5)),  # Вход: 60 баров, 5 признаков
            tf.keras.layers.GRU(32, return_sequences=True, activation='tanh'),  # Уменьшенное количество нейронов, активация tanh
            tf.keras.layers.BatchNormalization(),  # Batch Normalization для стабилизации
            tf.keras.layers.GRU(16, activation='tanh'),  # Следующий GRU слой
            tf.keras.layers.Dense(1)  # Прогнозируем цену закрытия
        ])

        # Уменьшенная скорость обучения
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(optimizer=optimizer, loss='mse')
        return model
        
    def initialize_model(self):
        if os.path.exists(self.model_path):
            logging.info("Модель найдена, загружаем...")
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            logging.info("Модель не найдена, создаем и обучаем с нуля...")
            self.model = self.build_model()
            x_train, y_train = self.prepare_initial_data()
            logging.info("Начало обучения модели...")
            self.train(x_train, y_train)
            self.model.save(self.model_path)
            logging.info(f"Модель успешно сохранена по пути {self.model_path}")

    def train(self, x_train, y_train):
        history = self.model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
        self.log_training_results(history)

    def log_training_results(self, history):
        log_path = "./logs/model_5m_log.txt"
        with open(log_path, 'a') as log_file:
            log_file.write("=== Обучение модели ===\n")
            log_file.write(f"Эпохи: {len(history.epoch)}\n")
            for epoch, loss in enumerate(history.history['loss']):
                log_file.write(f"Эпоха {epoch + 1}: Loss = {loss}\n")
            log_file.write("========================\n")
        logging.info(f"Результаты обучения сохранены в {log_path}")

    def update(self):
        logging.info("Обновление модели на новых данных...")
        x_train, y_train = self.prepare_new_data()
        self.train(x_train, y_train)
        self.model.save(self.model_path)
        logging.info(f"Модель обновлена и сохранена по пути {self.model_path}")

    def prepare_initial_data(self):
        x_train = []
        y_train = []
        sequence_length = 60

        logging.info("Подготовка начальных данных для обучения...")
        for symbol in self.ticker_list:
            file_path = f'./ws_data/{symbol}_5m.csv'
            if os.path.exists(file_path):
                logging.info(f"Чтение данных из файла: {file_path}")
                df = pd.read_csv(file_path)
                data = df[['open', 'high', 'low', 'close', 'volume']].values

                for i in range(sequence_length, len(data)):
                    x_train.append(data[i-sequence_length:i])
                    y_train.append(data[i, 3])  # Прогнозируем цену закрытия

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        logging.info("Данные для обучения успешно подготовлены.")
        return x_train, y_train

    def prepare_new_data(self):
        # Добавьте логику для подготовки новых данных, если нужно
        pass

    def predict_next_bars(self, last_data, n_bars=5):
        logging.info(f"Прогнозирование на следующие {n_bars} баров...")
        predictions = []
        for _ in range(n_bars):
            prediction = self.model.predict(last_data[np.newaxis, :])
            predictions.append(prediction[0, 0])  # Прогнозируем цену закрытия
            last_data = np.roll(last_data, -1, axis=0)
            last_data[-1, 3] = prediction  # Обновляем последний бар
        logging.info("Прогноз успешно завершен.")
        return predictions

    def save_predictions(self, predictions, n_bars=5):
        logging.info(f"Сохранение прогноза на следующие {n_bars} баров...")
        with open('./results/data.txt', 'a') as f:
            f.write(f"Прогноз на следующие {n_bars} баров:\n")
            for i, price in enumerate(predictions):
                f.write(f"Бар {i + 1}: {price}\n")
            f.write("=========================\n")
        logging.info("Прогноз успешно сохранен в './results/data.txt'.")

    def visualize_predictions(self, actual, predicted, n_bars=5):
        logging.info("Создание графика прогнозов...")
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(actual)), actual, label="Фактические данные")
        plt.plot(np.arange(len(actual), len(actual) + n_bars), predicted, label="Прогноз")
        plt.xlabel("Бар")
        plt.ylabel("Цена закрытия")
        plt.legend()
        plt.title("Прогноз цены на следующие бары")
        plt.savefig('./results/prediction_plot.png')
        plt.close()
        logging.info("График прогнозов сохранен в './results/prediction_plot.png'.")

    def get_last_data_for_prediction(self):
        symbol = 'BTCUSDT'  # Используем данные по одной монете для примера
        file_path = f'./ws_data/{symbol}_5m.csv'
        df = pd.read_csv(file_path)
        last_data = df[['open', 'high', 'low', 'close', 'volume']].values[-60:]
        logging.info(f"Последние 60 баров для {symbol} успешно загружены для предсказания.")
        return last_data

    def run_prediction(self):
        last_data = self.get_last_data_for_prediction()
        predictions = self.predict_next_bars(last_data, n_bars=5)
        self.save_predictions(predictions, n_bars=5)
        actual_data = self.get_actual_data_for_comparison()
        self.visualize_predictions(actual_data, predictions)

    def get_actual_data_for_comparison(self):
        # Здесь добавьте логику для получения фактических данных для сравнения
        # Для примера можно использовать последние данные из CSV
        return []
