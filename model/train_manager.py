from model_5m import Model5m
from model_30m import Model30m
from model_1h import Model1h
import logging

class TrainManager:
    def __init__(self):
        # Инициализация моделей для каждого таймфрейма
        self.model_5m = Model5m()
        self.model_30m = Model30m()
        self.model_1h = Model1h()

    def initialize_models(self):
        """
        Проверяем наличие файлов моделей, если не найдены, запускаем первичное обучение.
        """
        logging.info("Инициализация модели для 5m...")
        self.model_5m.initialize_model()

        logging.info("Инициализация модели для 30m...")
        self.model_30m.initialize_model()

        logging.info("Инициализация модели для 1h...")
        self.model_1h.initialize_model()

    def update_all(self):
        """
        Обновление всех моделей на новых данных
        """
        logging.info("Updating 5m model...")
        self.model_5m.update()

        logging.info("Updating 30m model...")
        self.model_30m.update()

        logging.info("Updating 1h model...")
        self.model_1h.update()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    manager = TrainManager()

    # Инициализируем модели (если нужно, обучаем в первый раз)
    manager.initialize_models()

    # Можно запланировать обновление моделей
    # manager.update_all()
