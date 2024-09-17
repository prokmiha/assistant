import os
import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
import ccxt

class BinanceDataCollector:
    def __init__(self):
        self.data_path = "./ws_data"  # Путь к папке с данными
        self.exchange = ccxt.binance()  # Инициализация биржи Binance через CCXT
        self.ticker_list = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]  # Список тикеров
        self.intervals = {
            "5m": timedelta(minutes=5),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1)
        }  # Таймфреймы

    async def start(self):
        # Перед запуском проверяем файлы и заполняем пропуски
        await self.check_files_and_fill_missing_data()

        # Основной цикл сбора данных каждые 5 минут
        while True:
            await self.check_and_collect_data()
            await asyncio.sleep(5 * 60)  # Сон на 5 минут перед следующим циклом

    async def check_files_and_fill_missing_data(self):
        """
        Проверка наличия файлов для каждого тикера и таймфрейма.
        Если файл существует, проверяем пропуски и заполняем.
        Если файла нет, создаём новый и запрашиваем последние 1000 баров.
        """
        for symbol in self.ticker_list:
            for interval, delta in self.intervals.items():
                file_path = os.path.join(self.data_path, f"{symbol.replace('/', '')}_{interval}.csv")
                if os.path.exists(file_path):
                    # Проверяем последнюю запись в файле
                    df = pd.read_csv(file_path)
                    last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])  # Последняя запись
                    current_time = datetime.utcnow()

                    # Рассчитываем количество пропущенных баров
                    if current_time - last_timestamp > delta:
                        await self.fill_missing_data(symbol, interval, last_timestamp, current_time, delta)
                else:
                    # Если файла нет, создаём новый и запрашиваем последние 1000 баров
                    await self.create_new_file(symbol, interval)

    async def fill_missing_data(self, symbol, interval, last_timestamp, current_time, delta):
        """
        Запрашиваем и заполняем пропущенные данные, подстраиваясь под таймфрейм.
        """
        logging.info(f"Запрос пропущенных данных для {symbol} на таймфрейме {interval} с {last_timestamp} до {current_time}")
        missing_bars = self.calculate_missing_bars(last_timestamp, current_time, delta)
        klines = await self.get_historical_data(symbol, interval, custom_limit=int(missing_bars))

        if klines:
            df_new = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')  # Преобразуем время
            df_new.to_csv(f"{self.data_path}/{symbol.replace('/', '')}_{interval}.csv", mode='a', header=False, index=False)

    def calculate_missing_bars(self, last_timestamp, current_time, delta):
        """
        Рассчитывает количество пропущенных баров.
        """
        return (current_time - last_timestamp) // delta

    async def create_new_file(self, symbol, interval):
        """
        Запрашиваем данные за последние 1000 баров и создаём новый CSV файл.
        """
        logging.info(f"Создание нового файла для {symbol} на таймфрейме {interval}")
        klines = await self.get_historical_data(symbol, interval)
        if klines:
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df.to_csv(f"{self.data_path}/{symbol.replace('/', '')}_{interval}.csv", index=False)

    async def get_historical_data(self, symbol, interval, limit=1000, custom_limit=None):
        """
        Получение исторических данных через CCXT.
        """
        try:
            if custom_limit:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=interval, limit=custom_limit)
            else:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
            return ohlcv
        except Exception as e:
            logging.error(f"Ошибка при запросе исторических данных для {symbol}: {e}")
            return []

    async def check_and_collect_data(self):
        """
        Проверяем текущее время и записываем данные для соответствующих таймфреймов.
        """
        current_time = datetime.utcnow()
        logging.info(f"current_time - {current_time}")
        logging.info(f"current_time.minute - {current_time.minute}")
        # Записываем данные для 5-минутного таймфрейма каждые 5 минут
        logging.info(f"Запись данных для 5m в {current_time}")
        await self.collect_data_for_interval('5m')

        # Записываем данные для 30-минутного таймфрейма каждые 30 минут
        if current_time.minute in [30, 31, 32, 33, 34, 0, 1, 2, 3, 4] :
            logging.info(f"Запись данных для 30m в {current_time}")
            await self.collect_data_for_interval('30m')

        # Записываем данные для часового таймфрейма каждый час
        if current_time.minute in [0, 1, 2, 3, 4]:
            logging.info(f"Запись данных для 1h в {current_time}")
            await self.collect_data_for_interval('1h')

    async def collect_data_for_interval(self, interval):
        """
        Запрашиваем данные через CCXT для каждого символа и таймфрейма.
        """
        for symbol in self.ticker_list:
            await self.fetch_and_store(symbol, interval)

    async def fetch_and_store(self, symbol, interval):
        """
        Запрашиваем последний бар через CCXT и сохраняем его в CSV.
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=interval, limit=1)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

            file_path = f"{self.data_path}/{symbol.replace('/', '')}_{interval}.csv"
            df.to_csv(file_path, mode='a', header=False, index=False)
            logging.info(f"Данные для {symbol} на интервале {interval} успешно записаны.")
        except Exception as e:
            logging.error(f"Ошибка при запросе данных для {symbol} на интервале {interval}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = BinanceDataCollector()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(collector.start())
