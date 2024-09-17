import os
import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
import ccxt

class BinanceDataCollector:
    def __init__(self):
        self.data_path = "./ws_data"
        self.exchange = ccxt.binance()
        self.ticker_list = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]
        self.intervals = {
            "5m": timedelta(minutes=5),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1)
        }

    async def start(self):
        await self.check_files_and_fill_missing_data()

        while True:
            await self.check_and_collect_data()
            await asyncio.sleep(5 * 60)

    async def check_files_and_fill_missing_data(self):
        for symbol in self.ticker_list:
            for interval, delta in self.intervals.items():
                file_path = os.path.join(self.data_path, f"{symbol.replace('/', '')}_{interval}.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
                    current_time = datetime.utcnow()

                    if current_time - last_timestamp > delta:
                        await self.fill_missing_data(symbol, interval, last_timestamp, current_time, delta)
                else:
                    await self.create_new_file(symbol, interval)

    async def fill_missing_data(self, symbol, interval, last_timestamp, current_time, delta):"
        logging.info(f"Запрос пропущенных данных для {symbol} на таймфрейме {interval} с {last_timestamp} до {current_time}")
        missing_bars = self.calculate_missing_bars(last_timestamp, current_time, delta)
        klines = await self.get_historical_data(symbol, interval, custom_limit=int(missing_bars))

        if klines:
            df_new = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
            df_new.to_csv(f"{self.data_path}/{symbol.replace('/', '')}_{interval}.csv", mode='a', header=False, index=False)

    def calculate_missing_bars(self, last_timestamp, current_time, delta):
        return (current_time - last_timestamp) // delta

    async def create_new_file(self, symbol, interval):
        logging.info(f"Создание нового файла для {symbol} на таймфрейме {interval}")
        klines = await self.get_historical_data(symbol, interval)
        if klines:
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df.to_csv(f"{self.data_path}/{symbol.replace('/', '')}_{interval}.csv", index=False)

    async def get_historical_data(self, symbol, interval, limit=1000, custom_limit=None):
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
        current_time = datetime.utcnow()

        logging.info(f"Запись данных для 5m в {current_time}")
        await self.collect_data_for_interval('5m')

        if current_time.minute in [30, 31, 32, 33, 34, 0, 1, 2, 3, 4] :
            logging.info(f"Запись данных для 30m в {current_time}")
            await self.collect_data_for_interval('30m')

        if current_time.minute in [0, 1, 2, 3, 4]:
            logging.info(f"Запись данных для 1h в {current_time}")
            await self.collect_data_for_interval('1h')

    async def collect_data_for_interval(self, interval):
        for symbol in self.ticker_list:
            await self.fetch_and_store(symbol, interval)

    async def fetch_and_store(self, symbol, interval):
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
