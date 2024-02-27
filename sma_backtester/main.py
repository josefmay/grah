import numpy as np
import pandas as pd
import requests as req
from dotenv import dotenv_values
import matplotlib.pyplot as plt


config = dotenv_values("../.env")

class SMABacktester(object):

    '''
    A simple backtester on simple moving average strategies
    '''

    def __init__(self, symbol: str, sma1: int, sma2: int, start: str, end: str) -> None:
        '''
        Initialize the backtester object
        then
        Retrieve base data
        '''
        self.symbol = symbol
        self.sma1 = sma1
        self.sma2 = sma2
        self.start = start
        self.end = end
        self.results = None
        self.get_data()
    
    def get_data(self) -> None:
        '''
        Get base data for backtesting
        '''
        headers = {
        'Content-Type': 'application/json'
        }
        url = f"https://api.tiingo.com/tiingo/daily/{self.symbol}/prices?startDate={self.start}&endDate={self.end}&token={config["TIINGO_API_TOKEN"]}"
        try:
            requestResponse = req.get(url, headers=headers)
            metadata = requestResponse.json()
        except:
            print("Error connecting to API and Parsing response")
            return None

        data_df = pd.DataFrame(metadata)
        data_df.dropna(inplace=True)
        data_df['date'] = data_df['date'].apply(lambda x: x[:10])
        data_df.rename(columns={'close':'price'}, inplace=True)
        data_df[f'SMA{self.sma1}'] = data_df['price'].rolling(self.sma1).mean()
        data_df[f'SMA{self.sma2}'] = data_df['price'].rolling(self.sma2).mean()
        data_df.set_index("date", inplace=True)
        data_df['returns'] = np.log(data_df['price']/data_df['price'].shift(1))
        data_df.dropna(inplace=True)
        self.data=data_df

    def run_strategy(self) -> None:
        '''
        Backtest Strategy
        '''
        if self.data is None:
            print('Stock data is non-existent')
            return
        data_df = self.data.copy()
        data_df['position'] = np.where(data_df[f'SMA{self.sma1}']>data_df[f'SMA{self.sma2}'], 1, -1)
        data_df['strategy'] = data_df['position'].shift(1) * data_df['returns']
        data_df.dropna(inplace=True)
        data_df['creturns']  = data_df['returns'].cumsum().apply(np.exp)
        data_df['cstrategy']  = data_df['strategy'].cumsum().apply(np.exp)
        self.results = data_df

    def plot_results(self) -> None:
        '''
        Plots the cumlative performance of the trading strategy compared to regular returns
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy")
        title=f'{self.symbol} | SMA1={self.sma1}, SMA2={self.sma2}'
        self.results[['creturns', 'cstrategy']].plot(title=title, figsize=(10,6))
        plt.show()


if __name__ == "__main__":
    test_object = SMABacktester('aapl', 42, 242, '2014-6-04', '2023-11-09')
    test_object.run_strategy()
    test_object.plot_results()