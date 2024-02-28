import numpy as np
import pandas as pd
import requests as req
from dotenv import dotenv_values
import matplotlib.pyplot as plt
from scipy.optimize import brute


config = dotenv_values("../.env")

class SMABacktester(object):

    '''
    A simple backtester on simple moving average strategies

    Attributes
    ==========
    symbol: str
        Stock ticker with which to work
    SMA1: int
        shorter time window
    SMA2: int
        longer time window
    start: int
        start date for analysis range
    end: int
        end date for analysis range
    data: pandas DataFrame
        formatted base dataset
    results: pandas DataFrame
        results from backtesting trading strategy

    Methods
    =======
    get_data:
        grabs the basic data
    run_strategy:
        runs the backtest for the SMA-based strategy
    plot_results:
        plot the performance of the strategy against the basic investment
    set_parameters:
        sets SMA values
    update_run:
        updates SMA values, runs strategy with updated values
    optimize_parameters:
        brute force optimization to find optimal SMA values
        
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
        url = f"https://api.tiingo.com/tiingo/daily/{self.symbol}/prices?startDate={self.start}&endDate={self.end}&token={config['TIINGO_API_TOKEN']}"
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
        data_df['SMA1'] = data_df['price'].rolling(self.sma1).mean()
        data_df['SMA2'] = data_df['price'].rolling(self.sma2).mean()
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
        data_df['position'] = np.where(data_df['SMA1']>data_df['SMA2'], 1, 0)
        data_df['strategy'] = data_df['position'].shift(1) * data_df['returns']
        data_df['creturns']  = data_df['returns'].cumsum().apply(np.exp)
        data_df['cstrategy']  = data_df['strategy'].cumsum().apply(np.exp)
        self.results = data_df

        gross_perf = data_df['cstrategy'].iloc[-1]
        reg_perf = data_df['creturns'].iloc[-1]

        return(round(gross_perf, 2), round(reg_perf, 2))



    def plot_results(self) -> None:
        '''
        Plots the cumlative performance of the trading strategy compared to regular returns
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy")
        title=f'{self.symbol} | SMA1={self.sma1}, SMA2={self.sma2}'
        self.results[['creturns', 'cstrategy']].plot(title=title, figsize=(10,6))
        plt.show()

    def set_parameters(self, sma1=None, sma2=None) -> None:
        '''
        Updates SMA parameters and corresponding data in set

        Parameters
        ==========
        SMA1, SMA2: int
            new SMAs
        '''
        if sma1 is not None:
            self.sma1=sma1
            self.data['SMA1'] = self.data['price'].rolling(self.sma1).mean()
        if sma2 is not None:
            self.sma2=sma2
            self.data['SMA2'] = self.data['price'].rolling(self.sma2).mean()

    def update_run(self, sma):
        '''
        Updates SMA params and runs the strategy
        Returns negative performance (for minimazation algo)

        Parameters
        ==========
        SMA: tuple
            SMA parameter tuple
        '''
        self.set_parameters(int(sma[0]), int(sma[1]))
        return -self.run_strategy()[0]

    def optimize_parameters(self, sma1_range, sma2_range):
        '''
        Finds global maximum given a range of SMA parameters

        Parameters
        ==========
        SMA1_RANGE, SMA2_RANGE: tuple
            tuples of the form (start, end, step size)
        '''
        optimal = brute(self.update_run, (sma1_range, sma2_range), finish=None)
        return (optimal, -self.update_run(optimal))

if __name__ == "__main__":
    test_object = SMABacktester('aapl', 42, 264, '2014-6-04', '2023-11-09')
    print(test_object.run_strategy())
    test_object.set_parameters(50, 200)
    print(test_object.run_strategy())
    print(test_object.optimize_parameters((30,56,4), (200, 300, 4)))
    test_object.plot_results()