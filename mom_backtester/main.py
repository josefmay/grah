import numpy as np
import pandas as pd
import requests as req
from dotenv import dotenv_values
import matplotlib.pyplot as plt
from scipy.optimize import brute


config = dotenv_values("../.env")

class MOMBacktester(object):

    '''
    A simple backtester on momentum strategies

    Attributes
    ==========
    symbol: str
        Stock ticker with which to work
    amount: int
        amount invested in dollars
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
        runs the backtest for the Momentum-based strategy
    plot_results:
        plot the performance of the strategy against the basic investment
    set_parameters:
        sets Momentum values
    update_run:
        updates Momentum values, runs strategy with updated values
    optimize_parameters:
        brute force optimization to find optimal Momentum values
        
    '''

    def __init__(self, symbol: str, amount: int, start: str, end: str) -> None:
        '''
        Initialize the backtester object
        then
        Retrieve base data
        '''
        self.symbol = symbol
        self.amount = amount
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
        data_df.set_index("date", inplace=True)
        data_df['returns'] = np.log(data_df['price']/data_df['price'].shift(1))
        data_df.dropna(inplace=True)
        self.data=data_df

    def run_strategy(self, momentum=1) -> None:
        '''
        Backtest Strategy

        Parameters
        ==========
        momentum: int
            number range to assess momentum
        '''
        self.momentum = momentum
        if self.data is None:
            print('Stock data is non-existent')
            return
        data_df = self.data.copy()
        data_df['position'] = np.sign(data_df['price'].rolling(momentum).mean())
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
        title=f'{self.symbol} | Momentum = {self.momentum}'
        self.results[['creturns', 'cstrategy']].plot(title=title, figsize=(10,6))
        plt.show()

    
if __name__ == "__main__":
    test_object = MOMBacktester('ivv', 1000, '2014-6-04', '2023-11-09')
    print(test_object.run_strategy(3))
    test_object.plot_results()