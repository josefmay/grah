import numpy as np
import pandas as pd
import requests as req
from dotenv import dotenv_values
import matplotlib.pyplot as plt
import logging 

config = dotenv_values("../.env")


class LRBackTester(object):
    
    def __init__(self, symbol: str, start: str, end: str) -> None:
        '''
        Initialize the backtester object
        then
        Retrieve base data
        '''
        self.symbol = symbol
        self.start = start
        self.end = end
        self.results = None
        self.get_data()

    # Request api for data
    def req_api(self):
        headers = {
        'Content-Type': 'application/json'
        }
        url = f"https://api.tiingo.com/tiingo/daily/{self.symbol}/prices?startDate={self.start}&endDate={self.end}&token={config['TIINGO_API_TOKEN']}"
        try:
            requestResponse = req.get(url, headers=headers)
            metadata = requestResponse.json()
        except:
            logging.error("Error connecting to API and Parsing response")
            return None
        return metadata

    def get_data(self) -> None:
        '''
        Get base data for backtesting
        '''
        metadata = self.req_api()
        if metadata == None:
            logging.error("couldn't fetch data from TIINGO")
            return None

        data_df = pd.DataFrame(metadata)
        # data_df.dropna(inplace=True)

        # Trim Date to proper format
        data_df['date'] = data_df['date'].apply(lambda x: x[:10])
        data_df.rename(columns={'close':'price'}, inplace=True)
        # data_df.set_index("date", inplace=True)

        # Calculate logarithmic returns per day of basic investment
        data_df['returns'] = np.log(data_df['price']/data_df['price'].shift(1))
        data_df.dropna(inplace=True)
        self.data=data_df

    def select_data(self, start, end):
        data=self.data[(self.data.index >= start) & (self.data.index<=end)].copy()
        return data

    def prepare_lags(self, start, end):
        # data = self.select_data(start, end)
        data=self.data

        '''Shift the returns to match with proper date data'''
        self.cols=[]
        for lag in range(1, self.lags+1):
            col=f'lag{lag}'
            data[col]=data['returns'].shift(lag)
            self.cols.append(col)
        data.dropna(inplace=True)
        self.lagged_data = data

    def fit_model(self, start, end):
        '''Implements the regression step'''
        self.prepare_lags(start, end)
        x_vector=np.linalg.lstsq(self.lagged_data[self.cols], 
                                np.sign(self.lagged_data['returns']),
                                rcond=None)[0]
        self.x_vec=x_vector

    def run_strategy(self, train_start, train_end, predict_start, predict_end, lags=3):
        self.lags=lags
        self.fit_model(train_start, train_end)
        # self.results = self.select_data(predict_start, predict_end).iloc[lags:]
        self.results = self.data.iloc[lags:]

        self.prepare_lags(predict_start, predict_end)
        prediction = np.sign(np.dot(self.lagged_data[self.cols], self.x_vec))
        self.results['prediction']=prediction
        self.results['strategy']=self.results['prediction']*self.results['results']

        # Base returns of investment
        self.results['creturns']  = self.results['returns'].cumsum().apply(np.exp)
        # Base returns of investment
        self.results['cstartegy']  = self.results['strategy'].cumsum().apply(np.exp)

        aperf = self.results['cstrategy'].iloc[-1]
        operf = self.results['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2)

    def plot_results(self) -> None:
        '''
        Plots the cumlative performance of the trading strategy compared to regular returns
        '''
        if self.results is None:
            logging.warning("No results to plot yet. Run a strategy")
        title=f'{self.symbol} | Predictions'
        self.results[['creturns', 'cstrategy']].plot(title=title, figsize=(10,6))
        plt.show()
    

if __name__ == '__main__':
    lrbt = LRBackTester('ivv', '2014-6-04', '2023-11-09')
    print(lrbt.run_strategy('2014-6-04', '2023-11-09','2014-6-04', '2023-11-09', 5))
    