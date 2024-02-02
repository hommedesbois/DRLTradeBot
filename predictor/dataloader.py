"""Contains methods and classes to collect data from
Yahoo Finance API
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import t
from datetime import timedelta

class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker=None, ticker_list=None, interval='1d'):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker
        self.ticker_list = ticker_list
        self.interval = interval 

    def fetch_data(self, proxy=None) -> pd.DataFrame:
        """Fetches data from Yahoo API
        
        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0
        if self.ticker is None and self.ticker_list is not None:
            for tic in self.ticker_list:
                temp_df = yf.download(
                    tic, start=self.start_date, end=self.end_date, interval=self.interval, proxy=proxy
                )
                temp_df["tic"] = tic
                if len(temp_df) > 0:
                    # data_df = data_df.append(temp_df)
                    data_df = pd.concat([data_df, temp_df], axis=0)
                else:
                    num_failures += 1
            if num_failures == len(self.ticker_list):
                raise ValueError("no data is fetched.")
            # reset the index, we want to use numbers as index instead of dates
            data_df = data_df.reset_index()
            try:
                # convert the column names to standardized names
                data_df.columns = [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adjcp",
                    "volume",
                    "tic",
                ]
                # use adjusted close price instead of close price
                data_df["close"] = data_df["adjcp"]
                # drop the adjusted close price column
                data_df = data_df.drop(labels="adjcp", axis=1)
            except NotImplementedError:
                print("the features are not supported currently")
            # create day of the week column (monday = 0)
            data_df["day"] = data_df["date"].dt.dayofweek
            # convert date to standard string format, easy to filter
            data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
            # drop missing data
            data_df = data_df.dropna()
            data_df = data_df.reset_index(drop=True)
            print("Shape of DataFrame: ", data_df.shape)
            # print("Display DataFrame: ", data_df.head())

            data_df = data_df.sort_values(by=["date"]).reset_index(drop=True)

            return data_df

        elif self.ticker is not None and self.ticker_list is None:
            
            data_df = yf.download(self.ticker, start=self.start_date, end=self.end_date, 
                                  interval = self.interval, proxy=proxy)
            
            if len(data_df) == 0:
                raise ValueError("no data is fetched.")
            
            data_df = data_df.reset_index()      
            try:
            # convert the column names to standardized names
                data_df.columns = [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adjcp",
                    "volume",
                ]
                # use adjusted close price instead of close price
                data_df["close"] = data_df["adjcp"]
                # drop the adjusted close price column
                data_df = data_df.drop(labels="adjcp", axis=1)
            except NotImplementedError:
                print("the features are not supported currently")
            # create day of the week column (monday = 0)
            data_df["day"] = data_df["date"].dt.dayofweek
            data_df["month"] = data_df["date"].dt.month
            # convert date to standard string format, easy to filter
            if self.interval == '1d': 
                data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
            # drop missing data
            data_df = data_df.dropna()
            data_df = data_df.reset_index(drop=True)
            print("Shape of DataFrame: ", data_df.shape)
            # print("Display DataFrame: ", data_df.head())

            data_df = data_df.sort_values(by=["date"]).reset_index(drop=True)

            return data_df
        else:
            raise ValueError("Either provide ticker or ticker list")


class LocalDataLoader:

    def __init__(self, start_date: str, end_date: str, ticker=None, interval='1d'):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker[0]
        self.interval = interval

        
    def fetch_data(self) -> pd.DataFrame:    
        path_to_file = f'data/{self.ticker}.csv'
        data = pd.read_csv(path_to_file)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
        data.dropna(inplace=True)
        try:
            # convert the column names to standardized names
            data.columns = [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "volume_usd",
                    "weighted_price",
                ]
        except NotImplementedError:
            print("the features are not supported currently")

        data.drop(labels = ["volume_usd", "weighted_price"], axis=1, inplace=True)
        complete_days =  data.groupby(data['date'].dt.date).filter(lambda x: len(x) == 1440).reset_index(drop=True)

        # Define your desired date range
        desired_dates = [self.start_date, self.end_date]
        desired_dates = pd.to_datetime(desired_dates)

        # Filter for desired dates
        filtered_data = complete_days[(complete_days['date'].dt.date >= desired_dates[0].date()) & 
                                  (complete_days['date'].dt.date <= desired_dates[1].date())]

        # Get unique complete days
        unique_complete_days = filtered_data['date'].dt.date.unique()

        # Randomly select x days or the total number of days if x is greater
        num_days_to_select = min(len(unique_complete_days), self.number_of_days)
        np.random.seed(42)
        selected_days = np.random.choice(unique_complete_days, num_days_to_select, replace=False)

        # Return data for the selected days
        return filtered_data[filtered_data['date'].dt.date.isin(selected_days)]



    def generate_artifical_data(self, base_price = 100, period = 5, amplitude = 15, add_noise=False) -> pd.DataFrame:
        freq = self.interval
        if freq[-1].lower() == 'm':
            freq = freq[:-1]+'T'
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq=freq)
        n_points = len(dates)
        period = n_points / period
        point_arr = np.arange(n_points)
        stock_data = base_price + amplitude * np.sin(2 * np.pi * point_arr / period)
        # create noise with stundet t distribution of degree 5
        if add_noise:
            dof = 5  # Degrees of freedom for t-distribution
            mean_return = 0.1
            std_dev = int(self.ticker.split('_')[-1])

            # Generate returns
            noise = t.rvs(dof, size=n_points)

            # Scale and shift the returns
            scaled_noise = noise * std_dev + mean_return

            stock_data += scaled_noise
        
        df = pd.DataFrame({'close': stock_data}, index=dates)
        df.index.name = 'date'
        df = df.reset_index()
        return df