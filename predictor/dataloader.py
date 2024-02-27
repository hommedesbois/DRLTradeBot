"""Contains methods and classes to collect data from
Yahoo Finance API
"""
from __future__ import annotations
import io, sys, os
from pathlib import Path
#from .enums import *
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import t
from datetime import *
import urllib.request
import zipfile

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

class BinanceDownloader:
  
  BASE_URL = 'https://data.binance.vision/'

  def __init__(self, start_date, end_date, ticker, interval='1d'):
    self.ticker = ticker[0] # to be consistent with other data loaders
    self.interval = interval
    self.start_date = start_date
    self.end_date = end_date
    self.save_to_file = True 


  def fetch_data(self):
    # check if self.interval is supported by binance 
    dates = pd.date_range(start = self.start_date, end=self.end_date, freq="D").to_pydatetime().tolist()
    dates = [date.strftime("%Y-%m-%d") for date in dates]
    raw_df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    i = 1
    for date in dates:
      path = f'data/spot/daily/klines/{self.ticker.upper()}/{self.interval}/'
      file_name = "{}-{}-{}.zip".format(self.ticker.upper(), self.interval, date)
      try:
        daily_raw_df = self._load_file(path, file_name)
      except FileNotFoundError:
        daily_raw_df = self._download_file(path, file_name)
      daily_raw_df = daily_raw_df.iloc[:,:6]
      daily_raw_df.columns = raw_df.columns
      raw_df = pd.concat([raw_df, daily_raw_df], ignore_index=True)
      self._print_progress_bar(i, len(dates), prefix = 'Progress', suffix = 'Complete')
      i += 1

    raw_df['date'] = pd.to_datetime(raw_df['date'], unit='ms')
    return raw_df

  def _download_file(self, base_path, file_name):
    download_path = "{}{}".format(base_path, file_name)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.dirname(current_dir)

    save_path = os.path.join(root_path, base_path)
  
    # in case file AND dir do not exist 
    if not os.path.exists(save_path): 
      Path(save_path).mkdir(parents=True, exist_ok=True)
      #print("Creating save path")
    
    try:
      download_url = urllib.parse.urljoin(self.BASE_URL, download_path) 
      zip_file = urllib.request.urlopen(download_url)
      length = zip_file.getheader('content-length')
      if length:
        length = int(length)
        blocksize = max(4096,length//100)
      
      if self.save_to_file:
        with open(f'{save_path}/{file_name}', 'wb') as out_file:
          dl_progress = 0
          #print("\nFile Download: {}".format(save_path))
          while True:
            buf = zip_file.read(blocksize)   
            if not buf:
              break
            dl_progress += len(buf)
            out_file.write(buf)
            #done = int(50 * dl_progress / length)
            #sys.stdout.write("\r[%s%s]" % ('#' * done, '.' * (50-done)) )    
            #sys.stdout.flush()

        zip_file_path = os.path.join(save_path, file_name)
        #zip_in_memory = io.BytesIO(zip_file.read())
        with zipfile.ZipFile(zip_file_path) as zf:
          csv_file_name = file_name.replace('.zip', '.csv')
          with zf.open(csv_file_name) as csv_file:
            df = pd.read_csv(csv_file)
        
          return df
      else:
        zip_in_memory = io.BytesIO(zip_file.read())
        with zipfile.ZipFile(zip_in_memory) as zf:
          csv_file_name = file_name.replace('.zip', '.csv')
          with zf.open(csv_file_name) as csv_file:
            df = pd.read_csv(csv_file)
        
          return df
    except urllib.error.HTTPError:
      print("\nFile not found: {}".format(download_url))
      pass

  def _load_file(self, base_path, file_name):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.dirname(current_dir)
    save_path = os.path.join(root_path, base_path)
    zip_file_path = os.path.join(save_path, file_name)
    #if not os.path.exists(zip_file_path):
       #print("File not on harddrive, proceeding to download!")
       #print(zip_file_path)
    #   return FileNotFoundError 
    with zipfile.ZipFile(zip_file_path) as zf:
      #  # Since the ZIP and CSV have the same name, derive the CSV name directly
        csv_file_name = file_name.replace('.zip', '.csv')  # Assuming the file extension is .zip
        with zf.open(csv_file_name) as csv_file:
            df = pd.read_csv(csv_file)
        
        return df

  def _print_progress_bar(self, iteration, total, prefix='', suffix='', decimals=0, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix)),
    # Print New Line on Complete
    if iteration == total: 
        print()


    
class LocalDataLoader:

  def __init__(self, start_date: str, end_date: str, ticker=None, interval='1d'):

    self.start_date = start_date
    self.end_date = end_date
    self.ticker = ticker[0]
    self.interval = interval

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