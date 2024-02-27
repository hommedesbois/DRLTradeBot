import math
import random
import numpy as np
import pandas as pd

from .dataloader import YahooDownloader, LocalDataLoader, BinanceDownloader
from .preprocessor import FeatureEngineer


class ObservationSpace:
    def __init__(self, n):
        self.shape = (n,)

class ActionSpace:
    def __init__(self, n):
        self.n = n
    def sample(self):
        return random.randint(0, self.n - 1)

class Finance:
    def __init__(self, symbol, interval, features_and_scaling, lags, min_performance=0.9, 
                 min_sharpe=0.2, start=0, end=None):
        self.symbol = symbol
        self.interval = interval
        self.features_and_scaling = features_and_scaling
        self.features = [feat_scal[0] for feat_scal in features_and_scaling]
        self.n_features = len(self.features)
        self.lags = lags
        self.grace_period = 128 # must be same as batch_size 
        self.min_performance = min_performance
        self.min_sharpe = min_sharpe
        self.start = start
        self.end = end
        self.observation_space = ObservationSpace(self.lags)
        self.from_file = False
        self.artifical_data = ['sinus'] # add more prefixes for artificial data
        self.crypto_currencies = ['BTCUSDT']
        self.yahoo_downloader = YahooDownloader(self.start, self.end, ticker = self.symbol, interval=self.interval)
        self.binance_downloader = BinanceDownloader(self.start, self.end, ticker = self.symbol, interval=self.interval)
        self.local_data_loader = LocalDataLoader(self.start, self.end, ticker = self.symbol, interval=self.interval)
        self.feature_engineer = FeatureEngineer(self.features_and_scaling)
        self.action_space = ActionSpace(2)
        self._get_data()
        self._prepare_data()
    
    def _get_data(self):

        if self.symbol[0].split('_')[0] in self.artifical_data:
            self.raw = self.local_data_loader.generate_artifical_data(add_noise=True)
            #self.raw = self.file_loader.fetch_data()
        elif self.symbol[0] in self.crypto_currencies:
            self.raw = self.binance_downloader.fetch_data()    
        else:
            self.raw = self.yahoo_downloader.fetch_data()

    def _prepare_data(self):
        self.data = self.feature_engineer.preprocess_data(self.raw)
        self.data['log-ret-no-scale'] = np.log(self.data['close']/self.data['close'].shift())
        self.data = self.data.iloc[1:]
        self.data['dir'] = np.where(self.data['log-ret-no-scale'] > 0, 1, 0)
        self.data['dir'] = self.data['dir'].astype(int)

    def _get_state(self, done): # returns df
        if done: 
            return self.data[self.features].iloc[-self.lags:]
        else:
            return self.data[self.features].iloc[self.bar-self.lags:self.bar]
    
    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self, shift=0):
        self.performance = 1
        self.sharpe = self.min_sharpe
        self.returns = []
        self.bar = self.lags + shift
        state = self.data[self.features].iloc[self.bar-
                        self.lags:self.bar] # state goes to index self.bar-1
        return state.values
    
    def step(self, action):
        correct = action == self.data['dir'].iloc[self.bar]
        ret = self.data['log-ret-no-scale'].iloc[self.bar]
        p_l = abs(ret) if correct else -abs(ret) # long / short 
        #p_l = action * ret # buy only strategy
        daily_return = (math.exp(p_l) - 1) * 100 # in percent
        if correct:
            reward = min(1, daily_return)
        else:
            reward = max(-1, daily_return)
        self.bar += 1 
        self.performance *= math.exp(p_l)
        self.returns.append(math.exp(p_l)-1)
        # use maybe rolling sharpe later
        if self.bar > 2*self.lags:
            self.sharpe = np.mean(self.returns[-self.lags:]) / np.std(self.returns[-self.lags:])
        else:
            self.sharpe = 0
        #reward += max(-1, self.sharpe)
        if self.bar >= len(self.data):
            done = True
        #elif (self.sharpe < self.min_sharpe  and
        #      self.bar >= self.lags + self.grace_period):
        #    done = True
        elif (self.performance < self.min_performance and
              self.bar > self.lags + self.grace_period):
            done = True 
        else:
            done = False
        state = self._get_state(done)
        info = {}
        return state.values, reward, done, info

    def step_val(self, action):
        correct = action == self.data['dir'].iloc[self.bar]
        ret = self.data['log-ret-no-scale'].iloc[self.bar]
        reward = 1 if correct else 0
        p_l = abs(ret) if correct else -abs(ret)
        #p_l = action * ret # only buy strategy 
        self.bar += 1
        self.performance *= math.exp(p_l)
        if self.bar >= len(self.data):
            done = True
        else:
            done = False
        state = self._get_state(done)
        info = {}
        return state.values, reward, done, info


    
    
    