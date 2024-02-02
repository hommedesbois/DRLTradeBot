from __future__ import annotations

from multiprocessing.sharedctypes import Value

import re
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as sdf
from .utils import min_max_scaling, z_scaling, no_scaling

class UserStatsCalculator:
    def __init__(self):
        self.features_dict = {
            'delta_sma': self._compute_delta_sma,
            'ratio_sma': self._compute_ratio_sma,
            'hour_otd': self._compute_hour,
            'candle_body': self._compute_candle_body,
            'upper_wick': self._compute_upper_wick,
            'lower_wick': self._compute_lower_wick
        }

    def calculate(self, data, feature):
        """ 
        Calculates user defined features. Requires that 
        the window sizes go to the end of the name string 
        """
        parts = feature.split('_')
        for i in range(1, len(parts) + 1):
            feature_key = '_'.join(parts[:i])
            if feature_key in self.features_dict:
                func = self.features_dict[feature_key]
                temp_feature = func(data, parts[i:], feature)
                return temp_feature
        raise KeyError("Unsupported feature")

    def _compute_delta_sma(self, data, params, feature):
                
        if len(params) != 2:
            raise ValueError("delta_sma requires a long and a short window")
        shorter_window = int(params[0])
        longer_window = int(params[1])
        try:
            sma1 = data[f'close_{shorter_window}_sma']
        except KeyError:
            sma1 = data['close'].rolling(window=shorter_window).mean()
        try:
            sma2 = data[f'close_{longer_window}_sma']
        except KeyError:
            sma2 = data['close'].rolling(window=longer_window).mean()

        df = pd.DataFrame({'date': data['date'], feature: sma1 - sma2})

        return df

    def _compute_ratio_sma(self, data, params, feature):
                
        if len(params) != 2:
            raise ValueError("delta_ratio requires a long and a short window")
        shorter_window = int(params[0])
        longer_window = int(params[1])
        try:
            sma1 = data[f'close_{shorter_window}_sma']
        except KeyError:
            sma1 = data['close'].rolling(window=shorter_window).mean()
        try:
            sma2 = data[f'close_{longer_window}_sma']
        except KeyError:
            sma2 = data['close'].rolling(window=longer_window).mean()

        df = pd.DataFrame({'date': data['date'], feature: sma1/sma2 - 1})

        return df

    def _compute_hour(self, data, params, feature):
        df = pd.DataFrame({'date': data['date'], 'hour_otd': data['date'].dt.hour})
        return df 

    def _compute_candle_body(self, data, params, feature):
        df = pd.DataFrame({'date': data['date'], feature: data['close'] - data['open']})
        return df

    def _compute_upper_wick(self, data, params, feature):
        df = pd.DataFrame({'date': data['date'], feature: data['high'] - data[['open', 'close']].max(axis=1)})
        return df

    def _compute_lower_wick(self, data, params, feature):
        df = pd.DataFrame({'date': data['date'], feature: data['low'] + data[['open', 'close']].min(axis=1)})
        return df


class FeatureEngineer:
    """ Provides methods for preprocessing the stock price data """

    def __init__(
        self,
        features,
    ):
        self.features = features
        self.user_stats_calculator = UserStatsCalculator()
        self.scaling_dict = {
            'minmax' : min_max_scaling,
            'z' : z_scaling,
            'no': no_scaling
        }

    def preprocess_data(self, data):
        """main method to do the feature engineering"""
        
        def _extract_numbers(s):
            nums = re.findall(r'\d+', s) 
            return [int(num) for num in nums]    
       
        lag = 1
        df = data.copy()
        
        features = self.features.copy()

        for feature, scaling in features:
            if feature == 'volume':
                scaling_func = self.scaling_dict[scaling]
                df = scaling_func(df,feature)
                print(f"Successfully added {feature} to DataFrame") 
            else:
                windows = _extract_numbers(feature)
                if len(windows) > 0:
                    lag = max(lag, max(windows))
                try:
                    new_feature_df = self._stockstats_calculate(data, feature)
                    scaling_func = self.scaling_dict[scaling]
                    new_feature_df = scaling_func(new_feature_df, feature) 
                    print(f"Successfully added {feature} to DataFrame") 
                    print()
                except KeyError:   
                
                    try:
                        new_feature_df = self.user_stats_calculator.calculate(data, feature)
                        scaling_func = self.scaling_dict[scaling]
                        new_feature_df = scaling_func(new_feature_df, feature) 
                        print(f"Successfully added {feature} to DataFrame") 
                        print()  
                    except KeyError:
                        print(f"{feature} not supported")
                        continue

                df = df.merge(new_feature_df, on="date", how="left")
    
        df = df.ffill().bfill()
        
        length = len(df) - lag
        return df.tail(length).reset_index(drop=True)

    def _stockstats_calculate(self, data, feature):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        """
        stock = sdf.retype(data.copy())
        
        try:
            temp_feature = stock[[feature]]
        except KeyError:
            raise  # Re-raise the KeyError if the feature is not in stockstats
        except UserWarning as e:
            # Convert UserWarning to KeyError for uniform handling
            raise KeyError(str(e))  
        
        temp_feature.reset_index(inplace=True)
        temp_feature = pd.DataFrame(temp_feature)
        
        if feature not in temp_feature.columns:
            raise KeyError(f"Feature {feature} not found after using stockstats")

        return temp_feature[["date", feature]]

