import pandas as pd
from ta.momentum import RSIIndicator
from scipy.stats import linregress
import numpy as np
from configparser import ConfigParser
from os import path,makedirs
from config.Read_Config_file import Read_Config_file


class RawData_Weighting_OHLC:
    def __init__(self,trade_history:pd=None):
        if trade_history is None:
            raise ValueError("There is no Data passed, this is an empty dataset")
        self.trade_history=trade_history
        config=Read_Config_file()
        self.trade_win_criteria=config.get_section('train_trade_win_criteria_column')

    def generate_clean_data(self):
        def compute_slope(nump_array,window_width=2):
            rolling_data=nump_array.rolling(window=window_width)
            x=np.arange(window_width)
            slopes=[]
            for y in rolling_data:
                slope=linregress(x,y.to_numpy())[0]
                slopes.append(int(slope))
            return slopes

        # Calculate gains on Certain window, and push them up to be on the same line the model evaludating
        shift_window=self.trade_win_criteria.getint('gain_criteria_win_window')

        self.trade_history['gain'] = self.trade_history['close'] - self.trade_history['open']
        self.trade_history['gain_shifted_interval'] = (self.trade_history['gain'].rolling(window=shift_window, min_periods=1).sum())
        self.trade_history['gain_shifted_interval'] = (self.trade_history['gain_shifted_interval'].shift(-1*shift_window)).fillna(0.0)
        self.trade_history['action'] = (self.trade_history['gain_shifted_interval'] > self.trade_win_criteria.getint('postive_gain_threshold')).astype(int)
        
        # RSI calculation
        rsi = RSIIndicator(close=self.trade_history['close'], window=14).rsi()
        self.trade_history['RSI'] = rsi
        self.trade_history['RSI']=self.trade_history['RSI'].fillna(0.0)

        # MACD and EMA Calculations
        self.trade_history['EMA_50']=self.trade_history['close'].ewm(span=50, adjust=False).mean()
        self.trade_history['EMA_100']=self.trade_history['close'].ewm(span=100, adjust=False).mean()

        ema_12 = self.trade_history['close'].ewm(span=12, adjust=False).mean()
        ema_26 = self.trade_history['close'].ewm(span=26, adjust=False).mean()
        self.trade_history['MACD'] = ema_12 - ema_26
        self.trade_history['Signal'] = self.trade_history['MACD'].ewm(span=9, adjust=False).mean()
        self.trade_history['MACD_Position'] = self.trade_history['MACD'] - self.trade_history['Signal']

        #Calculate the slope and window width
        self.trade_history['middle_value'] = abs(self.trade_history['close'] - self.trade_history['open'])/2 + self.trade_history[['close', 'open']].min(axis=1)
        self.trade_history['slope']=compute_slope(self.trade_history['middle_value'],2)
        self.trade_history['slope']=self.trade_history['slope'].fillna(0.0)

        #Instead of candle figure classification i will choose to make add wicks to body ratio
        self.trade_history["Body"] = (self.trade_history["close"] - self.trade_history["open"]).abs()
        self.trade_history["UpperWick"] = self.trade_history["high"] - self.trade_history[["open", "close"]].max(axis=1)
        self.trade_history["LowerWick"] = self.trade_history[["open", "close"]].min(axis=1) - self.trade_history["low"]

        self.trade_history["UpperWickRatio"] = np.where(
            self.trade_history["Body"] == 0, self.trade_history["UpperWick"]/np.float32(1), self.trade_history["UpperWick"] / self.trade_history["Body"]
        )

        self.trade_history["LowerWickRatio"] = np.where(
            self.trade_history["Body"] == 0, self.trade_history["LowerWick"]/np.float32(1), self.trade_history["LowerWick"] / self.trade_history["Body"]
        )

        self.trade_history["Direction"] = np.select(
            [
                self.trade_history["close"] > self.trade_history["open"],
                self.trade_history["close"] < self.trade_history["open"],
            ],
            [1, -1],
            default=0
        )

        self.trade_history = self.trade_history.drop(columns=['gain_shifted_interval','MACD','Signal','middle_value','UpperWick','LowerWick'])
        #Save to Data Directory
        makedirs('Data',exist_ok=True)
        self.trade_history.to_csv(path.join('Data','Binance_Bitcoin_minute_clean.csv'), index=False)
        #Print Summry in the Terminal
        print("##################Dataset Summry####################")
        print(self.trade_history.groupby('action')['Direction'].count().rename(index={0:'Exit Entries Count',1:'Enter Entries Count'}))
        return self.trade_history

if __name__ == "__main__":
    cleaner = RawData_Weighting_OHLC()
    cleaner.generate_clean_data()
