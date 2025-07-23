import time
from datetime import datetime, timezone
import ccxt
import pandas as pd
from os import path,makedirs
from typing import List


class HistoricalOHLCDownloader:

    HEADER = ['date', 'open', 'high', 'low', 'close', 'volume']
    INTERVALS=['1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d','3d','1w','1M']
    SYMBOL='BTC/USDT'
    
    def __init__(self,interval:str):
       """
       This class  generate two files
       file1 : will be used in Model trainin and Model evaluation
       file2 : Will be used to check how the Model will react on real or live data and how much money it will get out of it
        
        - We Provide start Data and data will be fetched till today.

       Fact--> file2 is always 10% of the interval provided.
       So for instance if the interval provided 1-Jan and today is 1-Nov, so file2 will contain all October value.
       """
       if interval is None:
           raise ValueError(f'you must specify interval value on of these {self.INTERVALS}')
       if interval not in self.INTERVALS:
           raise ValueError(f'The Interval value passed is not valid value it should be in {self.INTERVALS}')
       
       self.interval=interval
       self.exchange = ccxt.binance()
       self.symbol=self.SYMBOL

    def _download(self,start_millisecond)->List:
        """
        This method takes start time as millisecond integer
        """
        if not isinstance(start_millisecond,int):
            raise ValueError('The value passed should be integer represents millisecond of epoch time.')
        
        all_data=[]
        now_millisecond=int(datetime.now().timestamp()*1000)
        while start_millisecond < now_millisecond:
            try:
                candles = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.interval, since=start_millisecond, limit=1000)
            except Exception as e:
                raise RuntimeError(f'Fetch to get data from exchange, probely connection issue: details {e}')
            if not candles:
                break

            all_data.extend(candles)
            #Record the last Candel time
            start_millisecond = candles[-1][0] + 1  # next start timestamp

            dt = datetime.fromtimestamp(start_millisecond / 1000, tz=timezone.utc)
            print(dt)
            time.sleep(self.exchange.rateLimit / 1000)  # respect rate limit
        return all_data
    
    def _split_list(self,all_data_list:List,split_range:int=0.9):
        if all_data_list is None or not isinstance(all_data_list,list):
            raise ValueError('Be sure list is passed, Value passed is either None or not passed. ')
        if split_range>0.9 or not isinstance(split_range,float):
            raise ValueError(f'Be Sure the split value is less thatn 0.9 and its float value')

        seperator=int(split_range*(len(all_data_list)))
        return all_data_list[:seperator],all_data_list[seperator:]

    def fetch_ohlcv_range(self,start_time_in_str:str)-> None:
        try:
            start_time=datetime.strptime(start_time_in_str,'%Y%m%d')
        except:
            raise ValueError(f'The value provided is wrong like for 20-July-2024 you should enter 20240720 like this mask YYYYMMDD')
 
        #Convert the time in millisecond
        start_millisecond=int(start_time.timestamp()*1000)
        all_data=self._download(start_millisecond)

        train_list,evaluation_list=self._split_list(all_data)

        makedirs('Data', exist_ok=True)

        df_file1=pd.DataFrame(train_list,columns=self.HEADER)
        df_file1['date'] = pd.to_datetime(df_file1['date'], unit='ms', utc=True)
        df_file1.to_csv(path.join('Data',f'BitCoin_{self.interval}_DataForModelToTrain.csv'),index=False)

        df_file2=pd.DataFrame(evaluation_list,columns=self.HEADER)
        df_file2['date'] = pd.to_datetime(df_file2['date'], unit='ms', utc=True)
        df_file2.to_csv(path.join('Data',f'BitCoin_{self.interval}_DataToValidateModel.csv'),index=False)
