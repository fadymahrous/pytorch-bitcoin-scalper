import pandas as pd
from os import path
from Core_Trade.RawData_Weighting_OHLC import RawData_Weighting_OHLC
from LSTM_Training.Stock_Trainer import StockTrainer
from LSTM_Trade_Decision.Assess_Trade_Simulator import AssessTradeSimulator
from Helper.Data_Normalizer import DataNormalizer
from Helper.Historical_OHLC_Downloader import HistoricalOHLCDownloader
from config.Read_Config_file import Read_Config_file

config=Read_Config_file()
fetch_ohlc_config=config.get_section('fetch_OHLC_data')
ohlc_fetcher=HistoricalOHLCDownloader(fetch_ohlc_config.get('interval'))
ohlc_fetcher.fetch_ohlcv_range(fetch_ohlc_config.get('since'))

#Features and labels, the label must be the last column
features=['MACD_Position','EMA_50','EMA_100',
                'slope','UpperWickRatio',
                'LowerWickRatio','volume']
label='action'

#Intialize Singletone Data Normalizer instance
normalizer=DataNormalizer(feature_range=(-1,1),features=features,label=label)  

#load the OHLC data and Generate the clean weigted data like scales, candles shapes and slopes
ohlc_dataset=pd.read_csv(path.join('Data',f'BitCoin_{fetch_ohlc_config.get("Interval")}_DataForModelToTrain.csv'),parse_dates=['date'])
weighting_data=RawData_Weighting_OHLC(ohlc_dataset)
weighted_dataset=weighting_data.generate_clean_data()

#start the Model Training
trainer_out=StockTrainer(weighted_dataset,features,'action')
trainer_out.train()

###This is the last month data we feed it to the model and see wheter it will be able to make profit out of it or no
simulator=AssessTradeSimulator(features)
prev_month=pd.read_csv(path.join('Data',f'BitCoin_{fetch_ohlc_config.get("Interval")}_DataToValidateModel.csv'),parse_dates=['date'])
simulator.estimate_wins(prev_month)
