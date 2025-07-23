import pandas as pd
import torch
from typing import List

from Helper.Data_Normalizer import DataNormalizer
from Core_Trade.RawData_Weighting_OHLC import RawData_Weighting_OHLC
from config.Read_Config_file import Read_Config_file
from LSTM_Trade_Decision.Load_Best_Trained_Model import LoadBestTradeModel


class AssessTradeSimulator:
    def __init__(self, features: List[str]):
        self.features = features
        config = Read_Config_file().get_section('Assess_Trading')

        self.sequence_length = config.getint('sequence_length')
        self.stoploss = config.getint('stoploss')
        self.win_threshold = config.getint('win_session_threshold')
        self.capital = config.getint('investment_capital')
        self.trade_fees = (config.getfloat('trade_cost_percentage') / 100) * self.capital

        self.normalizer = DataNormalizer(feature_range=(-1, 1), features=features, label='action')
        self.model = LoadBestTradeModel(features)

    def _prepare_input(self, df: pd.DataFrame, start: int, end: int) -> torch.Tensor:
        slice = df.iloc[start:end + 1].to_numpy(dtype='float32')
        return torch.tensor(slice, dtype=torch.float32).unsqueeze(0)

    def _predict(self, tensor: torch.Tensor) -> int:
        return torch.argmax(self.model.predict(tensor)).item()

    def estimate_wins(self, prev_month: pd.DataFrame) -> pd.DataFrame:
        weighted_df = RawData_Weighting_OHLC(prev_month).generate_clean_data()
        normalized_df = self.normalizer.transform_evaluation_data(weighted_df[self.features])

        result = []
        gain = 0
        active_trade = False
        index = self.sequence_length

        while index < len(weighted_df):
            row = weighted_df.iloc[index]
            input_tensor = self._prepare_input(normalized_df, index - self.sequence_length, index)
            action = self._predict(input_tensor)

            gain += row.gain

            if action == 1 and not active_trade:
                active_trade = True
                gain = 0
                trade = {'Start_date': row.date}

            #elif active_trade and (gain < self.stoploss or gain > self.win_threshold or action == 0):
            elif active_trade and (action == 0 or gain>self.win_threshold ):
                active_trade = False
                trade['End_date'] = row.date
                trade['gain'] = gain
                trade['real_win'] = (self.capital * gain / row.close) - self.trade_fees
                result.append(trade)

            index += 1
        #Make Sure that no active session is ongoing
        if active_trade:
            active_trade = False
            trade['End_date'] = row.date
            trade['gain'] = gain
            trade['real_win'] = (self.capital * gain / row.close) - self.trade_fees
            result.append(trade)


        sessions = pd.DataFrame(result)
        if not sessions.empty:
            print(sessions)
            print(f"Total Wins: {sessions['real_win'].sum():.2f}")
        return sessions
