import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from typing import List

class TorchDataset(Dataset):
    def __init__(self, ready_dataset: pd.DataFrame, features: List, label: str, sequence_length:int):
        self.ready_dataset=ready_dataset
        self.sequence_length=sequence_length
        self.features=features
        self.label=label

        #convert to tensors
        self.tensor_features=torch.tensor(self.ready_dataset[self.features].values,dtype=torch.float32)
        self.tensor_label=torch.tensor(self.ready_dataset[self.label].values,dtype=torch.long)

    def __len__(self):
        return len(self.ready_dataset)-self.sequence_length

    def __getitem__(self,idx):
        #Shape: [sequence_length, num_features]
        x=self.tensor_features[idx:idx+self.sequence_length]
        # Label: the label at the end of the sequence
        y=self.tensor_label[idx+self.sequence_length]
        return x,y
