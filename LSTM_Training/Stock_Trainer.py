import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from LSTM_Training.Torch_Dataset import TorchDataset
from LSTM_Training.LSTM_Stock_Model import LSTMStockModel
from config.Read_Config_file import Read_Config_file
from os import path
from Helper.Data_Normalizer import DataNormalizer
from typing import List


class StockTrainer:
    def __init__(self, dataframe: pd.DataFrame, features:List, label:str):
        """
        Initialize all configuration and prepare dataset, model, optimizer.
        """
        configuration=Read_Config_file()
        lstm_params =configuration.get_section('LSTM_Hyperparameters')

        #Store data references
        self.features = features
        self.label = label

        #Initial Normalizer, it should come from singleton
        self.normalizer=DataNormalizer(feature_range=(-1,1),features=features,label=label) 

        #Hyperparameters (defined internally)
        self.sequence_length = lstm_params.getint('sequence_length')
        self.hidden_size = lstm_params.getint('hidden_size')
        self.num_layers = lstm_params.getint('num_layers')
        self.batch_size = lstm_params.getint('batch_size')
        self.num_epochs = lstm_params.getint('num_epochs')
        self.learning_rate = lstm_params.getfloat('learning_rate')

        #Split the dataset train and test
        split_index = int(len(dataframe) * 0.8)
        train_df = dataframe.iloc[:split_index]
        train_df_normalized=self.normalizer.fit_transform_training_data(train_df)
        test_df = dataframe.iloc[split_index:]
        test_df_normalized=self.normalizer.transform_testing_data(test_df)

        #Prepare dataset & dataloaders
        self.train_dataset = TorchDataset(train_df_normalized, self.features, self.label, sequence_length=self.sequence_length)
        self.test_dataset = TorchDataset(test_df_normalized, self.features, self.label, sequence_length=self.sequence_length)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        #Model
        input_size = len(self.features)
        num_classes = train_df[label].nunique()
        self.model = LSTMStockModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=num_classes,
        )

        #Make Weigths for the incoming classes
        classes_Weights=dataframe.groupby(self.label).count()['MACD_Position'].to_numpy(dtype='float32')
        classes_Weights=torch.from_numpy(1/classes_Weights)

        #Loss and optimizer
        # compute weights inversely proportional to class frequency
        self.criterion = nn.CrossEntropyLoss(weight=classes_Weights)
        #self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        #Saved model path/filename
        self.best_saved_model=path.join('Model','best_BTC_model.pth')

    def train(self):
        self.model.train()
        accurecy_validation_best=-1*np.inf
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            predections_list=[]
            targets_list=[]
            for inputs, targets in self.train_loader:
                outputs = self.model(inputs)

                #print(f"====>, Inputs shape: {inputs.shape}, Targets shape: {targets.shape}, Outputs shape: {outputs.shape}")
                #print(f"===>{targets}")
                #print(f"===>{outputs}")

                loss = self.criterion(outputs,targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

                # Collect predictions and targets for evaluation
                predections_list.append(torch.argmax(outputs, dim=1).cpu())
                targets_list.append(targets.cpu())

            predections_numpy = torch.cat(predections_list).numpy()
            targets_numpy = torch.cat(targets_list).numpy()
            predections_numpy = torch.cat(predections_list).numpy()
            ff=pd.DataFrame({'predictions': predections_numpy, 'targets': targets_numpy})
            total_0 = ((ff['targets'] == 0)).sum()
            total_1 = ((ff['targets'] == 1)).sum()
            correct_0 = ((ff['predictions'] == 0) & (ff['targets'] == 0)).sum()
            correct_1 = ((ff['predictions'] == 1) & (ff['targets'] == 1)).sum()
            train_result=f"Train - Accurecy-Exit: {correct_0/total_0*100:.2f}, Accurecy-Entrance: {correct_1/total_1*100:.2f}"

            #print(f"Train Dict:{assessment_dict}")
            avg_loss = epoch_loss / len(self.train_loader)

            # Optionally evaluate after each epoch
            accurecy_validationof0,accurecy_validationof1=self.evaluate()
            save=False
            if accurecy_validationof1+accurecy_validationof0>accurecy_validation_best:
                accurecy_validation_best=accurecy_validationof1+accurecy_validationof0
                torch.save(self.model.state_dict(), self.best_saved_model)
                save=True
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f} Validation Accurecy-Exit:{accurecy_validationof0:.2f},Accurecy-Entrance:{accurecy_validationof1:.2f} save:{str(save)} ==> {train_result}")



    def evaluate(self):
        self.model.eval()
        predictions = []
        targets_all = []
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)  # Get class with highest score

                predictions.append(predicted.cpu())
                targets_all.append(targets.cpu())

        predections_numpy = torch.cat(predictions).numpy()
        targets_numpy = torch.cat(targets_all).numpy()
        ff=pd.DataFrame({'predictions': predections_numpy, 'targets': targets_numpy})
        total_0 = ((ff['targets'] == 0)).sum()
        total_1 = ((ff['targets'] == 1)).sum()
        correct_0 = ((ff['predictions'] == 0) & (ff['targets'] == 0)).sum()
        correct_1 = ((ff['predictions'] == 1) & (ff['targets'] == 1)).sum()
        return correct_0/total_0*100,correct_1/total_1*100

