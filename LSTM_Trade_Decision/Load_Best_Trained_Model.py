import torch
from os import path
from typing import List

from config.Read_Config_file import Read_Config_file
from LSTM_Training.Stock_Trainer import LSTMStockModel

class LoadBestTradeModel:
    def __init__(self, features: List[str]):
        self.features = features

        # Load LSTM configuration
        config = Read_Config_file()
        lstm_params = config.get_section('LSTM_Hyperparameters')

        self.hidden_size = lstm_params.getint('hidden_size')
        self.num_layers = lstm_params.getint('num_layers')
        model_filename = lstm_params.get('trained_model_name')
        model_path = path.join('Model', model_filename)

        # Validate model file path
        if not path.exists(model_path):
            raise FileNotFoundError(f"Trained model file not found: {model_path}")

        # Load the trained model state
        model_state = torch.load(model_path)

        if "fc.weight" not in model_state:
            raise KeyError("Invalid model file: missing 'fc.weight'")

        num_classes = model_state["fc.weight"].shape[0]

        # Initialize the LSTM model and load weights
        self.model = LSTMStockModel(
            input_size=len(self.features),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=num_classes
        )
        self.model.load_state_dict(model_state)
        self.model.eval()

    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(input_tensor)
