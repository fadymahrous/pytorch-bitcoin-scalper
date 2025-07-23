import torch.nn as nn

class LSTMStockModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lstm_dropout=0.2, fc_dropout=0.2):
        super(LSTMStockModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout if num_layers > 1 else 0.0,  # Only applies if num_layers > 1
            batch_first=True,
            bias=True
        )
        self.dropout = nn.Dropout(fc_dropout)  # Dropout after LSTM output
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]       # Last time step
        out = self.dropout(out)        # Apply dropout before FC
        out = self.fc(out)
        return out
