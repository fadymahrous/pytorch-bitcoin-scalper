Here is a polished and professionally revised version of your `README.md` file:

---

# Bitcoin\_Scalping\_PyTorch\_LSTM

A Bitcoin scalping model built using PyTorch and LSTM.
The objective of this script is to evaluate the impact of various feature sets, time intervals, and model configurations on prediction accuracy. It also performs model evaluation using a holdout dataset not involved in either training or testing.

---

## 📌 Key Features

* 📈 Fetch historical OHLCV data using `ccxt`
* 🧠 LSTM-based model for generating scalping signals
* 🔄 Simulate trades in real-time or batch mode for backtesting
* 🧪 Enforce separation between training and testing datasets by time interval
* ⚙️ Fully configurable via `config.ini`
* 💾 Save/load trained models and scalers
* 📊 Log trade actions (entry/exit)
* 🧼 Normalize data using a Singleton-based scaler

---

## 📁 Project Structure

```
.
├── Data/                        # Contains historical OHLCV and live test data
├── Model/                       # Stores trained LSTM models and scalers
├── LSTM_Training/              # Model architecture and training logic
│   └── Stock_Trainer.py
├── Data_Preparation/           # Data normalization and preprocessing logic
│   └── Normalizer.py
├── Live_Testing/               # Trade simulation using trained models
│   └── Evaluate_Trade.py
├── config/                     # Configuration files (INI format)
│   └── project_config.ini
├── utils/                      # Utility functions (optional)
├── main.py                     # Entry point for training or evaluation
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration Overview

The entire behavior is controlled via `config/config.ini`.
Below is a breakdown of each section and how it influences training, evaluation, and live simulation.

---

### `[fetch_OHLC_data]`

| Key        | Description                                                 |
| ---------- | ----------------------------------------------------------- |
| `interval` | Candlestick timeframe (`1m`, `5m`, `1h`, etc.)              |
| `since`    | Start date for fetching historical data (`YYYYMMDD` format) |

> 🔁 **Effect:** Determines dataset size and granularity.

---

### `[train_trade_win_criteria_column]`

| Key                        | Description                                           |
| -------------------------- | ----------------------------------------------------- |
| `gain_criteria_win_window` | Look-ahead window for determining gain/loss over time |
| `postive_gain_threshold`   | Gain threshold to label a sample as a "win"           |

> ✅ **Effect:** Influences target label assignment for binary classification.

---

### `[LSTM_Hyperparameters]`

| Key                  | Description                        |
| -------------------- | ---------------------------------- |
| `sequence_length`    | Number of time steps per sequence  |
| `hidden_size`        | Number of LSTM units               |
| `num_layers`         | Stacked LSTM layers                |
| `batch_size`         | Batch size for training            |
| `num_epochs`         | Number of training epochs          |
| `learning_rate`      | Learning rate for optimizer        |
| `trained_model_name` | Output filename for the best model |

> 🧠 **Effect:** Defines the model architecture and training setup.

---

### `[Assess_Trading]`

| Key                     | Description                            |
| ----------------------- | -------------------------------------- |
| `sequence_length`       | Must match training sequence length    |
| `win_session_threshold` | Profit target to exit a trade as a win |
| `investment_capital`    | Capital allocated per trade            |
| `trade_cost_percentage` | Fee per trade (e.g. 0.2%)              |

> 📉 **Effect:** Simulates real-world trading, accounting for gain/loss, cost, and thresholds.

---

### 🔁 Example Workflow

1. Fetch historical data using `interval` and `since`.
2. Assign labels based on `[train_trade_win_criteria_column]`.
3. Train LSTM model using `[LSTM_Hyperparameters]`.
4. Simulate trading based on `[Assess_Trading]` settings.

> 🔥 **Note:** Changing the interval changes the data size and affects the test window. For example, with a `1h` interval, the last 10% of the data is reserved for testing.

---

## 🚀 Getting Started

After updating your config file, simply run:

```bash
python main.py
```

---

## 🧠 Model Architecture

The LSTM model is created based on:

* Number of selected features
* Configurable hidden size and number of layers
* Output layer adapted for binary classification (buy/sell)

---

## 📉 Data Normalization

MinMax scaling is applied using a Singleton design pattern.
The scaler is persisted to `Model/scaler.pkl` for consistency across runs.

---

## 📈 Trade Simulation

During the evaluation phase, predictions are interpreted as:

* `1` → Enter Trade
* `0` → Exit Trade

All trades are logged and aggregated for statistical analysis.

---

## 📊 Output Summary

### 🟡 OHLC Data Fetch Log

```
2024-02-11 → 2025-07-23 (weekly records)
```

---

### 📑 Dataset Summary (Training)

```
Exit Entries Count:  10080
Enter Entries Count:  2212
```

---

### 📈 Training Progress (10 Epochs)

```
Epoch [1/10] Loss: 0.6875 - Val Exit Acc: 0.00%, Enter Acc: 100.00%  - Train Exit Acc: 93.89%, Enter Acc: 8.99%
...
Epoch [10/10] Loss: 0.6732 - Val Exit Acc: 7.00%, Enter Acc: 90.08%  - Train Exit Acc: 79.80%, Enter Acc: 32.61%
```

---

### 📑 Dataset Summary (Unseen Data)

```
Exit Entries Count:  1112
Enter Entries Count:  254
```

---

### 🧪 Simulation Results (Live Trade Simulation)

| Start Date       | End Date         | Gain     | Real Win Ratio |
| ---------------- | ---------------- | -------- | -------------- |
| 2025-05-29 19:00 | 2025-05-29 21:00 | 711.27   | 0.47           |
| ...              | ...              | ...      | ...            |
| 2025-07-14 06:00 | 2025-07-23 00:00 | -2075.80 | -1.93          |

**Total Wins:** 9.80

---

## 👨‍💻 Author

**Fady Mahrous**
[GitHub Profile](https://github.com/fadymahrous)

---

Would you like this returned as a downloadable `.md` file or added back into your project directly?
