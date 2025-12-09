
# -*- coding: utf-8 -*-
"""
LSTM (PyTorch) do predykcji ceny zamknięcia (Close) dla AAPL z pliku stock_data/AAPL.csv.

Pipeline:
- Wczytanie i przygotowanie danych
- Skalowanie tylko na train (MinMax)
- Sekwencje czasowe (univariate: Close)
- Model LSTM (PyTorch), walidacja, early stopping
- Ewaluacja (RMSE/MAE) i wykres
- Autoregresyjna prognoza na kilka dni

Autor: wersja demonstracyjna
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------
# Parametry
# -----------------------
CSV_PATH = "stock_data/AAPL.csv"
TARGET_COL = "Close"
SEQ_LEN = 60
TEST_RATIO = 0.2
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
FORECAST_DAYS = 10
MODEL_OUT = "aapl_lstm_pytorch.pt"
PATIENCE = 10  # early stopping patience (epoki bez poprawy)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# -----------------------
# Dataset do sekwencji
# -----------------------
class TimeSeriesDataset(Dataset):
    """
    Univariate: wejście X ma kształt (seq_len, 1), target y ma kształt (1,)
    Zwracamy batch-first dla LSTM (batch, seq_len, features).
    """
    def __init__(self, series_scaled: np.ndarray, seq_len: int):
        """
        series_scaled: np.ndarray o kształcie (n, 1) po skalowaniu
        """
        assert series_scaled.ndim == 2 and series_scaled.shape[1] == 1, "Oczekuję 2D (n,1) serii"
        self.series = series_scaled.astype(np.float32)
        self.seq_len = seq_len
        self.n = len(series_scaled)

        if self.n <= self.seq_len:
            raise ValueError(f"Za mało danych ({self.n}) względem okna ({self.seq_len}).")

        # liczba dostępnych próbek (każda to okno seq_len i następna wartość jako target)
        self.samples = self.n - self.seq_len

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        x = self.series[idx:idx + self.seq_len]              # (seq_len, 1)
        y = self.series[idx + self.seq_len]                  # (1,)
        x = torch.from_numpy(x)                              # float32
        y = torch.from_numpy(y)                              # float32
        return x, y


# -----------------------
# Model LSTM (PyTorch)
# -----------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size1=64, hidden_size2=32, dropout=0.2):
        super().__init__()
        # batch_first=True -> wejście: (batch, seq_len, input_size)
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size1,
                             batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2,
                             batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        # Bierzemy ostatni krok czasowy
        last = out[:, -1, :]           # (batch, hidden_size2)
        y = self.fc(last)              # (batch, 1)
        return y


# -----------------------
# 1) Wczytanie danych
# -----------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Nie znaleziono pliku: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

required_cols = {"Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Brakuje kolumn w CSV: {missing}")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
df[TARGET_COL] = df[TARGET_COL].ffill()

# 2) Przygotowanie serii i podział train/test
series = df[TARGET_COL].values.reshape(-1, 1)  # (n,1)
n = len(series)
n_test = int(n * TEST_RATIO)
n_train = n - n_test
series_train = series[:n_train]
series_test = series[n_train:]

# Skalowanie tylko na train
scaler = MinMaxScaler(feature_range=(0, 1))
series_train_scaled = scaler.fit_transform(series_train)   # (n_train,1)
series_test_scaled = scaler.transform(series_test)         # (n_test,1)

# Tworzymy dataset dla train i test
train_ds = TimeSeriesDataset(series_train_scaled, SEQ_LEN)
test_ds = TimeSeriesDataset(series_test_scaled, SEQ_LEN)

# Walidacja: ostatnie 10% próbek z train (chronologicznie)
val_size = max(1, int(0.1 * len(train_ds)))
train_size = len(train_ds) - val_size

# Uwaga: TimeSeriesDataset zwraca okna przesuwne, więc podział indeksami zachowuje chronologię
indices = np.arange(len(train_ds))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Subset via torch.utils.data.Subset
from torch.utils.data import Subset
train_subset = Subset(train_ds, train_indices.tolist())
val_subset = Subset(train_ds, val_indices.tolist())

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# Daty dla testu (pierwsze SEQ_LEN dni testowych są "zużyte" na okno)
test_dates = df["Date"].iloc[n_train + SEQ_LEN:].values

print(f"Rozmiary: train_samples={len(train_ds)}, val_samples={len(val_subset)}, test_samples={len(test_ds)}")

# -----------------------
# 3) Budowa i trening modelu
# -----------------------
model = LSTMRegressor(input_size=1, hidden_size1=64, hidden_size2=32, dropout=0.2).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float("inf")
no_improve_epochs = 0

for epoch in range(1, EPOCHS + 1):
    # --- trening ---
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb = xb.to(DEVICE)  # (batch, seq_len, 1)
        yb = yb.to(DEVICE)  # (batch, 1)

        optimizer.zero_grad()
        preds = model(xb)                    # (batch, 1)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    # --- walidacja ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_losses.append(loss.item())

    avg_train = np.mean(train_losses) if train_losses else float("nan")
    avg_val = np.mean(val_losses) if val_losses else float("nan")
    print(f"Epoka {epoch:03d} | train_loss={avg_train:.6f} | val_loss={avg_val:.6f}")

    # early stopping
    if avg_val < best_val_loss - 1e-6:
        best_val_loss = avg_val
        no_improve_epochs = 0
        torch.save(model.state_dict(), MODEL_OUT)  # zapis najlepszego
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= PATIENCE:
            print(f"Brak poprawy przez {PATIENCE} epok — zatrzymuję trening.")
            break

# Wczytaj najlepszy zapis (na wszelki wypadek)
if os.path.exists(MODEL_OUT):
    model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE))
    print(f"Wczytano najlepszy model z: {MODEL_OUT}")

# -----------------------
# 4) Ewaluacja na teście
# -----------------------
model.eval()
y_test_scaled_list = []
y_pred_scaled_list = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        preds = model(xb)
        # Zbieramy do list (na CPU)
        y_test_scaled_list.append(yb.cpu().numpy())   # (batch,1)
        y_pred_scaled_list.append(preds.cpu().numpy())

# Sklej do (n_samples, 1)
y_test_scaled = np.vstack(y_test_scaled_list)
y_pred_scaled = np.vstack(y_pred_scaled_list)

# Odwrócenie skalowania (MinMax wymaga 2D wejścia)
y_test_inv = scaler.inverse_transform(y_test_scaled)       # (n_samples,1)
y_pred_inv = scaler.inverse_transform(y_pred_scaled)       # (n_samples,1)

rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)
print(f"Test RMSE (USD): {rmse:.4f}")
print(f"Test MAE  (USD): {mae:.4f}")

# Wykres: rzeczywista vs prognoza na zbiorze testowym
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_inv.flatten(), label="Rzeczywista (Close)", color="tab:blue")
plt.plot(test_dates, y_pred_inv.flatten(), label="Prognoza LSTM (PyTorch)", color="tab:orange")
plt.title("Predykcja ceny zamknięcia AAPL (zbiór testowy) — PyTorch LSTM")
plt.xlabel("Data")
plt.ylabel("Cena (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------
# 5) Prognoza kolejnych dni (autoregresja)
# -----------------------
# Przygotowanie okna z końca całej serii (train+test), skalowane tym samym scalerem
series_all_scaled = scaler.transform(series)    # (n,1) Uwaga: scaler dopasowany do train
last_window = series_all_scaled[-SEQ_LEN:].astype(np.float32)  # (seq_len,1)

model.eval()
forecast_scaled = []

with torch.no_grad():
    x_win = torch.from_numpy(last_window).unsqueeze(0).to(DEVICE)  # (1, seq_len, 1)
    for _ in range(FORECAST_DAYS):
        next_val = model(x_win)          # (1,1)
        forecast_scaled.append(next_val.cpu().numpy()[0, 0])
        # aktualizujemy okno: odcinamy pierwszy element i dokładamy predykcję
        next_val_2d = next_val.cpu().numpy().reshape(1, 1)         # (1,1)
        # sklej (seq_len-1) ostatnich + nowa wartość
        new_win = np.concatenate([x_win.cpu().numpy()[:, 1:, :],
                                  next_val_2d.reshape(1, 1, 1)], axis=1)  # (1, seq_len, 1)
        x_win = torch.from_numpy(new_win).to(DEVICE)

forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)  # (FORECAST_DAYS,1)
forecast_inv = scaler.inverse_transform(forecast_scaled).flatten()

last_date = df["Date"].iloc[-1]
forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1),
                               periods=FORECAST_DAYS, freq="D")

plt.figure(figsize=(12, 6))
plt.plot(df["Date"].iloc[-200:], df[TARGET_COL].iloc[-200:], label="Historia (ostatnie ~200 dni)", color="tab:blue")
plt.plot(forecast_dates, forecast_inv, label=f"Prognoza na {FORECAST_DAYS} dni", color="tab:red")
plt.title("Prognoza LSTM ceny zamknięcia AAPL (autoregresyjnie) — PyTorch")
plt.xlabel("Data")
plt.ylabel("Cena (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Prognoza kolejnych dni (USD):")
for d, v in zip(forecast_dates, forecast_inv):
    print(f"{d.date()}: {v:.2f}")
