
# -*- coding: utf-8 -*-
"""
LSTM do predykcji ceny zamknięcia (Close) dla AAPL na podstawie pliku stock_data/AAPL.csv.
- Tworzy sekwencje czasowe z 'Close' (univariate)
- Trenuje LSTM
- Ocena na zbiorze testowym + wykres
- Prostą prognozę n kolejnych dni w przód

Autor: Zofia (wersja demonstracyjna)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------
# Parametry
# -----------------------
CSV_PATH = "stock_data/AAPL.csv"  # ścieżka do danych
TARGET_COL = "Close"              # przewidujemy cenę zamknięcia
SEQ_LEN = 60                      # długość okna sekwencji (dni)
TEST_RATIO = 0.2                  # procent danych na test
EPOCHS = 50
BATCH_SIZE = 32
FORECAST_DAYS = 10                # ile dni w przód prognozujemy po treningu
MODEL_OUT = "aapl_lstm.keras"     # ścieżka zapisu modelu

# -----------------------
# Funkcje pomocnicze
# -----------------------

def create_sequences(series: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(seq_len, len(series)):
        X.append(series[i-seq_len:i])   # okno wejściowe: (seq_len, 1)
        y.append(series[i][0])          # pojedyncza wartość
    X = np.array(X)                      # (n_samples, seq_len, 1)
    y = np.array(y).reshape(-1, 1)       # (n_samples, 1)
    return X, y


def plot_predictions(dates, y_true, y_pred, title="Predykcja ceny zamknięcia AAPL (test)"):
    plt.figure(figsize=(12,6))
    plt.plot(dates, y_true, label="Rzeczywista (Close)", color="tab:blue")
    plt.plot(dates, y_pred, label="Prognoza LSTM", color="tab:orange")
    plt.title(title)
    plt.xlabel("Data")
    plt.ylabel("Cena (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------
# 1) Wczytanie danych
# -----------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Nie znaleziono pliku: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Sprawdź minimalny zestaw kolumn
required_cols = {"Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Brakuje kolumn w CSV: {missing}")

# Parsuj datę, posortuj rosnąco
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# W razie braków w danych, wypełnij ostatnią znaną wartością
df[TARGET_COL] = df[TARGET_COL].ffill()

# -----------------------
# 2) Przygotowanie szeregów, skalowanie
# -----------------------
# Jednowymiarowa seria docelowa (Close)
series = df[TARGET_COL].values.reshape(-1, 1)

# Podział na train/test chronologicznie
n = len(series)
n_test = int(n * TEST_RATIO)
n_train = n - n_test

series_train = series[:n_train]
series_test = series[n_train:]  # ostatnie 20% na test

# Skalowanie (dopasowanie tylko na train!)
scaler = MinMaxScaler(feature_range=(0, 1))
series_train_scaled = scaler.fit_transform(series_train)
series_test_scaled = scaler.transform(series_test)

# Utwórz sekwencje
X_train, y_train = create_sequences(series_train_scaled, SEQ_LEN)
X_test, y_test = create_sequences(series_test_scaled, SEQ_LEN)

# Dopasuj daty testowe dla wykresu (pierwsze SEQ_LEN dni testowych są "zużyte" na okno)
test_dates = df["Date"].iloc[n_train+SEQ_LEN:].values

print(f"Rozmiary: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")

# -----------------------
# 3) Budowa modelu LSTM
# -----------------------
tf.random.set_seed(42)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1)  # przewidujemy jedną wartość (Close)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="mse",
              metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])

# Early stopping na podstawie walidacji z końcówki zbioru treningowego
# Walidacja: ostatnie 10% próbek treningowych, zachowując chronologię
val_size = max(1, int(0.1 * len(X_train)))
X_val, y_val = X_train[-val_size:], y_train[-val_size:]
X_train_final, y_train_final = X_train[:-val_size], y_train[:-val_size]

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint(MODEL_OUT, monitor="val_loss", save_best_only=True, verbose=1)
]

history = model.fit(
    X_train_final, y_train_final,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    callbacks=callbacks
)

# -----------------------
# 4) Ewaluacja i wizualizacja
# -----------------------
# Ocena na zbiorze testowym (w skali znormalizowanej)
test_loss, test_rmse_scaled = model.evaluate(X_test, y_test, verbose=0)

# Predykcja na zbiorze testowym
y_pred_scaled = model.predict(X_test, verbose=0)

# Odwróć skalowanie, żeby mieć ceny w USD
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred_scaled)

rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)

print(f"Test RMSE (USD): {rmse:.4f}")
print(f"Test MAE  (USD): {mae:.4f}")

# Wykres: rzeczywista vs prognoza na zbiorze testowym
plot_predictions(test_dates, y_test_inv.flatten(), y_pred_inv.flatten(),
                 title="Predykcja ceny zamknięcia AAPL (zbiór testowy)")

# -----------------------
# 5) Prognoza kolejnych dni (FORECAST_DAYS) metodą autoregresyjną
# -----------------------
# Weź ostatnie SEQ_LEN punktów z CAŁEGO zbioru (train+test), przeskaluj
series_all_scaled = scaler.transform(series)  # uwaga: używamy scaler dopasowanego na train!
last_window = series_all_scaled[-SEQ_LEN:].copy().reshape(1, SEQ_LEN, 1)

forecast_scaled = []
for _ in range(FORECAST_DAYS):
    next_scaled = model.predict(last_window, verbose=0)  # kształt (1,1)
    forecast_scaled.append(next_scaled[0,0])
    # doklej prognozę na koniec okna i usuń najstarszy element
    last_window = np.concatenate([last_window[:, 1:, :], next_scaled.reshape(1,1,1)], axis=1)

forecast_scaled = np.array(forecast_scaled).reshape(-1,1)
forecast_inv = scaler.inverse_transform(forecast_scaled).flatten()

# Daty dla prognozy: kolejne dni kalendarzowe po ostatniej dacie z danych
last_date = df["Date"].iloc[-1]
forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D")

plt.figure(figsize=(12,6))
plt.plot(df["Date"].iloc[-200:], df[TARGET_COL].iloc[-200:], label="Historia (ostatnie ~200 dni)", color="tab:blue")
plt.plot(forecast_dates, forecast_inv, label=f"Prognoza na {FORECAST_DAYS} dni", color="tab:red")
plt.title("Prognoza LSTM ceny zamknięcia AAPL (autoregresyjnie)")
plt.xlabel("Data")
plt.ylabel("Cena (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Prognoza kolejnych dni (USD):")
for d, v in zip(forecast_dates, forecast_inv):
    print(f"{d.date()}: {v:.2f}")

# (Opcjonalnie) Zapisz wytrenowany model
model.save(MODEL_OUT)
print(f"Model zapisany do: {MODEL_OUT}")
