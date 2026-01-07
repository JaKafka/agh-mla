import os
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data
import torch.nn as nn

import matplotlib.pyplot as plt

def load_dataset_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)["Close"].astype('float32')
    return df

def process_dataset_dataframe(df, test_size = 0.2, history_length = 60):

    def reshape_to_feature(npar):
        return npar.reshape((npar.shape[0], npar.shape[1], 1))

    def reshape_to_target(npar):
        return npar.reshape((npar.shape[0], 1))

    train_size = int(len(df) * (1 - test_size))
    test_size = len(df) - test_size

    train = df[:train_size].reset_index(drop=True)
    test =  df[train_size:].reset_index(drop=True)

    def portion_data(stream, history_length):
        X, y = [], []
        for i in range(0, len(stream) - history_length, 1):
            X.append(stream.values[i:i+history_length])
            y.append(stream.values[i+history_length])
        return torch.tensor(X), torch.tensor(y)

    train_x, train_y = portion_data(train, history_length)
    test_x, test_y = portion_data(test, history_length)

    print("-- Processed dataset --")
    print("Training set examples:",train_x.shape[0])
    print("Testing set examples:",test_x.shape[0])

    train_x = reshape_to_feature(train_x)
    train_y = reshape_to_target(train_y)
    test_x = reshape_to_feature(test_x)
    test_y = reshape_to_target(test_y)

    return train_x, train_y, test_x, test_y


def run_training_loop_MSE(training_device, model_to_train, epoch_count, train_x, train_y, test_x, test_y):
    train_loader = data.DataLoader(data.TensorDataset(train_x, train_y), shuffle=True, batch_size=32)
    test_loader = data.DataLoader(data.TensorDataset(test_x, test_y), shuffle=True, batch_size=32)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_to_train.parameters())

    print("Starting training loop...")

    train_rmse = []
    test_rmse = []

    for epoch in range(1, epoch_count):
        model_to_train.train()

        loss_value = 0
        for xb, yb in train_loader:
            xb = xb.to(training_device)  # (batch, seq_len, 1)
            yb = yb.to(training_device)  # (batch, 1)

            optimizer.zero_grad()
            preds = model_to_train(xb) # (batch, 1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            loss_value+=loss.item()
        train_rmse.append(loss_value)

        model_to_train.eval()
        loss_value = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(training_device)
                yb = yb.to(training_device)

                preds = model_to_train(xb)
                loss = criterion(preds, yb)
                loss_value+=loss.item()
        test_rmse.append(loss_value)

    print("Training loop finished!")

    return {'train_rmse': train_rmse, 'test_rmse': test_rmse}

def print_train_summary_MSE(training_device, trained_model, training_output, train_x, train_y, test_x, test_y):
    fig, ax = plt.subplots(1, 3, figsize=(15,5))

    ax[0].plot(training_output["train_rmse"], label="Train MSE")
    ax[0].plot(training_output["test_rmse"], label="Test MSE")
    ax[0].legend()
    ax[0].set_title("Loss during training")
    ax[0].set_xlabel("Epoch")
    ax[0].set_yscale("log")

    train_loader = data.DataLoader(data.TensorDataset(train_x, train_y), shuffle=False, batch_size=32)
    test_loader = data.DataLoader(data.TensorDataset(test_x, test_y), shuffle=False, batch_size=32)


    with torch.no_grad():
        results_actual_train = []
        results_predicted_train = []
        results_actual_test = []
        results_predicted_test = []
        for xb, yb in train_loader:
            xb = xb.to(training_device)
            yb = yb.to(training_device)
            results_predicted_train.append(trained_model(xb).detach().cpu().numpy())
            results_actual_train.append(yb.cpu().numpy())

        for xb, yb in test_loader:
            xb = xb.to(training_device)
            yb = yb.to(training_device)
            results_predicted_test.append(trained_model(xb).detach().cpu().numpy())
            results_actual_test.append(yb.cpu().numpy())

        results_predicted_train = np.ravel(np.concatenate(results_predicted_train, axis = 0))
        results_predicted_test = np.ravel(np.concatenate(results_predicted_test, axis = 0))
        
        results_actual_train = np.ravel(np.concatenate(results_actual_train, axis = 0))
        results_actual_test = np.ravel(np.concatenate(results_actual_test, axis = 0)) # A consequence of not noticing shuffle was on :(

        ax[1].plot(results_actual_train,c="blue", label="Original")
        ax[1].plot(results_predicted_train,c="green", label="Model output")
        ax[1].set_title("Training output")
        ax[1].legend()
        ax[1].set_xlabel("Day")

        ax[2].plot(results_actual_test,c="blue", label="Original")
        ax[2].plot(results_predicted_test,c="green", label="Model output")
        ax[2].set_title("Test output")
        ax[2].legend()
        ax[2].set_xlabel("Day")
    
    plt.show()

def run_training_loop_RMSE(training_device, model_to_train, epoch_count, train_x, train_y, test_x, test_y):
    train_loader = data.DataLoader(data.TensorDataset(train_x, train_y), shuffle=True, batch_size=32)
    test_loader = data.DataLoader(data.TensorDataset(test_x, test_y), shuffle=True, batch_size=32)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_to_train.parameters())

    print("Starting training loop...")

    train_rmse = []
    test_rmse = []

    for epoch in range(1, epoch_count):
        model_to_train.train()

        loss_value = 0
        for xb, yb in train_loader:
            xb = xb.to(training_device)  # (batch, seq_len, 1)
            yb = yb.to(training_device)  # (batch, 1)

            optimizer.zero_grad()
            preds = model_to_train(xb) # (batch, 1)
            loss = torch.sqrt(criterion(preds, yb))
            loss.backward()
            optimizer.step()
            loss_value+=loss.item()
        train_rmse.append(np.sqrt(loss_value))

        model_to_train.eval()
        loss_value = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(training_device)
                yb = yb.to(training_device)

                preds = model_to_train(xb)
                loss = torch.sqrt(criterion(preds, yb))
                loss_value+=loss.item()
        test_rmse.append(np.sqrt(loss_value))

    print("Training loop finished!")

    return {'train_rmse': train_rmse, 'test_rmse': test_rmse}

def print_train_summary_RMSE(training_device, trained_model, training_output, train_x, train_y, test_x, test_y):
    fig, ax = plt.subplots(1, 3, figsize=(15,5))

    ax[0].plot(training_output["train_rmse"], label="Train RMSE")
    ax[0].plot(training_output["test_rmse"], label="Test RMSE")
    ax[0].legend()
    ax[0].set_title("Loss during training")
    ax[0].set_xlabel("Epoch")
    ax[0].set_yscale("log")

    train_loader = data.DataLoader(data.TensorDataset(train_x, train_y), shuffle=False, batch_size=32)
    test_loader = data.DataLoader(data.TensorDataset(test_x, test_y), shuffle=False, batch_size=32)


    with torch.no_grad():
        results_actual_train = []
        results_predicted_train = []
        results_actual_test = []
        results_predicted_test = []
        for xb, yb in train_loader:
            xb = xb.to(training_device)
            yb = yb.to(training_device)
            results_predicted_train.append(trained_model(xb).detach().cpu().numpy())
            results_actual_train.append(yb.cpu().numpy())

        for xb, yb in test_loader:
            xb = xb.to(training_device)
            yb = yb.to(training_device)
            results_predicted_test.append(trained_model(xb).detach().cpu().numpy())
            results_actual_test.append(yb.cpu().numpy())

        results_predicted_train = np.ravel(np.concatenate(results_predicted_train, axis = 0))
        results_predicted_test = np.ravel(np.concatenate(results_predicted_test, axis = 0))
        
        results_actual_train = np.ravel(np.concatenate(results_actual_train, axis = 0))
        results_actual_test = np.ravel(np.concatenate(results_actual_test, axis = 0)) # A consequence of not noticing shuffle was on :(

        ax[1].plot(results_actual_train,c="blue", label="Original")
        ax[1].plot(results_predicted_train,c="green", label="Model output")
        ax[1].set_title("Training output")
        ax[1].legend()
        ax[1].set_xlabel("Day")

        ax[2].plot(results_actual_test,c="blue", label="Original")
        ax[2].plot(results_predicted_test,c="green", label="Model output")
        ax[2].set_title("Test output")
        ax[2].legend()
        ax[2].set_xlabel("Day")
    
    plt.show()
