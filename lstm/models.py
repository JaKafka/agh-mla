import torch.nn as nn

#
# A simple many-to-one LSTM with only one LSTM layer and a linear output layer
#

class SimpleLSTMRegressor(nn.Module):
    def __init__(self, feature_num=1, hidden_size1=64, lstm_layers=1, dropout=0):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size=feature_num, hidden_size=hidden_size1,
                             batch_first=True, num_layers = lstm_layers, dropout=dropout)
        
        self.fc = nn.Linear(hidden_size1, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        y = self.fc(out)[:, -1, :]
        return y

