class EEGDenoiseLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.3):
        super(EEGDenoiseLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, h0=None, c0=None):
        batch_size = x.size(0)
        device = x.device

        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out)
        return out

class EEGDenoiseLSTM_basic(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.3):
        super(EEGDenoiseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)  # Output one value per time step

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        dropped = self.dropout(lstm_out)
        out = self.fc(dropped)
        return out