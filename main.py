import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
df = pd.read_csv(
    "LD2011_2014.txt",
    sep=";",
    decimal=",",
    index_col=0,
    parse_dates=True
)
series = df.MT_001
first_non_zero_idx = series.ne(0).idxmax()
clean_series = series.loc[first_non_zero_idx:]

def create_seq(dff, seg_len):
    xs, ys = [], []
    for i in range(len(dff) - seg_len):
        x = dff.iloc[i:i + seg_len]
        y = dff.iloc[i + seg_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
X_train, y_train = create_seq(clean_series, seg_len=24*4)
dataset_train = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
        self.fc = nn.Linear(in_features=32, out_features=1)
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 32)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
