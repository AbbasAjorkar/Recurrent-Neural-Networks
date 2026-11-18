import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
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
print(len(dataset_train))