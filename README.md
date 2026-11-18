# Recurrent-Neural-Networks
# Electricity Load Forecasting with Recurrent Neural Networks (PyTorch)

This project implements a time-series forecasting pipeline using Recurrent Neural Networks (RNN/LSTM/GRU) in PyTorch on the **ElectricityLoadDiagrams20112014** dataset.

## Dataset Setup

Due to GitHub's file size limits, the raw dataset (~711 MB) is not included in this repository. Please follow the steps below to download and prepare the data before running the code:

1. **Download the dataset** from the UCI Machine Learning Repository:
   - [ElectricityLoadDiagrams20112014 Dataset (UCI)](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
2. **Extract the archive** (`electricityloaddiagrams20112014.zip`) to obtain `LD2011_2014.txt`.
3. **Place the file** inside a `data/` directory at the root of this project:
   ```text
   Recurrent-Neural-Networks/
   ├── data/
   │   └── LD2011_2014.txt
   ├── main.py
   ├── .gitignore
   └── README.md