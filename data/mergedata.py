from time import sleep
import requests
import json
from pandas import json_normalize
import pandas as pd
from pandas import DataFrame

import pickle

data = pd.DataFrame()
# 1~7
for i in range(1, 8):
    with open(f"match_data/match_data{i}.pickle", "rb") as f:
        data_t = pickle.load(f)
    data = pd.concat([data, data_t])

# 9~15
for i in range(9, 16):
    with open(f"match_data/match_data{i}.pickle", "rb") as f:
        data_t = pickle.load(f)
    data = pd.concat([data, data_t])

# 22~30
for i in range(22, 31):
    with open(f"match_data/match_data{i}.pickle", "rb") as f:
        data_t = pickle.load(f)
    data = pd.concat([data, data_t])

with open("match_data/match_info20.pickle", "rb") as f:
    data_t = pickle.load(f)

data = pd.concat([data, data_t])

with open(f"match_data/match_data.pickle", "wb") as f:
    pickle.dump(data, f)

# data.to_csv("match_data/match_data.csv", encoding="utf-8")

print(len(data))
