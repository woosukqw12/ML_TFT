from pandas import json_normalize
import pandas as pd
from pandas import DataFrame

import pickle

with open(f"match_data/match_data22.pickle", "rb") as f:
    data = pickle.load(f)

data.to_csv("check.csv", encoding="utf-8")