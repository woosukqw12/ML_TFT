# # https://catboost.ai/en/docs/concepts/python-reference_catboost

from catboost import CatBoostRegressor, CatBoostClassifier, Pool
import pandas as pd
from encoder import ordered_target_encoding, multi_hot_encoding
from datetime import datetime
import pytz
import utils

time = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
# read data from merged pickle
df = pd.read_pickle("./data/match_data/match_data.pickle")

# delete 404 Not Found and pairs & turbo mode
df = df[df["status.status_code"].isna()]
df = df[df["info.tft_game_type"] == "standard"]
print(len(df))

# get ordered target encoded categorical features
X, label, X_raw = ordered_target_encoding(df)
assert len(X) == len(label)

# split dataset into train, valid, test
train_X, valid_X, test_X = utils.split_data(X)
train_label, valid_label, test_label = utils.split_data(label)
train_X_raw, valid_X_raw, test_X_raw = utils.split_data(X_raw)


train_data = Pool(data=train_X, label=train_label)
valid_data = Pool(data=valid_X, label=valid_label)
test_data = Pool(data=test_X, label=test_label)
# default lr = 0.03
# CatBoost Regression Tree
model = CatBoostRegressor(iterations=5000, task_type="GPU")

print("Train Start!")

# Train
model.fit(train_data, eval_set=valid_data, plot=True)
print(model.get_best_iteration())
print(model.get_best_score())

# Inference
preds_rank = model.predict(test_X)
print(preds_rank[:30])
print(test_X_raw[:30])
print(test_label[:30])

# Save best model
model.save_model(f"./models/catboost_{time}.cbm")