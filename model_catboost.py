# # https://catboost.ai/en/docs/concepts/python-reference_catboost

# 증강체 등급 구별

from catboost import CatBoostRegressor, CatBoostClassifier, Pool
import pandas as pd
from encoder import ordered_target_encoding, multi_hot_encoding
from datetime import datetime
import pytz
import utils

time = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
df = pd.read_pickle("./data/match_data/match_data.pickle")

# delete 404 and pairs & turbo mode
df = df[df["status.status_code"].isna()]
df = df[df["info.tft_game_type"] == "standard"]
print(len(df))

X, label, X_raw = ordered_target_encoding(df)
assert len(X) == len(label)
train_X, valid_X, test_X = utils.split_data(X)
train_label, valid_label, test_label = utils.split_data(label)
train_X_raw, valid_X_raw, test_X_raw = utils.split_data(X_raw)


train_data = Pool(data=train_X, label=train_label)
valid_data = Pool(data=valid_X, label=valid_label)
test_data = Pool(data=test_X, label=test_label)
# default lr = 0.03
model = CatBoostRegressor(iterations=5000, task_type="GPU", learning_rate=0.1)
# x_train, x_val = X[:int(len(X)/2)], X[int(len(X)/2):]
# y_train, y_val = label[:int(len(X)/2)], label[int(len(X)/2):]
# model.fit(x_train, y_train,cat_features=list(range(len(MetaData_item))),
#           eval_set=(x_val, y_val))
print("Train Start!")
model.fit(train_data, eval_set=valid_data, plot=True)
print(model.get_best_iteration())
print(model.get_best_score())
preds_rank = model.predict(test_X)
print(preds_rank[:10])
print(test_X_raw[:10])
print(test_label[:10])

model.save_model(f"./models/catboost_{time}.cbm")