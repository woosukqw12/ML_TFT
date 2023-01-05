import pandas as pd
import re
import category_encoders as ce
import math
import torch
from torch.utils.data import Dataset, dataloader
import torchvision.transforms as transforms
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, transform=None):
        #   데이터셋의 전처리를 해주는 부분
        self.transform = transform
        ## Feature list import
        df = pd.read_pickle("./data/match_data/match_data.pickle")
        # delete 404 and pairs & turbo mode
        df = df[df["status.status_code"].isna()]
        df = df[df["info.tft_game_type"] == "standard"]

        MetaData = []
        cost = [0, 1, 2, 3, 4, 8, 5, 10]
        with open("./data/metaList_ordenc.txt", "r") as a:
            s = a.readlines()
        for line in s:
            res = re.sub("'|,| |\n", "", line)
            MetaData.append(res)

        # print(MetaData)
        self.X = []
        raw_X = []
        self.label = []
        train_aug1 = []
        train_aug2 = []
        train_aug3 = []
        train_aug = {}
        # depenency category_encoders
        cbe_aug = ce.CatBoostEncoder()
        for game in df["info.participants"]:
            # game is list of 8 players
            # j has augments, companion, gold_left, last_round, level, placement, players_eliminated, puuid, traits, units
            # j is single player
            for j in game:
                # item (112) + trait (29) + aug (3) + value (2) = 146
                value = 0
                X_i = [0 for i in range(141)]
                # ignore if died before 4-2
                if len(j["augments"]) < 3:
                    continue
                for champ in j["units"]:
                    value += cost[champ["rarity"]] * pow(3, champ["tier"] - 1)
                    if champ["itemNames"] != []:
                        if champ["itemNames"][0] == "TFT_Item_ThiefsGloves":
                            idx = MetaData.index("TFT_Item_ThiefsGloves")
                            X_i[idx] = 1
                        else:
                            for item in champ["itemNames"]:
                                if item in MetaData:
                                    idx = MetaData.index(item)
                                    X_i[idx] = 1

                # trait tier_current is activated number of synergy (e.g. 석호6)
                for trait in j["traits"]:
                    if trait["name"] in MetaData:
                        idx = MetaData.index(trait["name"])
                        # print("trait index is " + str(idx))
                        # print(trait["tier_current"])
                        X_i[idx] = math.exp(trait["num_units"]) / 4
                train_aug1.append(j["augments"][0])
                train_aug2.append(j["augments"][1])
                train_aug3.append(j["augments"][2])
                X_i.append(value)  # total cost of deck
                X_i.append(j["total_damage_to_players"])  # total damage to players
                self.label.append(j["placement"])
                self.X.append(X_i)
                raw_X.append(j)

        train_aug["aug1"] = train_aug1
        train_aug["aug2"] = train_aug2
        train_aug["aug3"] = train_aug3
        # print("Start cbe")
        # print(len(train_aug1), len(train_aug2), len(train_aug3), len(label))
        train_aug_df = pd.DataFrame(train_aug)

        # X of cbe_aug should be pd.DataFrame
        cbe_aug.fit(train_aug_df, self.label)
        aug_cbe = cbe_aug.transform(train_aug_df)
        for i in range(len(self.X)):
            # print(aug_cbe["aug1"][i], aug_cbe["aug2"][i], aug_cbe["aug3"][i])
            self.X[i].append(aug_cbe["aug1"][i])
            self.X[i].append(aug_cbe["aug2"][i])
            self.X[i].append(aug_cbe["aug3"][i])
            assert len(self.X[i]) == 146
        # return X, label, raw_X

    def __len__(self):
        #   데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
        return len(self.X)

    def __getitem__(self, idx):
        X_ = self.X[idx]
        if self.transform:
            X_ = self.transform(X_)
        #   데이터셋에서 특정 1개의 샘플을 가져오는 함수
        # X_ = np.expand_dims(X_, axis=(1,2)) # just DNN에선 지우기
        X_ = torch.tensor(X_).to(torch.float32)
        # X_ = X_.expand(-1, 4, 4)
        # (146, 1, 1) -> (146, 4, 4) !!!!! 인코딩 값 146 시너지 + 아이템 + 증강체 + 커스텀 밸류
        # X_ = X_.expand(-1, 8, 8)
        # X_ = X_.expand(-1, 16, 16)

        label_ = torch.tensor(self.label[idx] - 1).to(torch.float32)
        # label_ = label_.view(1)
        label_ = label_.unsqueeze(0)
        # print(label_.shape,"Wdqw")
        # print(X_.shape)
        # print("asdasd", X_i.shape)

        # print("asdasd", X_i.shape)
        return X_, label_


# class CustomDataset2(Dataset):
#     def __init__(self, transform=None):
#     #   데이터셋의 전처리를 해주는 부분
#         self.transform = transform
#         ## Feature list import
#         df = pd.read_pickle("./data/match_data/match_data.pickle")
#         # delete 404 and pairs & turbo mode
#         df = df[df["status.status_code"].isna()]
#         df = df[df["info.tft_game_type"] == "standard"]

#         MetaData = []
#         cost = [0, 1, 2, 3, 4, 8, 5, 10]
#         with open("./data/metaList_ordenc.txt", "r") as a:
#             s = a.readlines()
#         for line in s:
#             res = re.sub("'|,| |\n", "", line)
#             MetaData.append(res)

#         # print(MetaData)
#         self.X = []
#         raw_X = []
#         self.label = []
#         train_aug1 = []
#         train_aug2 = []
#         train_aug3 = []
#         train_aug = {}
#         # depenency category_encoders
#         cbe_aug = ce.CatBoostEncoder()
#         for game in df["info.participants"]:
#             # game is list of 8 players
#             # j has augments, companion, gold_left, last_round, level, placement, players_eliminated, puuid, traits, units
#             # j is single player
#             for j in game:
#                 # item (112) + trait (29) + aug (3) + value (2) = 146
#                 value = 0
#                 X_i = [0 for i in range(141)]
#                 # ignore if died before 4-2
#                 if len(j["augments"]) < 3:
#                     continue
#                 for champ in j["units"]:
#                     value += cost[champ["rarity"]] * pow(3, champ["tier"] - 1)
#                     if champ["itemNames"] != []:
#                         if champ["itemNames"][0] == "TFT_Item_ThiefsGloves":
#                             idx = MetaData.index("TFT_Item_ThiefsGloves")
#                             X_i[idx] = 1
#                         else:
#                             for item in champ["itemNames"]:
#                                 if item in MetaData:
#                                     idx = MetaData.index(item)
#                                     X_i[idx] = 1

#                 # trait tier_current is activated number of synergy (e.g. 석호6)
#                 for trait in j["traits"]:
#                     if trait["name"] in MetaData:
#                         idx = MetaData.index(trait["name"])
#                         # print("trait index is " + str(idx))
#                         # print(trait["tier_current"])
#                         X_i[idx] = math.exp(trait["num_units"]) / 4
#                 train_aug1.append(j["augments"][0])
#                 train_aug2.append(j["augments"][1])
#                 train_aug3.append(j["augments"][2])
#                 X_i.append(value)  # total cost of deck
#                 X_i.append(j["total_damage_to_players"])  # total damage to players
#                 self.label.append(j["placement"])
#                 self.X.append(X_i)
#                 raw_X.append(j)

#         train_aug["aug1"] = train_aug1
#         train_aug["aug2"] = train_aug2
#         train_aug["aug3"] = train_aug3
#         # print("Start cbe")
#         # print(len(train_aug1), len(train_aug2), len(train_aug3), len(label))
#         train_aug_df = pd.DataFrame(train_aug)

#         # X of cbe_aug should be pd.DataFrame
#         cbe_aug.fit(train_aug_df, self.label)
#         aug_cbe = cbe_aug.transform(train_aug_df)
#         for i in range(len(self.X)):
#             # print(aug_cbe["aug1"][i], aug_cbe["aug2"][i], aug_cbe["aug3"][i])
#             self.X[i].append(aug_cbe["aug1"][i])
#             self.X[i].append(aug_cbe["aug2"][i])
#             self.X[i].append(aug_cbe["aug3"][i])
#             assert len(self.X[i]) == 146
#         # return X, label, raw_X


#     def __len__(self):
#         #   데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
#         return len(self.X)


#     def __getitem__(self, idx):
#         X_ = self.X[idx]
#         if self.transform:
#             X_ = self.transform(X_)
#         #   데이터셋에서 특정 1개의 샘플을 가져오는 함수
#         X_ = np.expand_dims(X_, axis=(1,2))
#         X_ = torch.tensor(X_).to(torch.float32)
#         # X_ = X_.expand(-1, 4, 4) # (146, 1, 1) -> (146, 4, 4) !!!!! 인코딩 값 146 시너지 + 아이템 + 증강체 + 커스텀 밸류
#         # X_ = X_.expand(-1, 8, 8)
#         # X_ = X_.expand(-1, 16, 16)


#         label_ = torch.tensor(self.label[idx]-1).to(torch.float32)
#         # label_ = label_.view(1)
#         label_ = label_.unsqueeze(0)
#         # print(label_.shape,"Wdqw")
#         # print(X_.shape)
#         # print("asdasd", X_i.shape)

#         # print("asdasd", X_i.shape)
#         return X_, label_
