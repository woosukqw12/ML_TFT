import pandas as pd
import re
import category_encoders as ce
import math

# rarity - cost array
# e.g. Ao Shin is Rarity 7, 10 cost
cost = [0, 1, 2, 3, 4, 8, 5, 10]
pw = [
    [0, 1, 2, 3, 4, 8, 5, 10],
    [
        0,
        3,
        5,
    ],
]


def multi_hot_encoding(df):
    ## Feature list import
    MetaData = []
    # open metadata for multi hot encoding
    with open("./data/metaList4.txt", "r") as a:
        s = a.readlines()
    for line in s:
        res = re.sub("'|,| |\n", "", line)
        MetaData.append(res)

    X = []
    raw_X = []
    label = []
    for game in df["info.participants"]:
        # game is list of 8 players
        # j has augments, companion, gold_left, last_round, level, placement, players_eliminated, puuid, traits, units
        # j is single player
        for j in game:
            value = 0
            X_i = []
            # ignore if died before 4-2 (outlier)
            if len(j["augments"]) < 3:
                continue
            # trait tier_current is activated number of synergy (e.g. 석호6)
            for trait in j["traits"]:
                if trait["name"] + str(trait["tier_current"]) in MetaData:
                    X_i.append(trait["name"] + str(trait["num_units"]))
            for champ in j["units"]:
                value += cost[champ["rarity"]] * pow(3, champ["tier"] - 1)
                if champ["itemNames"] != []:
                    if champ["itemNames"][0] == "TFT_Item_ThiefsGloves":
                        X_i.append("TFT_Item_ThiefsGloves")
                    else:
                        for item in champ["itemNames"]:
                            if item in MetaData:
                                X_i.append(item)
            for aug in j["augments"]:
                if aug in MetaData:
                    X_i.append(aug)

            X_i.append(value)  # total cost of deck
            X_i.append(j["total_damage_to_players"])  # total damage to players
            raw_X.append(j)
            label.append(j["placement"])  # rank (label)
            X.append(sorted(X_i))  # sorted

    X = [[1 if i in row_x else 0 for i in MetaData] for row_x in X]
    # multi-hot_encoding
    return X, label, raw_X


def ordered_target_encoding(df):
    ## Feature list import
    MetaData = []
    with open("./data/metaList_ordenc.txt", "r") as a:
        s = a.readlines()
    for line in s:
        res = re.sub("'|,| |\n", "", line)
        MetaData.append(res)

    # print(MetaData)
    X = []
    raw_X = []
    label = []
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
            # item (112) + trait (29) + aug (3) + special feature (2) = 146
            value = 0
            X_i = [0 for i in range(141)]
            # ignore if died before 4-2 (outlier)
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
                    X_i[idx] = math.exp(trait["num_units"])
            train_aug1.append(j["augments"][0])
            train_aug2.append(j["augments"][1])
            train_aug3.append(j["augments"][2])
            X_i.append(value)  # total cost of deck
            X_i.append(j["total_damage_to_players"])  # total damage to players
            label.append(j["placement"])  # rank (label)
            X.append(X_i)
            raw_X.append(j)

    train_aug["aug1"] = train_aug1
    train_aug["aug2"] = train_aug2
    train_aug["aug3"] = train_aug3
    print("Start cbe")
    print(len(train_aug1), len(train_aug2), len(train_aug3), len(label))
    train_aug_df = pd.DataFrame(train_aug)

    # X of cbe_aug should be pd.DataFrame
    # Ordered Target Encoding
    cbe_aug.fit(train_aug_df, label)
    aug_cbe = cbe_aug.transform(train_aug_df)
    for i in range(len(X)):
        # print(aug_cbe["aug1"][i], aug_cbe["aug2"][i], aug_cbe["aug3"][i])
        X[i].append(aug_cbe["aug1"][i])
        X[i].append(aug_cbe["aug2"][i])
        X[i].append(aug_cbe["aug3"][i])
        assert len(X[i]) == 146
    return X, label, raw_X