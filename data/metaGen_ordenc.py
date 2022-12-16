# meta generator for ordered hot encoding

import pandas as pd

df = pd.read_pickle("match_data/match_data.pickle")

item_set = set()
augment_set = set()
trait_set = set()

Combination_Item = [
    "TFT_Item_B.F.Sword",
    "TFT_Item_BFSword",
    "TFT_Item_ChainVest",
    "TFT_Item_GiantsBelt",
    "TFT_Item_NeedlesslyLargeRod",
    "TFT_Item_NegatronCloak",
    "TFT_Item_RecurveBow",
    "TFT_Item_SparringGloves",
    "TFT_Item_Spatula",
    "TFT_Item_TearOfTheGoddess",
]

df = df[df["status.status_code"].isna()]
df = df[df["info.tft_game_type"] == "standard"]

for game in df["info.participants"]:
    for j in game:
        # trait
        for trait in j["traits"]:
            trait_set.add(trait["name"])

        # item
        for item in j["units"]:
            if item["itemNames"] != []:
                item_list = item["itemNames"]
                if "TFT_Item_ThiefsGloves" in item_list:
                    item_set.add("TFT_Item_ThiefsGloves")
                    break

                for item in item_list:
                    if item not in Combination_Item:
                        item_set.add(item)

        # augment
        for aug in j["augments"]:
            augment_set.add(aug)


print("[item set]\n")
print(item_set)

print("[trait set]\n")
print(trait_set)

print("[augment set]\n")
print(augment_set)

f = open("metaList_ordenc.txt", "w")
for i in sorted(item_set):
    if not i.startswith("TFT_Tutorial"):
        f.write(i + "\n")
for i in sorted(trait_set):
    if not i.startswith("TFT_Tutorial"):
        f.write(i + "\n")
for i in sorted(augment_set):
    if not i.startswith("TFT_Tutorial"):
        f.write(i + "\n")
