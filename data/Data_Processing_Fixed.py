import pandas as pd
import csv
import json
import pickle

f = open("check.csv", "r")

# f = open("match_data/temp_ch_2022-11-10.csv", "r")

df = pd.read_pickle("match_data/match_data22.pickle")

MetaData_item = [
    "TFT_Item_ArchangelsStaff",
    "TFT_Item_Bloodthirster",
    "TFT_Item_BrambleVest",
    "TFT_Item_Chalice",
    "TFT_Item_Deathblade",
    "TFT_Item_DragonsClaw",
    "TFT_Item_ForceOfNature",
    "TFT_Item_FrozenHeart",
    "TFT_Item_GargoyleStoneplate",
    "TFT_Item_GuardianAngel",
    "TFT_Item_GuinsoosRageblade",
    "TFT_Item_HextechGunblade",
    "TFT_Item_InfinityEdge",
    "TFT_Item_IonicSpark",
    "TFT_Item_JeweledGauntlet",
    "TFT_Item_LastWhisper",
    "TFT_Item_LocketOfTheIronSolari",
    "TFT_Item_MadredsBloodrazor",
    "TFT_Item_Morellonomicon",
    "TFT_Item_PowerGauntlet",
    "TFT_Item_Quicksilver",
    "TFT_Item_RabadonsDeathcap",
    "TFT_Item_RapidFireCannon",
    "TFT_Item_RedBuff",
    "TFT_Item_Redemption",
    "TFT_Item_RunaansHurricane",
    "TFT_Item_SeraphsEmbrace",
    "TFT_Item_Shroud",
    "TFT_Item_SpearOfShojin",
    "TFT_Item_StatikkShiv",
    "TFT_Item_ThiefsGloves",
    "TFT_Item_TitanicHydra",
    "TFT_Item_TitansResolve",
    "TFT_Item_UnstableConcoction",
    "TFT_Item_WarmogsArmor",
    "TFT_Item_ZekesHerald",
    "TFT_Item_Zephyr",
]

X_ch_add = []
Y_ch = []

Combination_Item = [
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

# delete 404
df = df[df["status.status_code"].isna()]
print(len(df))

for game in df["info.participants"]:
    # game is list of 8 players
    # j has augments, companion, gold_left, last_round, level, placement, players_eliminated, puuid, traits, units
    # j is single player
    for j in game:
        X_ch = []
        traits = []
        items = []
        # trait style is activated number of synergy (e.g. 석호6)
        for trait in j["traits"]:
            traits.append(trait["name"] + str(trait["style"]))
        for champ in j["units"]:
            if champ["itemNames"] != []:
                for item in len(champ["itemNames"]):
                    if item not in MetaData_item:
                        del item
                if champ["itemNames"][0] == "TFT_Item_ThiefsGloves":
                    items.append(champ["itemNames"][0])
                else:
                    for item in champ["itemNames"]:
                        items.append(item)

        X_ch.extend(j["augments"])  # append -> extend
        X_ch.extend(traits)
        X_ch.extend(items)
        Y_ch.append(j["placement"])
        X_ch_add.append(X_ch)

# print(X_ch_add)
# print(Y_ch)

print(len(X_ch_add))
print(len(X_ch_add[4]))
print(X_ch_add[0])
print("\n\n")
