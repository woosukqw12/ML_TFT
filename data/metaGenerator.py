import pandas as pd
import json
from pprint import pprint

# f = open("./match_data/temp_ch_2022-11-10.csv", "r")
# f = open("./check.csv", "r")
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
            trait_set.add(trait["name"] + str(trait["tier_current"]))

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

"""
print("[item set]\n")
pprint(item_set)

print("[trait set]\n")
pprint(trait_set)

print("[augment set]\n")
pprint(augment_set)
"""
f = open("metaList4.txt", "w")
for i in sorted(item_set):
    if not i.startswith("TFT_Tutorial"):
        f.write(i + "\n")
for i in sorted(trait_set):
    if not i.startswith("TFT_Tutorial"):
        f.write(i + "\n")
for i in sorted(augment_set):
    f.write(i + "\n")

"""
'TFTTutorial_Demon0',
 'TFTTutorial_Gunslinger0',
 'TFTTutorial_Imperial0',
 'TFTTutorial_Pirate0',
 'TFTTutorial_Ranger0',
 'TFTTutorial_Shapeshifter0',
 'TFTTutorial_Sorcerer0',
 'TFTTutorial_TutorialPrimary1',
 'TFTTutorial_TutorialSecondary1' 
"""

"""
 'TFT4_Item_OrnnAnimaVisage',
 'TFT4_Item_OrnnDeathsDefiance',
 'TFT4_Item_OrnnEternalWinter',
 'TFT4_Item_OrnnInfinityForce',
 'TFT4_Item_OrnnMuramana',
 'TFT4_Item_OrnnObsidianCleaver',
 'TFT4_Item_OrnnRanduinsSanctum',
 'TFT4_Item_OrnnRocketPropelledFist',
 'TFT4_Item_OrnnTheCollector',
 'TFT4_Item_OrnnZhonyasParadox',
 'TFT5_Item_ArchangelsStaffRadiant',
 'TFT5_Item_BloodthirsterRadiant',
 'TFT5_Item_BlueBuffRadiant',
 'TFT5_Item_BrambleVestRadiant',
 'TFT5_Item_ChaliceOfPowerRadiant',
 'TFT5_Item_DeathbladeRadiant',
 'TFT5_Item_DragonsClawRadiant',
 'TFT5_Item_FrozenHeartRadiant',
 'TFT5_Item_GargoyleStoneplateRadiant',
 'TFT5_Item_GiantSlayerRadiant',
 'TFT5_Item_GuardianAngelRadiant',
 'TFT5_Item_GuinsoosRagebladeRadiant',
 'TFT5_Item_HandOfJusticeRadiant',
 'TFT5_Item_HextechGunbladeRadiant',
 'TFT5_Item_InfinityEdgeRadiant',
 'TFT5_Item_IonicSparkRadiant',
 'TFT5_Item_JeweledGauntletRadiant',
 'TFT5_Item_LastWhisperRadiant',
 'TFT5_Item_LocketOfTheIronSolariRadiant',
 'TFT5_Item_MorellonomiconRadiant',
 'TFT5_Item_QuicksilverRadiant',
 'TFT5_Item_RabadonsDeathcapRadiant',
 'TFT5_Item_RapidFirecannonRadiant',
 'TFT5_Item_RedemptionRadiant',
 'TFT5_Item_RunaansHurricaneRadiant',
 'TFT5_Item_ShroudOfStillnessRadiant',
 'TFT5_Item_SpearOfShojinRadiant',
 'TFT5_Item_StatikkShivRadiant',
 'TFT5_Item_SunfireCapeRadiant',
 'TFT5_Item_ThiefsGlovesRadiant',
 'TFT5_Item_TitansResolveRadiant',
 'TFT5_Item_TrapClawRadiant',
 'TFT5_Item_WarmogsArmorRadiant',
 'TFT5_Item_ZekesHeraldRadiant',
 'TFT5_Item_ZephyrRadiant',
 'TFT5_Item_ZzRotPortalRadiant',
 'TFT7_Item_AssassinEmblemItem',
 'TFT7_Item_BruiserEmblemItem',
 'TFT7_Item_CannoneerEmblemItem',
 'TFT7_Item_CavalierEmblemItem',
 'TFT7_Item_DarkflightEmblemItem',
 'TFT7_Item_DarkflightEssence',
 'TFT7_Item_DragonmancerEmblemItem',
 'TFT7_Item_EvokerEmblemItem',
 'TFT7_Item_GuardianEmblemItem',
 'TFT7_Item_GuildEmblemItem',
 'TFT7_Item_JadeEmblemItem',
 'TFT7_Item_LagoonEmblemItem',
 'TFT7_Item_MageEmblemItem',
 'TFT7_Item_MirageEmblemItem',
 'TFT7_Item_MysticEmblemItem',
 'TFT7_Item_ScalescornEmblemItem',
 'TFT7_Item_ShimmerscaleCrownOfChampions',
 'TFT7_Item_ShimmerscaleDeterminedInvestor',
 'TFT7_Item_ShimmerscaleDeterminedInvestor_HR',
 'TFT7_Item_ShimmerscaleDiamondHands',
 'TFT7_Item_ShimmerscaleDiamondHands_HR',
 'TFT7_Item_ShimmerscaleDravensAxe',
 'TFT7_Item_ShimmerscaleDravensAxe_HR',
 'TFT7_Item_ShimmerscaleEmblemItem',
 'TFT7_Item_ShimmerscaleGamblersBlade',
 'TFT7_Item_ShimmerscaleGamblersBlade_HR',
 'TFT7_Item_ShimmerscaleGoldmancersStaff',
 'TFT7_Item_ShimmerscaleGoldmancersStaff_HR',
 'TFT7_Item_ShimmerscaleHeartOfGold',
 'TFT7_Item_ShimmerscaleMogulsMail',
 'TFT7_Item_ShimmerscaleMogulsMail_HR',
 'TFT7_Item_SwiftshotEmblemItem',
 'TFT7_Item_TempestEmblemItem',
 'TFT7_Item_WarriorEmblemItem'
"""