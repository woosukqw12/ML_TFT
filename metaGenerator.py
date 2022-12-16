import csv
import json
from pprint import pprint

f = open('./match_data/temp_ch_2022-11-10.csv', 'r')
rdr = csv.DictReader(f)

item_set = set()
augment_set = set()
trait_set = set()

Combination_Item = ['TFT_Item_B.F.Sword', 'TFT_Item_ChainVest', 'TFT_Item_GiantsBelt', 
'TFT_Item_NeedlesslyLargeRod', 'TFT_Item_NegatronCloak', 'TFT_Item_RecurveBow', 'TFT_Item_SparringGloves', 
'TFT_Item_Spatula', 'TFT_Item_TearOfTheGoddess']

for i in rdr:
    _dict = json.loads(i['info.participants'].replace("'", "\""))
    # print(_dict)
    # break
    for j in _dict:
        # trait
        for k in j['traits']:
            trait_set.add(k['name'] + str(k['style']))

        # item
        for h in j['units']:
            if h['itemNames'] != []:
                item_list = h['itemNames']
                if 'TFT_Item_ThiefsGloves' in item_list:
                    item_set.add('TFT_Item_ThiefsGloves')
                else:
                    for item in item_list:
                        if item not in Combination_Item:
                            item_set.add(item)

        # augment
        for aug in j['augments']:
            augment_set.add(aug)
print('[item set]\n')
pprint(item_set)

print('[trait set]\n')
pprint(trait_set)

print('[augment set]\n')
pprint(augment_set)