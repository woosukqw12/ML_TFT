import csv
import json

f = open('temp_ch_2022-11-10.csv', 'r')
rdr = csv.DictReader(f)

X_ch_add = []
Y_ch = []

Combination_Item = ['TFT_Item_B.F.Sword', 'TFT_Item_ChainVest', 'TFT_Item_GiantsBelt', 'TFT_Item_NeedlesslyLargeRod', 'TFT_Item_NegatronCloak', 'TFT_Item_RecurveBow', 'TFT_Item_SparringGloves', 'TFT_Item_Spatula', 'TFT_Item_TearOfTheGoddess']

for i in rdr:
    _dict = json.loads(i['info.participants'].replace("'", "\""))
    for j in _dict:
        X_ch = []
        traits = []
        items = []
        for k in j['traits']:
            traits.append(k['name'] + str(k['style']))
        for h in j['units']:
            if h['itemNames'] != []:
                for g in range(len(h['itemNames'])):
                    if h['itemNames'][g] in Combination_Item:
                        h['itemNames'][g] = 0
            
                if h['itemNames'][0] == 0:
                    del h['itemNames'][0]
                if len(h['itemNames']) > 1:
                    if h['itemNames'][1] == 0:
                        del h['itemNames'][1]
                if len(h['itemNames']) > 2:
                    if h['itemNames'][2] == 0:
                        del h['itemNames'][2]
                    
            if h['itemNames'] != []:
                if h['itemNames'][0] == 'TFT_Item_ThiefsGloves':
                    items.append(h['itemNames'][0])
                else:
                    items.append(h['itemNames'])
        X_ch.append(j['augments'])
        X_ch.append(traits)
        X_ch.append(items)
        Y_ch.append(j['placement'])
        X_ch_add.append(X_ch)

print(X_ch_add)
print(Y_ch)