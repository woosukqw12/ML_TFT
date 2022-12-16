# .\venv\Scripts\activate
# deactivate
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import json
# import xmltodict # xml파일의 내용을 딕셔너리에 저장할 수 있는 메소드들이 들어있는 모듈입니다. 
import pandas as pd
import csv
import re
from pprint import pprint
from PIL import Image
import numpy as np
# from tqdm import tqdm

class My_own_dataset():
    def __init__(self, root='./', train=True, transform=None, target_transform=None, resize=224) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.resize_factor = resize
        self.dir_path = "./data/match_data" # = self.root

        ###### Data Processing fixed.py
        self.f = open('./data/sorted_temp2.csv', 'r')
        # self.f = open('./data/sorted_temp2.csv', 'r') len:2555
        # data len: 2555*8 = 20440
        self.rdr = csv.DictReader(self.f)

        self.MetaData_item = [
            'TFT_Item_ArchangelsStaff',
            'TFT_Item_Bloodthirster',
            'TFT_Item_BrambleVest',
            'TFT_Item_Chalice',
            'TFT_Item_Deathblade',
            'TFT_Item_DragonsClaw',
            'TFT_Item_ForceOfNature',
            'TFT_Item_FrozenHeart',
            'TFT_Item_GargoyleStoneplate',
            'TFT_Item_GuardianAngel',
            'TFT_Item_GuinsoosRageblade',
            'TFT_Item_HextechGunblade',
            'TFT_Item_InfinityEdge',
            'TFT_Item_IonicSpark',
            'TFT_Item_JeweledGauntlet',
            'TFT_Item_LastWhisper',
            'TFT_Item_LocketOfTheIronSolari',
            'TFT_Item_MadredsBloodrazor',
            'TFT_Item_Morellonomicon',
            'TFT_Item_PowerGauntlet',
            'TFT_Item_Quicksilver',
            'TFT_Item_RabadonsDeathcap',
            'TFT_Item_RapidFireCannon',
            'TFT_Item_RedBuff',
            'TFT_Item_Redemption',
            'TFT_Item_RunaansHurricane',
            'TFT_Item_SeraphsEmbrace',
            'TFT_Item_Shroud',
            'TFT_Item_SpearOfShojin',
            'TFT_Item_StatikkShiv',
            'TFT_Item_ThiefsGloves',
            'TFT_Item_TitanicHydra',
            'TFT_Item_TitansResolve',
            'TFT_Item_UnstableConcoction',
            'TFT_Item_WarmogsArmor',
            'TFT_Item_ZekesHerald',
            'TFT_Item_Zephyr',
            'Set7_Assassin0',
            'Set7_Assassin1',
            'Set7_Assassin3',
            'Set7_Assassin4',
            'Set7_Astral0',
            'Set7_Astral1',
            'Set7_Astral2',
            'Set7_Astral3',
            'Set7_Bard3',
            'Set7_Bruiser0',
            'Set7_Bruiser1',
            'Set7_Bruiser2',
            'Set7_Bruiser3',
            'Set7_Bruiser4',
            'Set7_Cannoneer0',
            'Set7_Cannoneer1',
            'Set7_Cannoneer2',
            'Set7_Cannoneer3',
            'Set7_Cavalier0',
            'Set7_Cavalier1',
            'Set7_Cavalier2',
            'Set7_Cavalier3',
            'Set7_Cavalier4',
            'Set7_Darkflight0',
            'Set7_Darkflight1',
            'Set7_Darkflight2',
            'Set7_Darkflight3',
            'Set7_Darkflight4',
            'Set7_Dragon1',
            'Set7_Dragon2',
            'Set7_Dragon3',
            'Set7_Dragon4',
            'Set7_Dragonmancer0',
            'Set7_Dragonmancer1',
            'Set7_Dragonmancer2',
            'Set7_Dragonmancer3',
            'Set7_Dragonmancer4',
            'Set7_Evoker0',
            'Set7_Evoker1',
            'Set7_Evoker2',
            'Set7_Evoker3',
            'Set7_Guardian0',
            'Set7_Guardian1',
            'Set7_Guardian2',
            'Set7_Guardian3',
            'Set7_Guardian4',
            'Set7_Guild1',
            'Set7_Guild2',
            'Set7_Guild3',
            'Set7_Guild4',
            'Set7_Jade0',
            'Set7_Jade1',
            'Set7_Jade2',
            'Set7_Jade3',
            'Set7_Jade4',
            'Set7_Lagoon0',
            'Set7_Lagoon1',
            'Set7_Lagoon2',
            'Set7_Lagoon3',
            'Set7_Lagoon4',
            'Set7_Mage0',
            'Set7_Mage1',
            'Set7_Mage2',
            'Set7_Mage3',
            'Set7_Mage4',
            'Set7_Mirage0',
            'Set7_Mirage1',
            'Set7_Mirage2',
            'Set7_Mirage3',
            'Set7_Mirage4',
            'Set7_Monolith3',
            'Set7_Mystic0',
            'Set7_Mystic1',
            'Set7_Mystic2',
            'Set7_Mystic3',
            'Set7_Mystic4',
            'Set7_Prodigy3',
            'Set7_Ragewing0',
            'Set7_Ragewing1',
            'Set7_Ragewing2',
            'Set7_Ragewing3',
            'Set7_Scalescorn0',
            'Set7_Scalescorn1',
            'Set7_Scalescorn3',
            'Set7_Scalescorn4',
            'Set7_Shapeshifter0',
            'Set7_Shapeshifter1',
            'Set7_Shapeshifter3',
            'Set7_Shimmerscale0',
            'Set7_Shimmerscale1',
            'Set7_Shimmerscale2',
            'Set7_Shimmerscale3',
            'Set7_Shimmerscale4',
            'Set7_SpellThief3',
            'Set7_Starcaller3',
            'Set7_Swiftshot0',
            'Set7_Swiftshot1',
            'Set7_Swiftshot2',
            'Set7_Swiftshot3',
            'Set7_Swiftshot4',
            'Set7_Tempest0',
            'Set7_Tempest1',
            'Set7_Tempest2',
            'Set7_Tempest3',
            'Set7_Tempest4',
            'Set7_Warrior0',
            'Set7_Warrior1',
            'Set7_Warrior3',
            'Set7_Warrior4',
            'Set7_Whispers0',
            'Set7_Whispers1',
            'Set7_Whispers2',
            'Set7_Whispers3',
            'TFT6_Augment_Ascension',
            'TFT6_Augment_BandOfThieves2',
            'TFT6_Augment_Battlemage1',
            'TFT6_Augment_Battlemage2',
            'TFT6_Augment_Battlemage3',
            'TFT6_Augment_BinaryAirdrop',
            'TFT6_Augment_BlueBattery1',
            'TFT6_Augment_BlueBattery2',
            'TFT6_Augment_CalculatedLoss',
            'TFT6_Augment_CelestialBlessing1',
            'TFT6_Augment_CelestialBlessing2',
            'TFT6_Augment_CelestialBlessing3',
            'TFT6_Augment_ClearMind',
            'TFT6_Augment_ComponentGrabBag',
            'TFT6_Augment_CyberneticImplants1',
            'TFT6_Augment_CyberneticImplants2',
            'TFT6_Augment_CyberneticImplants3',
            'TFT6_Augment_CyberneticShell1',
            'TFT6_Augment_CyberneticShell2',
            'TFT6_Augment_CyberneticShell3',
            'TFT6_Augment_CyberneticUplink1',
            'TFT6_Augment_CyberneticUplink2',
            'TFT6_Augment_CyberneticUplink3',
            'TFT6_Augment_Distancing',
            'TFT6_Augment_Distancing2',
            'TFT6_Augment_Distancing3',
            'TFT6_Augment_Diversify1',
            'TFT6_Augment_Diversify2',
            'TFT6_Augment_Diversify3',
            'TFT6_Augment_Electrocharge1',
            'TFT6_Augment_Electrocharge2',
            'TFT6_Augment_Electrocharge3',
            'TFT6_Augment_Featherweights1',
            'TFT6_Augment_Featherweights2',
            'TFT6_Augment_Featherweights3',
            'TFT6_Augment_FirstAidKit',
            'TFT6_Augment_ForceOfNature',
            'TFT6_Augment_FuturePeepers',
            'TFT6_Augment_FuturePeepers2',
            'TFT6_Augment_GachaAddict',
            'TFT6_Augment_GrandGambler',
            'TFT6_Augment_HyperRoll',
            'TFT6_Augment_ItemGrabBag1',
            'TFT6_Augment_ItemGrabBag2',
            'TFT6_Augment_JeweledLotus',
            'TFT6_Augment_LudensEcho1',
            'TFT6_Augment_LudensEcho2',
            'TFT6_Augment_LudensEcho3',
            'TFT6_Augment_MeleeStarBlade1',
            'TFT6_Augment_MeleeStarBlade2',
            'TFT6_Augment_MeleeStarBlade3',
            'TFT6_Augment_MetabolicAccelerator',
            'TFT6_Augment_PandorasItems',
            'TFT6_Augment_PortableForge',
            'TFT6_Augment_RadiantRelics',
            'TFT6_Augment_Recombobulator',
            'TFT6_Augment_RichGetRicher',
            'TFT6_Augment_RichGetRicherPlus',
            'TFT6_Augment_SalvageBin',
            'TFT6_Augment_SecondWind1',
            'TFT6_Augment_SecondWind2',
            'TFT6_Augment_SlowAndSteady',
            'TFT6_Augment_SunfireBoard',
            'TFT6_Augment_TheGoldenEgg',
            'TFT6_Augment_TheGoldenEggHR',
            'TFT6_Augment_ThreesCompany',
            'TFT6_Augment_ThriftShop',
            'TFT6_Augment_ThrillOfTheHunt1',
            'TFT6_Augment_ThrillOfTheHunt2',
            'TFT6_Augment_TinyTitans',
            'TFT6_Augment_TomeOfTraits1',
            'TFT6_Augment_TradeSector',
            'TFT6_Augment_TradeSectorPlus',
            'TFT6_Augment_Traitless1',
            'TFT6_Augment_Traitless2',
            'TFT6_Augment_Traitless3',
            'TFT6_Augment_TriForce1',
            'TFT6_Augment_TriForce2',
            'TFT6_Augment_TriForce3',
            'TFT6_Augment_TrueTwos',
            'TFT6_Augment_Twins1',
            'TFT6_Augment_Twins2',
            'TFT6_Augment_Twins3',
            'TFT6_Augment_VerdantVeil',
            'TFT6_Augment_Windfall',
            'TFT6_Augment_WindfallPlus',
            'TFT6_Augment_WindfallPlusPlus',
            'TFT6_Augment_WoodlandCharm',
            'TFT7_Augment_AFK',
            'TFT7_Augment_AgeOfDragons',
            'TFT7_Augment_AssassinEmblem2',
            'TFT7_Augment_AssassinTrait',
            'TFT7_Augment_AstralIntercosmicProtection',
            'TFT7_Augment_AxiomArc1',
            'TFT7_Augment_AxiomArc2',
            'TFT7_Augment_BandOfThieves1',
            'TFT7_Augment_BestFriends1',
            'TFT7_Augment_BestFriends2',
            'TFT7_Augment_BestFriends3',
            'TFT7_Augment_BigFriend',
            'TFT7_Augment_BigFriend2',
            'TFT7_Augment_BirthdayPresents',
            'TFT7_Augment_Bloodlust1',
            'TFT7_Augment_BruiserEmblem',
            'TFT7_Augment_BruiserEmblem2',
            'TFT7_Augment_BruiserPersonalTraining',
            'TFT7_Augment_BruiserTitanicStrength',
            'TFT7_Augment_BruiserTrait',
            'TFT7_Augment_CannoneerEmblem',
            'TFT7_Augment_CannoneerEmblem2',
            'TFT7_Augment_CannoneerHotShot',
            'TFT7_Augment_CannoneerRicochet',
            'TFT7_Augment_CannoneerTrait',
            'TFT7_Augment_CavalierDevastatingCharge',
            'TFT7_Augment_CavalierEmblem',
            'TFT7_Augment_CavalierEmblem2',
            'TFT7_Augment_CavalierForAllUnits',
            'TFT7_Augment_CavalierTrait',
            'TFT7_Augment_ClutteredMind',
            'TFT7_Augment_Consistency',
            'TFT7_Augment_CursedCrown',
            'TFT7_Augment_DarkflightEmblem',
            'TFT7_Augment_DarkflightEmblem2',
            'TFT7_Augment_DarkflightSoulSiphon',
            'TFT7_Augment_DarkflightTrait',
            'TFT7_Augment_DragonImperialist',
            'TFT7_Augment_DragonTrait2',
            'TFT7_Augment_DragonmancerConference',
            'TFT7_Augment_DragonmancerEmblem',
            'TFT7_Augment_DragonmancerEmblem2',
            'TFT7_Augment_DragonmancerInTraining',
            'TFT7_Augment_EvokerEmblem',
            'TFT7_Augment_EvokerEmblem2',
            'TFT7_Augment_EvokerTrait',
            'TFT7_Augment_FirstAidKit2',
            'TFT7_Augment_GadgetExpert',
            'TFT7_Augment_GuardianEmblem',
            'TFT7_Augment_GuardianEmblem2',
            'TFT7_Augment_GuardianHeroicPresence',
            'TFT7_Augment_GuardianTrait',
            'TFT7_Augment_GuildEmblem',
            'TFT7_Augment_GuildEmblem2',
            'TFT7_Augment_GuildGearUpgrades',
            'TFT7_Augment_GuildLoot',
            'TFT7_Augment_GuildTrait',
            'TFT7_Augment_JadeEternalProtection',
            'TFT7_Augment_JadePenitence',
            'TFT7_Augment_JadeTrait',
            'TFT7_Augment_JadeTrait2',
            'TFT7_Augment_LagoonEmblem',
            'TFT7_Augment_LagoonEmblem2',
            'TFT7_Augment_LagoonHighTide',
            'TFT7_Augment_LagoonOasis',
            'TFT7_Augment_LagoonTrait',
            'TFT7_Augment_LastStand',
            'TFT7_Augment_LategameSpecialist',
            'TFT7_Augment_LivingForge',
            'TFT7_Augment_LuckyGloves',
            'TFT7_Augment_MageEmblem',
            'TFT7_Augment_MageEmblem2',
            'TFT7_Augment_MageEssenceTheft',
            'TFT7_Augment_MageTrait',
            'TFT7_Augment_MikaelsGift',
            'TFT7_Augment_MirageEmblem',
            'TFT7_Augment_MirageEmblem2',
            'TFT7_Augment_MirageHallucinate',
            'TFT7_Augment_MirageTrait',
            'TFT7_Augment_MysticTrait',
            'TFT7_Augment_MysticTrait2',
            'TFT7_Augment_PandorasBench',
            'TFT7_Augment_Preparation',
            'TFT7_Augment_Preparation2',
            'TFT7_Augment_Preparation3',
            'TFT7_Augment_RagewingScorch',
            'TFT7_Augment_RagewingTantrum',
            'TFT7_Augment_RagewingTrait',
            'TFT7_Augment_RagewingTrait2',
            'TFT7_Augment_SacrificialPact',
            'TFT7_Augment_ScalescornBaseCamp',
            'TFT7_Augment_ScalescornEmblem',
            'TFT7_Augment_ScalescornEmblem2',
            'TFT7_Augment_ScalescornTrait',
            'TFT7_Augment_ScopedWeapons1',
            'TFT7_Augment_ScopedWeapons2',
            'TFT7_Augment_ShapeshifterBeastsDen',
            'TFT7_Augment_ShapeshifterTrait',
            'TFT7_Augment_ShapeshifterTrait2',
            'TFT7_Augment_ShimmerscaleEmblem',
            'TFT7_Augment_ShimmerscaleSpending',
            'TFT7_Augment_ShimmerscaleTrait',
            'TFT7_Augment_ShimmerscaleTrait2',
            'TFT7_Augment_SwiftshotEmblem',
            'TFT7_Augment_SwiftshotEmblem2',
            'TFT7_Augment_SwiftshotPressTheAttack',
            'TFT7_Augment_SwiftshotTrait',
            'TFT7_Augment_TempestEmblem',
            'TFT7_Augment_TempestEmblem2',
            'TFT7_Augment_TempestEyeOfTheStorm',
            'TFT7_Augment_TempestTrait',
            'TFT7_Augment_ThinkFast',
            'TFT7_Augment_TomeOfTraits2',
            'TFT7_Augment_UrfsGrabBag2',
            'TFT7_Augment_WarriorEmblem',
            'TFT7_Augment_WarriorEmblem2',
            'TFT7_Augment_WarriorTiamat',
            'TFT7_Augment_WarriorTrait',
            'TFT7_Augment_WhispersTerrify',
            'TFT7_Augment_WhispersTrait']

        with open('./data/del_metaList2.txt', "r") as f:
            self.del_metaList = f.readlines()

        for i in range(len(self.del_metaList)):
            self.del_metaList[i] = re.sub("\n", "", metaList[i] )

        self.X = []
        self.label = []

        self.Combination_Item = ['TFT_Item_BFSword', 'TFT_Item_ChainVest', 'TFT_Item_GiantsBelt', 
        'TFT_Item_NeedlesslyLargeRod', 'TFT_Item_NegatronCloak', 'TFT_Item_RecurveBow', 
        'TFT_Item_SparringGloves', 'TFT_Item_Spatula', 'TFT_Item_TearOfTheGoddess']
        # 사용자가 사용한 데이터는 엠블럼 등이 있음 그거 배제
        cnt=0
        for i in self.rdr:
            _dict = json.loads(i['info.participants'].replace("'", "\""))
            for j in _dict:
                self.X_i = []
                # self.traits = []
                # self.items = []
                self.X_i.extend(j['augments'])
                for k in j['traits']:
                    if  k['name'] + str(k['style']) in self.MetaData_item:
                        self.X_i.append(k['name'] + str(k['style']))
                for h in j['units']:
                    if h['itemNames'] != []:
                        self.item_list = list(h['itemNames'])
                        if 'TFT_Item_ThiefsGloves' in self.item_list:
                            self.X_i.append('TFT_Item_ThiefsGloves')
                        else:
                            for g in range(len(h['itemNames']), -1):
                                # if (item_list[g] not in self.MetaData_item) or (item_list[g] in self.del_metaList):
                                #     del item_list[g]
                                if self.item_list[g] in self.MetaData_item:
                                    self.X_i.append(self.item_list[g])

                
                # self.X_i.extend(self.traits)
                # self.X_i.extend(self.item_list)
                if 'TFT7_Item_GuardianEmblemItem' in self.X_i:
                    print('Wsadncdasuidnasid')
                self.label.append(j['placement'])
                self.X.append(self.X_i)
            cnt+=1
            if cnt%500==0:
                print(cnt)
            # if cnt > 2550: 
            #     print()
            #     break
        ###### Data Processing fixed.py END
        print('end')
        ###### one-hot-encoding START
        # def one_hot_encoder_2dimList(x_data: list) -> list:
        self.one_hot_encoder_2dimList(self.X) # self.res 사용하세요
        print('encoding end')
        ###### one-hot-encoding END
        print(len(self.res))
        print(len(self.X))
        print(self.res[0])
        
    def __len__(self):
        return len(self.res  )
    
    def __getitem__(self, index) -> tuple([list, int]):
        x_ = self.res[index]
        y_ = self.label[index]
        # transform func
        x_ = transforms.ToTensor()
        y_ = transforms.ToTensor()
        
        return x_, y_ # tuple([7 of tensors], family_id)
        # if self.transform is not None:
        #     img = self.transform(img)

    def one_hot_encoder_2dimList(self, x_data: list) -> list:
        # self.res = [ [ 1 if i in row_x else 0 for i in self.MetaData_item ] for row_x in x_data ] 
        self.res = [ [ [ 1 if row_x_i==i_th_item else 0 for i_th_item in self.MetaData_item] for row_x_i in row_x ] \
            for row_x in x_data ] 
        # print(x_data[0])
        # with open('./data/metaList2.txt', "r") as f:
        #     self.metaList = f.readlines()

        # for i in range(len(self.metaList)):
        #     self.metaList[i] = re.sub("\n", "", self.metaList[i] )
        # print(type(self.metaList))
        # print(type(self.metaList[0]))
        # cnt = 0
        # self.one_hot_2dimList = [ [0 for i in range(358)] for i in range(len(x_data)) ]
        # [ [ 1 if i in row_x else 0 for i in self.MetaData_item ] for i in range(len(x_data)) ] 
        # for row_vec in self.one_hot_2dimList:
        #     if cnt%500==0:
        #         print(cnt)
        #     for row_x in x_data:
        #         for x_ in row_x:
        #             # print('x', x_)
        #             # print('index',self.metaList.index(x_))
        #             # print(row_vec[self.metaList.index(x_)])
        #             row_vec[self.MetaData_item.index(x_)] = 1
        #         self.res.append(row_vec)
    
if __name__ == '__main__':
    with open('./data/metaList2.txt', "r") as f:
        metaList = f.readlines()
    
    for i in range(len(metaList)):
        metaList[i] = re.sub("\n", "", metaList[i] )

    

    instance = My_own_dataset("./data/match_data/", transform=transforms.ToTensor())
    print(instance.__getitem__(0))
    # print(instance.__getitem__(0).size()) # return 넣으면 풀기
    print('instance len:',instance.__len__())
    
    