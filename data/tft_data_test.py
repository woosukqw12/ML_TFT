from time import sleep
import requests
import json
from pandas import json_normalize
import pandas as pd

key = "RGAPI-79219110-eed6-4cc3-9f3f-7613a4d90c4f"

game_record = pd.DataFrame()


def summoner_info(summonerId, key, country="kr"):
    request = requests.get(
        f"https://{country}.api.riotgames.com/tft/summoner/v1/summoners/by-name/{summonerId}?api_key={key}"
    )
    return json.loads(request.content)


def match_info(matchid, key, region="asia"):
    l = 100
    global game_record
    for i in matchid:
        l += 1
        if l % 100 == 0:
            sleep(120)
        try:
            request = requests.get(
                f"https://{region}.api.riotgames.com/tft/match/v1/matches/{i}?api_key={key}"
            )
            request = json.loads(request.content)
            request = json_normalize(request)
            game_record = pd.concat([game_record, request])
        except:
            pass
    return game_record


def get_matchid(puuid, key, n, region="asia"):
    matchid = []
    l = 100
    for i in puuid:
        l += 1
        if l % 100 == 0:
            sleep(120)
        try:
            request = requests.get(
                f"https://{region}.api.riotgames.com/tft/match/v1/matches/by-puuid/{i}/ids?count={n}&api_key={key}"
            )
            request = json.loads(request.content)
            matchid.extend(request)
        except:
            pass
    return list(set(matchid))


ret = summoner_info("HyperN0vA", key)
puuid = [ret["puuid"]]
matchid = get_matchid(puuid, key, 1)
print(matchid)
data = match_info(matchid, key)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
print(data)