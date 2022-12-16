from time import sleep
import requests
import json
from pandas import json_normalize
import pandas as pd
from pandas import DataFrame
import numpy as np

import pickle
from datetime import date
from requests.adapters import Retry, HTTPAdapter

key = "RGAPI-ff3f9cd8-0265-442b-a841-003126f2791f"
today = date.today().isoformat()


def league_summoner(key, tier, country="kr"):
    request = requests.get(
        f"https://{country}.api.riotgames.com/tft/league/v1/{tier}?api_key={key}"
    )
    return json.loads(request.content)


def summoner_info(summonerId, key, country="kr"):  # by_name
    try:
        request = requests.get(
            f"https://{country}.api.riotgames.com/tft/summoner/v1/summoners/{summonerId}?api_key={key}",
            timeout=5,
        )
    except requests.Timeout:
        for i in range(5):
            try:
                print(f"Timeout. {i+1}-th Retry...")
                request = requests.get(
                    f"https://{region}.api.riotgames.com/tft/match/v1/matches/by-puuid/{i}/ids?count={n}&api_key={key}",
                    timeout=5,
                )
                break
            except requests.Timeout:
                pass

    if request.status_code == 429:
        print("Rate limit exceeded!")
        sleep(120)
        try:
            request = requests.get(
                f"https://{country}.api.riotgames.com/tft/summoner/v1/summoners/{summonerId}?api_key={key}",
                timeout=5,
            )
        except requests.Timeout:
            for i in range(5):
                try:
                    print(f"Timeout. {i+1}-th Retry...")
                    request = requests.get(
                        f"https://{region}.api.riotgames.com/tft/match/v1/matches/by-puuid/{i}/ids?count={n}&api_key={key}",
                        timeout=5,
                    )
                    break
                except requests.Timeout:
                    pass

    return json.loads(request.content)


def summoner_info_retry(summonerId, key, country="kr"):  # by_name
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=0, status_forcelist=[502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    request = s.get(
        f"https://{country}.api.riotgames.com/tft/summoner/v1/summoners/{summonerId}?api_key={key}"
    )
    return json.loads(request.content)


def load_summonerId():
    with open(f"summoner_2022-11-13.pickle", "rb") as f:
        summonerId = pickle.load(f)
    return summonerId


def summoner_puuid(summonerId):
    summ_puuid = pd.DataFrame()
    step = 0
    row_idx = 0
    for id in summonerId:
        step += 1
        si = summoner_info(id, key, "kr")
        si = json_normalize(si)
        try:
            summ_puuid = pd.concat([summ_puuid, si["puuid"]], ignore_index=True)
        except KeyError as e:
            print(id)
            continue
        row_idx += 1
        print(f"step {step} done! : puuid = " + si["puuid"])

    return summ_puuid


def get_puuid(summonerId):
    for i in range(0, 15):
        print(i)
        puuid = summoner_puuid(summonerId[i])
        print(f"{i}-th iter done!")
        with open(f"C:/Users/edwar/TFTML/data/puuid_{i}.pickle", "wb") as f:
            pickle.dump(puuid, f)


def get_matchid(puuid, key, n, region="asia"):
    matchid = []
    step = 0
    for i in puuid:
        step += 1
        try:
            request = requests.get(
                f"https://{region}.api.riotgames.com/tft/match/v1/matches/by-puuid/{i}/ids?count={n}&api_key={key}",
                timeout=5,
            )
        except requests.Timeout:
            for i in range(5):
                try:
                    print(f"Timeout. {i+1}-th Retry...")
                    request = requests.get(
                        f"https://{region}.api.riotgames.com/tft/match/v1/matches/by-puuid/{i}/ids?count={n}&api_key={key}",
                        timeout=5,
                    )
                    break
                except requests.Timeout:
                    pass

        if request.status_code == 429:
            print("Rate limit exceeded!")
            sleep(120)
            try:
                request = requests.get(
                    f"https://{region}.api.riotgames.com/tft/match/v1/matches/by-puuid/{i}/ids?count={n}&api_key={key}",
                    timeout=5,
                )
            except requests.Timeout:
                request = requests.get(
                    f"https://{region}.api.riotgames.com/tft/match/v1/matches/by-puuid/{i}/ids?count={n}&api_key={key}",
                    timeout=5,
                )
        request = json.loads(request.content)
        print(request)
        matchid.extend(request)
        print(f"step {step} done!")
    return list(set(matchid))


def match_info(matchid, key, region="asia"):
    game_record = pd.DataFrame()
    step = 0
    for i in matchid:
        step += 1
        try:
            request = requests.get(
                f"https://{region}.api.riotgames.com/tft/match/v1/matches/{i}?api_key={key}",
                timeout=5,
            )
        except requests.Timeout:
            for i in range(5):
                try:
                    print(f"Timeout. {i+1}-th Retry...")
                    request = requests.get(
                        f"https://{region}.api.riotgames.com/tft/match/v1/matches/{i}?api_key={key}",
                        timeout=5,
                    )
                    break
                except requests.Timeout:
                    pass

        if request.status_code == 429:
            print("Rate limit exceeded!")
            sleep(120)
            try:
                request = requests.get(
                    f"https://{region}.api.riotgames.com/tft/match/v1/matches/{i}?api_key={key}",
                    timeout=5,
                )
            except requests.Timeout:
                for i in range(5):
                    try:
                        print(f"Timeout. {i+1}-th Retry...")
                        request = requests.get(
                            f"https://{region}.api.riotgames.com/tft/match/v1/matches/{i}?api_key={key}",
                            timeout=5,
                        )
                    except requests.Timeout:
                        pass
        request = json.loads(request.content)
        request = json_normalize(request)
        print(request)
        game_record = pd.concat([game_record, request])
        print(f"step {step} done!")
    return game_record


"""
summonerId = load_summonerId()
print(summonerId)

ch_matchid_list = []
for i in range(0, 15):
    with open(f"C:/Users/edwar/TFTML/data/puuid_{i}.pickle", "rb") as f:
        puuid = pickle.load(f)
    puuid_list = puuid.values.tolist()
    puuid_list = [i[0] for i in puuid_list]
    ch_matchid = get_matchid(puuid_list, key, 30)
    print(f"{i}-th iter done!")
    with open(f"C:/Users/edwar/TFTML/data/matchid_{i}.pickle", "wb") as f:
        pickle.dump(ch_matchid, f)
"""

for i in range(22, 31):
    with open(
        f"C:/Users/edwar/TFTML/data/merged_match_id/match_id{i}.pickle", "rb"
    ) as f:
        matchid = pickle.load(f)
    match_data = match_info(matchid, key, "asia")
    with open(f"C:/Users/edwar/TFTML/data/match_data/match_data{i}.pickle", "wb") as f:
        pickle.dump(match_data, f)