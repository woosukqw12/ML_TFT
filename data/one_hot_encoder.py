import pandas as pd
import re

with open('data/metaList2.txt', "r") as f:
    metaList = f.readlines()
    
for i in range(len(metaList)):
    metaList[i] = re.sub("\n", "", metaList[i] )
    
def one_hot_encoder(x_data: list) -> list:
    global metaList
    one_hot = [0 for i in range(358)]
    for x in x_data:
        one_hot[metaList.index(x)] = 1
        
    return one_hot

def one_hot_encoder_2dimList(x_data: list) -> list:
    global metaList
    one_hot_2dimList = [ [0 for i in range(358)] for i in range(len(x_data)) ]
    for row_vec in one_hot_2dimList:
        for x in x_data:
            row_vec[metaList.index(x)] = 1
        
    return one_hot_2dimList

def one_hot_encoder_dataframe(x_data: pd.DataFrame) -> pd.DataFrame:
    global metaList
    one_hot_dataFrame = pd.DataFrame()
    for x in x_data:
        one_hot = [0 for i in range(358)]
        for x in x_data:
            one_hot[metaList.index(x)] = 1
        
    return one_hot_dataFrame