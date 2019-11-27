import pandas as pd
import numpy as np
import json
import ast

def hashing(x):
    contents = pd.read_csv('./data/'+ str(x) + '.csv', header=0, index_col=0)
    df = pd.read_csv('./data/hashmap.tsv', sep='\t', header=None)

    for i in range(contents.count()[0]):
        if np.any(df[df[0] == contents['EPISODE'][i]][1].values != None):
            contents['EPISODE'][i] = df[df[0] == contents['EPISODE'][i]][1].values[0]
        else :
            continue
    contents.to_csv('./data/'+ str(x) + '.csv', encoding='utf-8')

def hashing_pro(x):
    contents = pd.read_csv('./data/'+ str(x) + '.csv', header=0, index_col=0)
    df = pd.read_csv('./data/hashmap.tsv', sep='\t', header=None)

    for i in range(contents.count()[0]):
        if np.any(df[df[0] == contents['Program'][i]][1].values != None):
            contents['Program'][i] = df[df[0] == contents['Program'][i]][1].values[0]
        else :
            continue
    contents.to_csv('./data/'+ str(x) + '.csv', encoding='utf-8')


file = open('meta/program.json','r')

dic = []
for line in file.readlines():
    json_line = json.loads(line)
    dic.append(json_line)
