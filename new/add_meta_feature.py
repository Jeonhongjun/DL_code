import pandas as pd
import numpy as np
import json

def meta_genre(x):

    file = open('meta/program.json','r')

    dic = []
    for line in file.readlines():
        json_line = json.loads(line)
        dic.append(json_line)

    temp = []

    for i in range(len(dic)):
        try:
            temp.append(dic[i]['meta']['pip']['raw']['detailgenre'][0]['gnrdtl_nm'])
        except IndexError:
            temp.append('NA')
        except KeyError:
            temp.append('NA')

    temp2 = []

    for i in range(len(dic)):
        try:
            temp2.append(dic[i]['eid']['cj:tving:contentid'])
        except IndexError:
            temp2.append('NA')
        except KeyError:
            temp2.append('NA')

    meta2 = pd.DataFrame(temp2, columns = ['program'])
    meta = pd.DataFrame(temp, columns = ['genre'])

    meta = pd.concat([meta2, meta], axis = 1)

    contents = pd.read_csv('./data/' + str(x) + '.csv', header=0, index_col=0)

    genre = []
    for i in range(contents.count()[0]):
        if np.any(meta[meta['program'] == contents['Program'][i]]['genre'].values != None):
            genre.append(meta[meta['program'] == contents['Program'][i]]['genre'].values[0])
        else :
            genre.append('NA')

    genre = pd.DataFrame(genre, columns = ['genre'])
    genre['genre'].unique()

    contents = pd.concat([contents, genre], axis = 1)

    Labeling = list(contents['genre'].values)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    le.fit(Labeling)
    Labeling = le.transform(Labeling)

    Labeling = pd.DataFrame(Labeling, columns = ['genre_Label'])

    contents = pd.concat([contents, Labeling], axis = 1)

    contents.to_csv('./data/'+ str(x) +'.csv', encoding='utf-8')

def meta_age(x):

    file = open('meta/episode.json','r')

    dic = []
    for line in file.readlines():
        temp = json.loads(line)
        dic.append(temp)

    temp = []

    for i in range(len(dic)):
        try:
            temp.append(dic[i]['meta']['pip']['code']['targetage'])
        except IndexError:
            temp.append('NA')
        except KeyError:
            temp.append('NA')

    temp2 = []

    for i in range(len(dic)):
        try:
            temp2.append(dic[i]['eid']['cj:tving:contentid'])
        except IndexError:
            temp2.append('NA')
        except KeyError:
            temp2.append('NA')

    meta2 = pd.DataFrame(temp2, columns = ['program'])
    meta = pd.DataFrame(temp, columns = ['target_age'])
    meta = pd.concat([meta2, meta], axis = 1)

    contents = pd.read_csv('./data/'+ str(x) +'.csv', header=0, index_col=0)

    target_age = []

    for i in range(contents.count()[0]):
        if np.any(meta[meta['program'] == contents['EPISODE'][i]]['target_age'].values != None):
            target_age.append(meta[meta['program'] == contents['EPISODE'][i]]['target_age'].values[0])
        else :
            target_age.append('NA')

    target_age = pd.DataFrame(target_age, columns = ['target_age'])

    contents = pd.concat([contents, target_age], axis = 1)

    contents.loc[contents["target_age"] == 19, "target_age"] = 5
    contents.loc[contents["target_age"] == 15, "target_age"] = 4
    contents.loc[contents["target_age"] == 12, "target_age"] = 3
    contents.loc[contents["target_age"] == 7, "target_age"] = 2
    contents.loc[contents["target_age"] == 0, "target_age"] = 1
    contents.loc[contents["target_age"] == 'NA', "target_age"] = 0

    contents = contents.dropna(subset = ["target_age"])
    contents.index = range(0, contents.count()[0])


    contents.to_csv('./data/'+ str(x) +'.csv', encoding='utf-8')

def meta_playtime(x):

    file = open('meta/episode.json','r')

    dic = []
    for line in file.readlines():
        temp = json.loads(line)
        dic.append(temp)

    temp = []

    for i in range(len(dic)):
        try:
            temp.append(dic[i]['meta']['stat']['playtime'])
        except IndexError:
            temp.append('NA')
        except KeyError:
            temp.append('NA')

    temp2 = []

    for i in range(len(dic)):
        try:
            temp2.append(dic[i]['eid']['cj:tving:contentid'])
        except IndexError:
            temp2.append('NA')
        except KeyError:
            temp2.append('NA')

    meta2 = pd.DataFrame(temp2, columns = ['program'])
    meta = pd.DataFrame(temp, columns = ['playtime'])
    meta = pd.concat([meta2, meta], axis = 1)

    meta[meta['playtime'] == 'NA'].count()
    meta[meta['playtime'] != 'NA'].count()

    contents = pd.read_csv('./data/'+ str(x) +'.csv', header=0, index_col=0)

    playtime = []
    for i in range(contents.count()[0]):
        if np.any(meta[meta['program'] == contents['EPISODE'][i]]['playtime'].values != None):
            playtime.append(meta[meta['program'] == contents['EPISODE'][i]]['playtime'].values[0])
        else :
            playtime.append('NA')


    playtime = pd.DataFrame(playtime, columns = ['playtime'])

    contents = pd.concat([contents, playtime], axis = 1)
    contents.loc[contents["playtime"] == 'NA', "playtime"] = 0

    contents.to_csv('./data/'+ str(x) +'.csv', encoding='utf-8')

def meta_channel(x):

    file = open('meta/episode.json','r')

    dic = []
    for line in file.readlines():
        temp = json.loads(line)
        dic.append(temp)

    temp = []

    for i in range(len(dic)):
        try:
            temp.append(dic[i]['meta']['pip']['code']["channelid"])
        except KeyError:
            temp.append('NA')
        except IndexError:
            temp.append('NA')

    temp2 = []

    for i in range(len(dic)):
        try:
            temp2.append(dic[i]['eid']['cj:tving:contentid'])
        except IndexError:
            temp2.append('NA')
        except KeyError:
            temp2.append('NA')

    meta2 = pd.DataFrame(temp2, columns = ['program'])
    meta = pd.DataFrame(temp, columns = ['channel'])
    meta = pd.concat([meta2, meta], axis = 1)

    contents = pd.read_csv('./data/'+ str(x) +'.csv', header=0, index_col=0)

    channel = []
    for i in range(contents.count()[0]):
        if np.any(meta[meta['program'] == contents['EPISODE'][i]]['channel'].values != None):
            channel.append(meta[meta['program'] == contents['EPISODE'][i]]['channel'].values[0])
        else :
            channel.append('NA')

    channel = pd.DataFrame(channel, columns = ['channel'])

    contents = pd.concat([contents, channel], axis = 1)

    Labeling = list(contents['channel'].values)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(Labeling)
    Labeling = le.transform(Labeling)
    Labeling = pd.DataFrame(Labeling, columns = ['channel_Label'])

    contents = pd.concat([contents, Labeling], axis = 1)

    contents.to_csv('./data/'+ str(x) +'.csv', encoding='utf-8')

def meta_contentnumber(x):

    file = open('meta/episode.json','r')

    dic = []
    for line in file.readlines():
        temp = json.loads(line)
        dic.append(temp)

    temp = []

    for i in range(len(dic)):
        try:
            temp.append(dic[i]['meta']['pip']['code']['contentnumber'])
        except IndexError:
            temp.append('NA')
        except KeyError:
            temp.append('NA')

    temp2 = []

    for i in range(len(dic)):
        try:
            temp2.append(dic[i]['eid']['cj:tving:contentid'])
        except IndexError:
            temp2.append('NA')
        except KeyError:
            temp2.append('NA')

    meta2 = pd.DataFrame(temp2, columns = ['program'])
    meta = pd.DataFrame(temp, columns = ['contentnumber'])
    meta = pd.concat([meta2, meta], axis = 1)

    contents = pd.read_csv('./data/'+ str(x) +'.csv', header=0, index_col=0)

    contentnumber = []

    for i in range(contents.count()[0]):
        if np.any(meta[meta['program'] == contents['EPISODE'][i]]['contentnumber'].values != None):
            contentnumber.append(meta[meta['program'] == contents['EPISODE'][i]]['contentnumber'].values[0])
        else :
            contentnumber.append('NA')

    contentnumber = pd.DataFrame(contentnumber, columns = ['contentnumber'])

    contents = pd.concat([contents, contentnumber], axis = 1)

    contents.index = range(0, contents.count()[0])

    contents.to_csv('./data/'+ str(x) +'.csv', encoding='utf-8')
