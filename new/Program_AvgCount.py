import pandas as pd
import numpy as np
import time
from tqdm import tqdm

def old_contents(x):

    old_list = []

    if str(x)[6:8] != '01':
        sql = "SELECT * FROM CJ.VOD" + str(x)[:6]
        cursor.execute(sql)
        df1 = pd.read_sql(sql, db)
        temp = df1[df1.index < x]
        old = list(temp.PROGRAM.unique())
        old_list.append(old)
    else:
        pass


    while str(x)[4:6] != '01':
        x = x - 100
        sql = "SELECT * FROM CJ.VOD" + str(x)[:6]
        cursor.execute(sql)
        df1 = pd.read_sql(sql, db)
        old = list(df1.PROGRAM.unique())
        old_list.append(old)

    old_list = np.asarray(old_list)
    old_list = old_list.T
    old_list = old_list.tolist()
    old_list = pd.DataFrame(old_list)
    old_list = old_list[0].unique()
    old_list = old_list.tolist()

    return old_list


def new_contents(x, Data):
    if str(x)[6:8] != '01':
        new_list = []
        Data = Data[Data.index < x + 2]
        Data = Data[Data.index > x - 1]
    else:
        Data = Data[Data.index < int(x) + 2]

    new_list = list(Data.PROGRAM.unique())

    return new_list

def new_data_gen(x, Data):
    new_data = Data[Data.index < x + 2]
    new_data = new_data[new_data.index > x - 1]

    return new_data


def new_contents_tagging(dates, old_list, new_list, new_data):
    new_contents = []
    for i in new_list:
        if i in old_list:
            new_contents.append(0)
        else:
            new_contents.append(1)

    new_contents = pd.DataFrame(new_contents, columns = ['New_Contents'])

    new_data.columns = ["HMS", "(DELETED)", "USER(HASHED)", "(DELETED).1", "LOG_TYPE", "Payment", "PROGRAM", "PROGRAM_TYPE", "EPISODE", "NA"]

    new_data.loc[(new_data["PROGRAM_TYPE"] == "DRAMA"), "PROGRAM_TYPE"] = 'VOD'

    a = []
    for i in new_data.PROGRAM.unique():
        b = new_data[new_data["PROGRAM"] == i].shape[0]
        a.append(b)

    Label = pd.DataFrame(a)

    Label.to_csv('Label.csv', encoding='utf-8')

    Label = pd.read_csv('./Label.csv', header=0)

    Label = Label.drop('Unnamed: 0', axis = 1)

    Label.columns = ['ViewCount']

    contents = pd.DataFrame(new_data.PROGRAM.unique(), columns = ['contents'])

    reconst = pd.concat([contents, Label], axis = 1)

    c= []
    for i in new_data.PROGRAM.unique():
        d = new_data[new_data["PROGRAM"] == i].EPISODE.unique().shape
        c.append(d)

    episode = pd.DataFrame(c, columns = ['Episode'])

    reconst = pd.concat([reconst, episode], axis = 1)

    m = list(new_data['PROGRAM_TYPE'].values)

    le.fit(m)
    m = le.transform(m)

    m = pd.DataFrame(m, columns = ['PROGRAM_TYPE'])

    m = pd.concat([PROGRAM, m], axis = 1)

    o = []
    for i in m.Program.unique():
        p = m[m["Program"] == i].PROGRAM_TYPE.mean()
        o.append(p)

    o = pd.DataFrame(o, columns = ['PROGRAM_TYPE'])

    reconst = pd.concat([reconst, o], axis = 1)

    reconst = reconst[['contents', 'Episode', 'PROGRAM_TYPE', 'ViewCount']]

    reconst = pd.concat([reconst, new_contents], axis = 1)

    reconst = reconst.sort_values(by=['ViewCount'], ascending=False)

    print("New contents count:",reconst[reconst['New_Contents'] == 1].shape[0])

    reconst.index = list(range(0,reconst.count()[0]))

    Hot_index = round(reconst.shape[0]/5)

    HC = []
    for i in range(reconst.shape[0]):
        if reconst.index[i] < (Hot_index + 1):
            HC.append('HOT')
        else:
            HC.append('COLD')

    HC = pd.DataFrame(HC, columns = ['H&C'])

    reconst = pd.concat([reconst, HC], axis =1)

    reconst.to_csv('Past_data/'+ str(dates) +'.csv', encoding='utf-8')

    contents = pd.read_csv('./Past_data/' + str(dates) +'.csv', header=0, index_col=0)

    past_hc = []
    for i in range(len(contents)):
        past_hc.append(contents["ViewCount"][i]/contents["Episode"][i])

    past_hc = pd.DataFrame(past_hc, columns = ["Past_count"])

    reconst = pd.concat([contents, past_hc], axis =1)

    reconst.to_csv('Past_data/'+ str(dates) +'.csv', encoding='utf-8')

    return reconst
