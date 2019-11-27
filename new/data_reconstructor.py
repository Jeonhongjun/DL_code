import pandas as pd
import numpy as np

def old_contents_search(x, cursor):

    old_list = []

    if str(x)[6:8] != '01':
        sql = "SELECT * FROM CJ.VOD" + str(x)[:6]
        cursor.execute(sql)
        df1 = pd.read_sql(sql, db)
        temp = df1[df1.index < int(x)]
        old = list(temp.EPISODE.unique())
        old_list.append(old)
    else:
        pass

    while str(x)[4:6] != '01':
        x = int(x) - 100
        sql = "SELECT * FROM CJ.VOD" + str(x)[:6]
        cursor.execute(sql)
        df1 = pd.read_sql(sql, db)
        old = list(df1.EPISODE.unique())
        old_list.append(old)

    old_list = np.asarray(old_list)
    old_list = old_list.T
    old_list = old_list.tolist()
    old_list = sum(old_list, [])
    old_list = pd.DataFrame(old_list)
    old_list = old_list[0].unique()
    old_list = old_list.tolist()

    return old_list

def new_contents_search(x, cursor):
    if str(x)[6:8] != '01':
        new_list = []
        sql = "SELECT * FROM CJ.VOD" + str(x)[:6]
        cursor.execute(sql)
        df1 = pd.read_sql(sql, db)
        df1 = df1[df1.index < int(x) + 2]
        df1 = df1[df1.index > int(x) - 1]
    else:
        new_list = []
        df1 = pd.read_sql(sql, db)
        df1 = df1[df1.index < int(x) + 2]

    new_list = list(df1.EPISODE.unique())

    return new_list


def new_data_gen(x, cursor):
    sql = "SELECT * FROM CJ.VOD" + str(x)[:6]
    cursor.execute(sql)
    new_data = pd.read_sql(sql, db)
    new_data = new_data[new_data.index < int(x) + 2]
    new_data = new_data[new_data.index > int(x) - 1]

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
    for i in new_data.EPISODE.unique():
        b = new_data[new_data["EPISODE"] == i].shape[0]
        a.append(b)

    Label = pd.DataFrame(a)

    Label.to_csv('Label.csv', encoding='utf-8')

    Label = pd.read_csv('./Label.csv', header=0)

    Label = Label.drop('Unnamed: 0', axis = 1)

    Label.columns = ['ViewCount']

    contents = pd.DataFrame(new_data.EPISODE.unique(), columns = ['EPISODE'])

    reconst = pd.concat([contents, Label], axis = 1)

    c= []
    for i in new_data.EPISODE.unique():
        d = new_data[new_data["EPISODE"] == i].PROGRAM.unique()
        c.append(d)

    for i in range(len(c)):
        c[i] = c[i].tolist()[0]

    Program = pd.DataFrame(c, columns = ['Program'])

    reconst = pd.concat([reconst, Program], axis = 1)

    new_data.loc[new_data["Payment"] == 'ADULT', "Payment"] = 'PRE'

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    g = list(new_data['Payment'].values)
    le.fit(g)
    g = le.transform(g)
    g = pd.DataFrame(g, columns = ['Pay_Label'])

    EPISODE = pd.DataFrame(new_data.EPISODE.values, columns = ['EPISODE'])
    Payment = pd.concat([EPISODE, g], axis = 1)

    Payment.loc[Payment["Pay_Label"] == 2, "Pay_Label"] = -1
    Payment.loc[Payment["Pay_Label"] == 1, "Pay_Label"] = 2
    Payment.loc[Payment["Pay_Label"] == 0, "Pay_Label"] = 1
    Payment.loc[Payment["Pay_Label"] == -1, "Pay_Label"] = 0

    h= []
    for i in Payment.EPISODE.unique():
        i = Payment[Payment["EPISODE"] == i].Pay_Label.mean()
        h.append(i)

    PAYMENT = pd.DataFrame(h, columns = ['PAYMENT'])

    reconst = pd.concat([reconst, PAYMENT], axis = 1)

    m = list(new_data['PROGRAM_TYPE'].values)

    le.fit(m)
    m = le.transform(m)

    m = pd.DataFrame(m, columns = ['PROGRAM_TYPE'])

    m = pd.concat([EPISODE, m], axis = 1)

    o = []
    for i in m.EPISODE.unique():
        p = m[m["EPISODE"] == i].PROGRAM_TYPE.mean()
        o.append(p)

    o = pd.DataFrame(o, columns = ['PROGRAM_TYPE'])
    len(o)
    reconst = pd.concat([reconst, o], axis = 1)

    temp = []
    temp2 = []
    day = []
    month = []

    for i in new_data.EPISODE.unique():
        for j in new_data[new_data["EPISODE"] == i].index.tolist():
            mon = str(j)[4:6]
            mon = int(mon)
            da = str(j)[6:8]
            da = int(da)
            temp.append(mon)
            temp2.append(da)
        month.append(round(sum(temp) / float(len(temp))))
        day.append(round(sum(temp2) / float(len(temp2))))

    p = pd.DataFrame(month, columns = ['month'])
    q = pd.DataFrame(day, columns = ['day'])

    reconst = pd.concat([reconst, p], axis = 1)
    reconst = pd.concat([reconst, q], axis = 1)

    time = []
    for i in new_data.EPISODE.unique():
        ti = round(new_data[new_data["EPISODE"] == i].HMS.mean())
        if ti < 10000:
            ti = str(ti)
            ti = '00' + ti
        elif ti < 100000:
            ti = str(ti)
            ti = '0' + ti
        else:
            ti = str(ti)
        time.append(int(ti[:2]))

    time = pd.DataFrame(time, columns = ['time'])
    reconst = pd.concat([reconst, time], axis = 1)

    reconst = reconst[['EPISODE', 'month', 'day', 'time', 'Program', 'PAYMENT', 'PROGRAM_TYPE', 'ViewCount']]

    reconst = pd.concat([reconst, new_contents], axis = 1)

    reconst = reconst.sort_values(by=['ViewCount'], ascending=False)

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

    reconst.to_csv('data/'+ str(dates) +'.csv', encoding='utf-8')

    return reconst

def episode(date):
    contents = pd.read_csv('./data/' + str(date) +'.csv', header=0, index_col=0)

    episode = []
    for i in contents.Program:
        episode.append(len(contents[contents['Program'] == i].Program.values))
    episode = pd.DataFrame(episode, columns = ['episode_count'])
    contents = pd.concat([contents, episode], axis = 1)
    contents = contents[contents['episode_count']<500]
    contents.index = range(len(contents))
    contents.to_csv('./data/'+ str(date) +'.csv', encoding='utf-8')
