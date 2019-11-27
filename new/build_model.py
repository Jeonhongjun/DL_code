from flask_restful import Resource, Api
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import json
import pickle
import pymysql

def vod_model_save(variable, Data):
    pd.options.mode.chained_assignment = None

    contents = pd.DataFrame(Data, columns = ['PAYMENT', 'PROGRAM_TYPE', 'New_Contents', 'genre_Label', 'target_age', 'playtime', 'channel_Label', 'contentnumber', 'episode_count', 'past_view'])
    x_train = contents[contents['New_Contents'] == 0]
    x_train.loc[:,'PROGRAM_TYPE'] = round(x_train.loc[:,'PROGRAM_TYPE'])
    x_train = x_train[x_train['PROGRAM_TYPE'] == x_train.PROGRAM_TYPE.unique()[0]]
    x_train.contentnumber = x_train.contentnumber.fillna(0)
    x_train.episode_count = x_train.episode_count.fillna(1)
    x_train = x_train.drop('New_Contents', axis = 1)
    x_train = x_train.drop('PROGRAM_TYPE', axis = 1)
    x_train = x_train.values
    x_train = x_train.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x_train)

    contents = pd.DataFrame(Data, columns = ['PROGRAM_TYPE', 'New_Contents', 'ViewCount'])
    y_train = contents[contents['New_Contents'] == 0]
    y_train = y_train.drop('New_Contents', axis = 1)
    y_train.loc[:,'PROGRAM_TYPE'] = round(y_train.loc[:,'PROGRAM_TYPE'])
    y_train = y_train[y_train['PROGRAM_TYPE'] == y_train.PROGRAM_TYPE.unique()[0]]
    y_train = y_train.drop('PROGRAM_TYPE', axis = 1)
    y_train = y_train.values
    y_train = y_train.astype('float32')

    import xgboost as xgb

    xgb = xgb.XGBRegressor(colsample_bytree = 1,
     learning_rate =0.4,
     n_estimators=1000,
     max_depth=8,
     min_child_weight=1,
     max_delta_step = 2.5,
     gamma=1.0,
     subsample=0.8,
     objective = 'reg:linear',
     n_jobs=8,
     scale_pos_weight=1.8,
     random_state=27,
     base_score = 0.5)

    xgb.fit(x_train, y_train)

    filename = './data/finalized_model.sav'
    pickle.dump(xgb, open(filename, 'wb'))

def movie_model_save(variable, Data):

    pd.options.mode.chained_assignment = None

    contents = pd.DataFrame(Data, columns = ['PAYMENT', 'PROGRAM_TYPE', 'New_Contents', 'genre_Label', 'target_age', 'playtime', 'channel_Label', 'contentnumber', 'episode_count', 'past_view'])
    x_train = contents[contents['New_Contents'] == 0]
    x_train.loc[:,'PROGRAM_TYPE'] = round(x_train.loc[:,'PROGRAM_TYPE'])
    x_train = x_train[x_train['PROGRAM_TYPE'] != x_train.PROGRAM_TYPE.unique()[0]]
    x_train.contentnumber = x_train.contentnumber.fillna(0)
    x_train.episode_count = x_train.episode_count.fillna(1)
    x_train = x_train.drop('New_Contents', axis = 1)
    x_train = x_train.drop('PROGRAM_TYPE', axis = 1)
    x_train = x_train.values
    x_train = x_train.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x_train)

    contents = pd.DataFrame(Data, columns = ['PROGRAM_TYPE', 'New_Contents', 'ViewCount'])
    y_train = contents[contents['New_Contents'] == 0]
    y_train = y_train.drop('New_Contents', axis = 1)
    y_train.loc[:,'PROGRAM_TYPE'] = round(y_train.loc[:,'PROGRAM_TYPE'])
    y_train = y_train[y_train['PROGRAM_TYPE'] != y_train.PROGRAM_TYPE.unique()[0]]
    y_train = y_train.drop('PROGRAM_TYPE', axis = 1)
    y_train = y_train.values
    y_train = y_train.astype('float32')

    import xgboost as xgb

    xgb = xgb.XGBRegressor(colsample_bytree = 1,
     learning_rate =0.4,
     n_estimators=1000,
     max_depth=8,
     min_child_weight=1,
     max_delta_step = 2.5,
     gamma=1.0,
     subsample=0.8,
     objective = 'reg:linear',
     n_jobs=8,
     scale_pos_weight=1.8,
     random_state=27,
     base_score = 0.5)

    xgb.fit(x_train, y_train)

    filename = './data/finalized_model_movie.sav'
    pickle.dump(xgb, open(filename, 'wb'))


def viewcount(variable, input, Data):

    try:

        contents = pd.DataFrame(Data, columns = ['PAYMENT', 'PROGRAM_TYPE', 'New_Contents', 'genre_Label', 'target_age', 'playtime', 'channel_Label', 'contentnumber', 'episode_count', 'past_view'])
        x_test = contents[contents['New_Contents'] == 1]
        x_test = x_test[x_test['PROGRAM_TYPE'] == x_test.PROGRAM_TYPE.unique()[0]]
        x_test.contentnumber = x_test.contentnumber.fillna(0)
        x_test.episode_count = x_test.episode_count.fillna(1)
        x_test = x_test.drop('New_Contents', axis = 1)
        x_test = x_test.drop('PROGRAM_TYPE', axis = 1)
        x_test = x_test.values
        x_test = x_test.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_test = scaler.fit_transform(x_test)

        contents_label = pd.DataFrame(Data, columns = ['PROGRAM_TYPE', 'New_Contents', 'ViewCount'])
        y_test = contents_label[contents_label['New_Contents'] == 1]
        y_test.loc[:,'PROGRAM_TYPE'] = round(y_test.loc[:,'PROGRAM_TYPE'])
        y_test = y_test[y_test['PROGRAM_TYPE'] == y_test.PROGRAM_TYPE.unique()[0]]
        y_test = y_test.drop('PROGRAM_TYPE', axis = 1)
        y_test = y_test.drop('New_Contents', axis = 1)
        y_test = y_test.values

        filename = './data/finalized_model.sav'
        xgb = pickle.load(open(filename, 'rb'))
        xgb_preds = xgb.predict(x_test)

        for i in range(len(xgb_preds)):
            if xgb_preds[i] < 1:
                xgb_preds[i] = 1
            else:
                xgb_preds[i] = int(xgb_preds[i])

        for i in range(len(xgb_preds)):
            xgb_preds[i] = round(xgb_preds[i])

        x_test = contents[contents['New_Contents'] == 1]
        x_test = x_test[x_test['PROGRAM_TYPE'] != x_test.PROGRAM_TYPE.unique()[0]]
        x_test.contentnumber = x_test.contentnumber.fillna(0)
        x_test.episode_count = x_test.episode_count.fillna(1)
        x_test = x_test.drop('New_Contents', axis = 1)
        x_test = x_test.drop('PROGRAM_TYPE', axis = 1)
        x_test = x_test.values
        x_test = x_test.astype('float32')
        x_test = scaler.transform(x_test)

        y_test = contents_label[contents_label['New_Contents'] == 1]
        y_test.loc[:,'PROGRAM_TYPE'] = round(y_test.loc[:,'PROGRAM_TYPE'])
        y_test = y_test[y_test['PROGRAM_TYPE'] != y_test.PROGRAM_TYPE.unique()[0]]
        y_test = y_test.drop('PROGRAM_TYPE', axis = 1)
        y_test = y_test.drop('New_Contents', axis = 1)
        y_test = y_test.values

        filename = './data/finalized_model_movie.sav'
        xgb = pickle.load(open(filename, 'rb'))
        xgb_preds2 = xgb.predict(x_test)

        for i in range(len(xgb_preds2)):
            if xgb_preds2[i] < 1:
                xgb_preds2[i] = 1
            else:
                xgb_preds2[i] = int(xgb_preds2[i])

        for i in range(len(xgb_preds2)):
            xgb_preds2[i] = round(xgb_preds2[i])

        xgb_preds = np.concatenate((xgb_preds, xgb_preds2), axis=0)
        scaler = MinMaxScaler(feature_range=(0, 1))
        xgb_preds = xgb_preds.reshape(len(xgb_preds),1)
        xgb_preds = scaler.fit_transform(xgb_preds)

        select_score = Data[Data['New_Contents'] == 1]
        select_score.loc[:,'PROGRAM_TYPE'] = round(select_score.loc[:,'PROGRAM_TYPE'])
        select_score = select_score[select_score['PROGRAM_TYPE'] == select_score.PROGRAM_TYPE.unique()[0]]

        select_score2 = Data[Data['New_Contents'] == 1]
        select_score2.loc[:,'PROGRAM_TYPE'] = round(select_score2.loc[:,'PROGRAM_TYPE'])
        select_score2 = select_score2[select_score2['PROGRAM_TYPE'] != select_score2.PROGRAM_TYPE.unique()[0]]

        select_score = pd.concat([select_score, select_score2], axis = 0)

        select_score.index = range(len(select_score))
        number = select_score[select_score['EPISODE'] == input].index[0]

        answer = xgb_preds[number:number+1][0]
        answer_viewcount = round(answer[0], 15)

        answer_episode = select_score.EPISODE[number:number+1].values[0]

        return answer_episode, answer_viewcount

    except IndexError:

        message = print("Check the episode name")

        return message

def predict_viewcount(variable, Data):

    vod_model_save(variable, Data)
    movie_model_save(variable, Data)

    db = pymysql.connect(host='gcsdbinstance.cu0nuaw6yxna.us-east-1.rds.amazonaws.com', port=3306, user='gcs_g', passwd='awsg1020*', db='gcs_database', charset='utf8')

    cursor = db.cursor()

    sql = "SELECT * FROM gcs_database." + str(variable)

    cursor.execute(sql)
    temp = pd.read_sql(sql, db)

    temp = temp[temp['New_Contents'] == 1]

    list = []

    data = pd.read_sql(sql, db)
    for i in tqdm(temp.EPISODE):
        try:
            episode, count = viewcount(variable, i, data)
            count_10 = round(count, 10)
            dict = {"contentID": episode, "viewCount": count_10}
            list.append(dict)

        except TypeError:
            message = print("Check the episode name")

    result = {"items": list}

    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(MyEncoder, self).default(obj)

    result = json.dumps(result, cls=MyEncoder)

    return result
