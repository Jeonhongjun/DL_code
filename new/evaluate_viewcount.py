import unittest
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score
import pymysql
import random
import os

def random_suffule(x):

    num_to_select = x
    date_list = pd.read_csv('./data/date_list.csv', sep='\t', header = 0)
    list_of_random_items = random.sample(list(date_list['date']), num_to_select)

    return list_of_random_items

def data_loader(x):

    db = pymysql.connect(host='gcsdbinstance.cu0nuaw6yxna.us-east-1.rds.amazonaws.com', port=3306, user='gcs_g', passwd='awsg1020*', db='gcs_database', charset='utf8')
    cursor = db.cursor()
    sql = "SELECT * FROM gcs_database." + str(i)
    cursor.execute(sql)
    Data = pd.read_sql(sql, db)

    return Data

def predict_result(list_of_random_items):

    def xgboost_hot_cold(date, Data):

        pd.options.mode.chained_assignment = None  # default='warn'

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

        x_test = contents[contents['New_Contents'] == 1]
        x_test = x_test[x_test['PROGRAM_TYPE'] == x_test.PROGRAM_TYPE.unique()[0]]
        x_test.contentnumber = x_test.contentnumber.fillna(0)
        x_test.episode_count = x_test.episode_count.fillna(1)
        x_test = x_test.drop('New_Contents', axis = 1)
        x_test = x_test.drop('PROGRAM_TYPE', axis = 1)
        x_test = x_test.values
        x_test = x_test.astype('float32')
        x_test = scaler.transform(x_test)

        contents = pd.DataFrame(Data, columns = ['PROGRAM_TYPE', 'New_Contents', 'ViewCount'])
        y_train = contents[contents['New_Contents'] == 0]
        y_train = y_train.drop('New_Contents', axis = 1)
        y_train.loc[:,'PROGRAM_TYPE'] = round(y_train.loc[:,'PROGRAM_TYPE'])
        y_train = y_train[y_train['PROGRAM_TYPE'] == y_train.PROGRAM_TYPE.unique()[0]]
        y_train = y_train.drop('PROGRAM_TYPE', axis = 1)
        y_train = y_train.values
        y_train = y_train.astype('float32')

        y_test = contents[contents['New_Contents'] == 1]
        y_test = y_test[y_test['PROGRAM_TYPE'] == y_test.PROGRAM_TYPE.unique()[0]]
        y_test = y_test.drop('PROGRAM_TYPE', axis = 1)
        y_test = y_test.drop('New_Contents', axis = 1)
        y_test = y_test.values

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
        xgb_preds = xgb.predict(x_test)

        for i in range(xgb_preds.shape[0]):
            if xgb_preds[i] < 1:
                xgb_preds[i] = 1
            else:
                xgb_preds[i] = int(xgb_preds[i])

        for i in range(xgb_preds.shape[0]):
            xgb_preds[i] = round(xgb_preds[i])

        contents = pd.DataFrame(Data, columns = ['EPISODE', 'PAYMENT', 'PROGRAM_TYPE', 'ViewCount', 'New_Contents', 'genre_Label', 'target_age', 'playtime', 'channel_Label', 'contentnumber', 'episode_count', 'past_view'])
        x_test = contents[contents['New_Contents'] == 1]
        x_test = x_test.drop('ViewCount', axis = 1)
        x_test = x_test[x_test['PROGRAM_TYPE'] == x_test.PROGRAM_TYPE.unique()[0]]

        xgb_preds = pd.DataFrame(xgb_preds, columns = ['ViewCount'])

        xgb_preds.index = x_test.index

        H_C = pd.concat([x_test, xgb_preds], axis = 1)

        old = contents[contents['New_Contents'] == 0]
        old = old[old['PROGRAM_TYPE'] == old.PROGRAM_TYPE.unique()[0]]

        H_C = H_C[['EPISODE', 'PAYMENT', 'PROGRAM_TYPE', 'ViewCount', 'New_Contents', 'genre_Label', 'target_age', 'playtime', 'channel_Label', 'contentnumber', 'episode_count', 'past_view']]

        yhat = pd.concat([old, H_C], axis = 0)

        yhat = yhat.sort_values(by=['ViewCount'], ascending=False)

        values = yhat.values

        new_index = yhat[yhat['New_Contents'] == 1].index
        Hot_index = round(yhat.shape[0]/5)
        yhat.index = range(0, len(yhat))

        HC = []
        for i in range(yhat.shape[0]):
            if yhat.index[i] < (Hot_index + 1):
                HC.append('HOT')
            else:
                HC.append('COLD')

        HC = pd.DataFrame(HC, columns = ['H&C'])
        yhat = pd.concat([yhat, HC], axis =1)

        yhat = yhat[yhat['New_Contents'] == 1]
        yhat.index = new_index
        yhat = yhat.sort_index(ascending=True)

        yhat = yhat['H&C'].values

        actual =  pd.DataFrame(Data, columns = ['PAYMENT', 'PROGRAM_TYPE', 'New_Contents', 'H&C', 'genre_Label', 'target_age', 'playtime', 'channel_Label', 'contentnumber', 'past_view'])
        actual = actual[actual['New_Contents'] == 1]
        actual = actual[actual['PROGRAM_TYPE'] == actual.PROGRAM_TYPE.unique()[0]]
        actual = actual['H&C'].values

        from sklearn.metrics import recall_score, precision_score, f1_score
        recall_scores = recall_score(actual, yhat, average='macro', labels = ['HOT'])
        recall_score_cold = recall_score(actual, yhat, average='macro', labels = ['COLD'])
        precision_scores = precision_score(actual, yhat, average='macro', labels = ['HOT'])
        precision_score_cold = precision_score(actual, yhat, average='macro', labels=['COLD'])
        f_score = f1_score(actual, yhat, average='macro', labels = ['HOT'])
        f_score_cold = f1_score(actual, yhat, average='macro', labels=['COLD'])

        return f_score, f_score_cold, recall_scores, precision_scores, recall_score_cold, precision_score_cold

    precision_cold = []
    precision = []
    recall_cold = []
    recall = []
    f1_score = []
    f1_score_colds = []

    for i in tqdm(list_of_random_items):

        db = pymysql.connect(host='gcsdbinstance.cu0nuaw6yxna.us-east-1.rds.amazonaws.com', port=3306, user='gcs_g', passwd='awsg1020*', db='gcs_database', charset='utf8')
        cursor = db.cursor()
        sql = "SELECT * FROM gcs_database." + str(i)
        cursor.execute(sql)
        Data = pd.read_sql(sql, db)
        print('Date: ', i)

        f_score, f_score_cold, recall_scores, precision_scores, recall_score_cold, precision_score_cold = xgboost_hot_cold(i, Data)
        print('Precision-Score: ', precision_scores)
        print('Recall-Score: ', recall_scores)
        print('F1-Score: ', f_score)

        precision_cold.append(precision_score_cold)
        precision.append(precision_scores)
        recall_cold.append(recall_score_cold)
        recall.append(recall_scores)
        f1_score.append(f_score)
        f1_score_colds.append(f_score_cold)

    print('View pattern classification Precision - New_HOT : ' + str(np.mean(precision)))
    print('View pattern classification Recall - New_HOT : ' + str(np.mean(recall)))
    print('View pattern classification F1-Score - New_HOT : ' + str(np.mean(f1_score)))
    print('View pattern classification Precision - New_COLD : ' + str(np.mean(precision_cold)))
    print('View pattern classification Recall - New_COLD : ' + str(np.mean(recall_cold)))
    print('View pattern classification F1-Score - New_COLD : ' + str(np.mean(f1_score_colds)))
    return 'F1-score : ' + str(np.mean(f1_score)), np.mean(f1_score)
