import pandas as pd
import numpy as np
import json
import argparse
import os
from tqdm import tqdm
import time
import unittest
import pymysql
import data_reconstructor
import add_meta_feature
import hashing_engineering
import add_past_view
import Program_AvgCount

def test(x, cursor):

    old_list = data_reconstructor.old_contents_search(x, cursor)
    new = data_reconstructor.new_contents_search(x, cursor)
    new_data = data_reconstructor.new_data_gen(x, cursor)
    data_reconstructor.new_contents_tagging(dates = x, old_list = old_list, new_list = new, new_data = new_data)
    hashing_engineering.hashing(x)
    hashing_engineering.hashing_pro(x)
    add_meta_feature.meta_genre(x)
    add_meta_feature.meta_age(x)
    add_meta_feature.meta_playtime(x)
    add_meta_feature.meta_channel(x)
    add_meta_feature.meta_contentnumber(x)
    data_reconstructor.episode(x)

    print('success '+ str(x) +'.csv')

if __name__ == "__main__":
    FILE_DIR = "data"
    filename = "date_list.csv"
    filepath = os.path.join(FILE_DIR, filename)

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", help="Select data filename", default=filepath)
    args = parser.parse_args()

    with open(filepath, "r") as target_file:
        date_list = [value.strip() for value in target_file.readlines()[1: ]]

    db = pymysql.connect(host='gcsdbinstance.cu0nuaw6yxna.us-east-1.rds.amazonaws.com', port=3306, user='gcs_g', passwd='awsg1020*', db='CJ', charset='utf8', max_allowed_packet = 64777216)
    connection = db.cursor()

    for i in tqdm(date_list):
        sql = "SELECT * FROM CJ.VOD" + str(i)[:6] + " limit 10"
        start_time = time.time()

        #connection.execute(sql)
        Data = pd.read_sql(sql, db)

        old_list = []
        j = i
        if str(j)[6:8] != '01':
            temp = Data[Data.index < int(j)]
            old = list(temp.EPISODE.unique())
            old_list.append(old)
        else:
            pass

        while str(j)[4:6] != '01':
            j = int(j) - 100
            sql = "SELECT * FROM CJ.VOD" + str(j)[:6]
            connection.execute(sql)
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

        print('old_list complete')
        new_list = Program_AvgCount.new_contents(i, Data)


        new_data = Program_AvgCount.new_data_gen(i, Data)
        Program_AvgCount.new_contents_tagging(i, old_list, new_list, new_data)

    for date_element in tqdm(date_list[1:]):
        test(date_element, cursor)

    date_list = pd.read_csv('./data/date_list.csv', sep='\t', header=0)
    add_past_view.past_view()

    filename = ""
    pass
