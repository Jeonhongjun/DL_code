import pandas as pd
import numpy as np
import json
import argparse
import os
from tqdm import tqdm
import unittest
import pymysql
import data_reconstructor
import add_meta_feature
import hashing_engineering
import add_past_view
import Program_AvgCount

def db_con(x):
    FILE_DIR = "data"
    filename = "date_list.csv"
    filepath = os.path.join(FILE_DIR, filename)

    with open(filepath, "r") as target_file:
        date_list = [value.strip() for value in target_file.readlines()[1: ]]

    db = pymysql.connect(host='gcsdbinstance.cu0nuaw6yxna.us-east-1.rds.amazonaws.com', port=3306, user='gcs_g', passwd='awsg1020*', db='CJ', charset='utf8', max_allowed_packet = 64777216)
    connection = db.cursor()

    for i in tqdm(date_list[x:x+1]):

        sql = "SELECT * FROM CJ.VOD" + str(i)[:6] + " limit 10"

        Data = pd.read_sql(sql, db)

    return Data

def data_check(date):

    db = pymysql.connect(host='gcsdbinstance.cu0nuaw6yxna.us-east-1.rds.amazonaws.com', port=3306, user='gcs_g', passwd='awsg1020*', db='gcs_database', charset='utf8')

    cursor = db.cursor()

    sql = "SELECT * FROM gcs_database." + str(date)

    cursor.execute(sql)
    Data = pd.read_sql(sql, db)

    return Data
