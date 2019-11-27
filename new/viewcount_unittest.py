# tests.py
import os
import unittest
import evaluate_viewcount
import viewcount_predictor
import json
from flask import jsonify
import warnings
import time
import timeout_decorator
import pymysql
import pandas as pd
import db_connection
import numpy

class connection_Test(unittest.TestCase):

    def test_db_con_check(self):
        Data = db_connection.db_con(0)
        warnings.simplefilter("ignore")
        self.assertEqual(Data.columns[0], 'date_time')
        self.assertEqual(Data.columns[1], 'USER')
        self.assertEqual(Data.columns[2], 'LOG_TYPE')
        self.assertEqual(Data.columns[3], 'AUTH_CL')
        self.assertEqual(Data.columns[4], 'PROGRAM')
        self.assertEqual(Data.columns[5], 'VOD_TYPE')
        self.assertEqual(Data.columns[6], 'EPISODE')

class preprocessing_Test(unittest.TestCase):

    def test_preprocessing_table_check(self):
        random = evaluate_viewcount.random_suffule(1)
        Data = db_connection.data_check(random[0])
        self.assertEqual(Data.columns[1], 'EPISODE')
        self.assertEqual(type(Data.EPISODE[0]), str)
        self.assertEqual(Data.columns[2], 'month')
        self.assertEqual(type(Data.month[0]), numpy.int64)
        self.assertEqual(Data.columns[3], 'day')
        self.assertEqual(type(Data.day[0]), numpy.int64)
        self.assertEqual(Data.columns[4], 'time')
        self.assertEqual(type(Data.time[0]), numpy.int64)
        self.assertEqual(Data.columns[5], 'Program')
        self.assertEqual(type(Data.Program[0]), str)
        self.assertEqual(Data.columns[6], 'PAYMENT')
        self.assertEqual(type(Data.PAYMENT[0]), numpy.float64)
        self.assertEqual(Data.columns[8], 'ViewCount')
        self.assertEqual(type(Data.ViewCount[0]), numpy.int64)

class response_Test(unittest.TestCase):

    def test_predict_result(self):
        date = evaluate_viewcount.random_suffule(1)
        warnings.simplefilter("ignore")
        response = viewcount_predictor.viewcount_test(date[0])
        self.assertIsInstance(response, dict)

class viewcount_Test(unittest.TestCase):

    def test_random_sample(self):
        random = evaluate_viewcount.random_suffule(30)
        self.assertEqual(len(random), 30)

    def test_model(self):
        date = evaluate_viewcount.random_suffule(30)
        model, performance = evaluate_viewcount.predict_result(date)
        self.assertEqual(model[:11], 'F1-score : ')
        self.assertTrue(performance < 1)
        self.assertTrue(performance > 0)

class loader_Test(unittest.TestCase):

    def test_result_table_check(self):
        date = evaluate_viewcount.random_suffule(1)
        warnings.simplefilter("ignore")
        result = viewcount_predictor.database_test(date[0])
        db = pymysql.connect(host='gcsdbinstance.cu0nuaw6yxna.us-east-1.rds.amazonaws.com', port=3306, user='gcs_g', passwd='awsg1020*', db='prediction_result', charset='utf8')
        cursor = db.cursor()
        sql = "SELECT * FROM prediction_result." + str(date[0]) + "_result"
        cursor.execute(sql)
        Data = pd.read_sql(sql, db)
        self.assertEqual(Data.columns[0], 'contentID')
        self.assertEqual(Data.columns[1], 'viewCount')

class timeoutTest(unittest.TestCase):

    @timeout_decorator.timeout(20)
    def testtimeout(self):
        date = evaluate_viewcount.random_suffule(1)
        warnings.simplefilter("ignore")
        response = viewcount_predictor.viewcount_test(date[0])
