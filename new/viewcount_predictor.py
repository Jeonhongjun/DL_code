from flask_restful import Resource
from build_model import vod_model_save
from build_model import movie_model_save
from build_model import viewcount
from build_model import predict_viewcount
import json
import pymysql
import pandas as pd
from flask import jsonify

class viewcount_predict(Resource):

    def get(self, variable):

        db = pymysql.connect(host='gcsdbinstance.cu0nuaw6yxna.us-east-1.rds.amazonaws.com', port=3306, user='gcs_g', passwd='awsg1020*', db='gcs_database', charset='utf8')

        cursor = db.cursor()

        sql = "SELECT * FROM gcs_database." + str(variable)

        try:
            cursor.execute(sql)
            Data = pd.read_sql(sql, db)
            result = predict_viewcount(variable, Data)
            result = json.loads(result)

            con = pymysql.connect(host='gcsdbinstance.cu0nuaw6yxna.us-east-1.rds.amazonaws.com', port=3306, user='gcs_g', passwd='awsg1020*', db='prediction_result', charset='utf8')
            cursor = con.cursor()

            def validate_string(val):
                if val != None:
                    if type(val) is int:
                        return str(val).encode('utf-8')
                    else:
                        return val

            sql ="CREATE TABLE %s_result (contentID VARCHAR(255), viewCount VARCHAR(255))"
            try:
                cursor.execute(sql, int(variable))

                for i, item in enumerate(result['items']):
                    contentID = validate_string(item.get("contentID", None))
                    viewCount = validate_string(item.get("viewCount", None))

                    sql = "insert into %s_result (contentID, viewCount) values (%s, %s)"
                    cursor.execute(sql, (int(variable), contentID, viewCount))
                con.commit()
                con.close()
            except pymysql.err.InternalError:
                print('Data already exists')
            response = jsonify(result)

        except pymysql.err.ProgrammingError:
            result = {"items": None}
            response = jsonify(result)

        return response

def viewcount_test(variable):

    db = pymysql.connect(host='gcsdbinstance.cu0nuaw6yxna.us-east-1.rds.amazonaws.com', port=3306, user='gcs_g', passwd='awsg1020*', db='gcs_database', charset='utf8')

    cursor = db.cursor()

    sql = "SELECT * FROM gcs_database." + str(variable)

    cursor.execute(sql)
    Data = pd.read_sql(sql, db)
    result = predict_viewcount(variable, Data)

    return result

def database_test(variable):

    db = pymysql.connect(host='gcsdbinstance.cu0nuaw6yxna.us-east-1.rds.amazonaws.com', port=3306, user='gcs_g', passwd='awsg1020*', db='gcs_database', charset='utf8')

    cursor = db.cursor()

    sql = "SELECT * FROM gcs_database." + str(variable)

    try:
        cursor.execute(sql)
        Data = pd.read_sql(sql, db)
        result = predict_viewcount(variable, Data)
        result = json.loads(result)

        con = pymysql.connect(host='gcsdbinstance.cu0nuaw6yxna.us-east-1.rds.amazonaws.com', port=3306, user='gcs_g', passwd='awsg1020*', db='prediction_result', charset='utf8')
        cursor = con.cursor()

        def validate_string(val):
            if val != None:
                if type(val) is int:
                    return str(val).encode('utf-8')
                else:
                    return val

        sql ="CREATE TABLE %s_result (contentID VARCHAR(255), viewCount VARCHAR(255))"
        cursor.execute(sql, int(variable))

        for i, item in enumerate(result['items']):
            contentID = validate_string(item.get("contentID", None))
            viewCount = validate_string(item.get("viewCount", None))

            sql = "insert into %s_result (contentID, viewCount) values (%s, %s)"
            cursor.execute(sql, (int(variable), contentID, viewCount))
        con.commit()
        con.close()

    except pymysql.err.ProgrammingError:
        result = {"items": None}

    return result
