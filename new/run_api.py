from flask import Flask
from flask_restful import Resource, Api
from viewcount_predictor import viewcount_predict
from flask import Flask, jsonify
import os


app = Flask(__name__)
api = Api(app)

api.add_resource(viewcount_predict, '/contents/new/<variable>')

if __name__ == '__main__':
    #context = ('./key/cert.pem', './key/key.pem')   #certificate and key files
    app.run(host=os.environ["GCS_IP"], port=5000,
            #ssl_context=context
            )
