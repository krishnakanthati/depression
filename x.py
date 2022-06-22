import json
import flask
from flask_cors import CORS, cross_origin
import pickle
import numpy as np
import pandas as pd
from pymongo import MongoClient
from bson import json_util
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk import capture_exception
import urllib.parse
from flask import url_for, redirect, render_template
username = urllib.parse.quote_plus('kris')
password = urllib.parse.quote_plus('@Krishna8')
# req_data = flask.request.get_json()
req_data = {
    "interferes_with_work": [
        "Always"
    ],
    "prev_mental_disorder": [
        "Yes"
    ],
    "if_it_interferes": [
        "Rarely"
    ],
    "Sought_Treatment": [
        "0"
    ],
    "are_you_willing_to_share_with_friends": [
        "Yes"
    ],
    "family_history_of_mental_health": [
        "Yes"
    ],
    "emphasis_on_physical_health": [
        "0"
    ],
    "medical_leave_for_depression": [
        "Somewhat difficult"
    ],
    "Number_of_employees_in_org": [
        "More than 1000"
    ],
    "well_handled": [
        "Yes, I observed"
    ],
    "comfortable_direct_sup_previous": [
        "No, none of my previous supervisors"
    ],
    "aware_of_importance": [
        "No, I only became aware later"
    ],
    "previous_help": [
        "No, none did"
    ],
    "mental_health_allowance": [
        "No"
    ],
    "employer_in_interview": [
        "No"
    ]
}
test = pd.DataFrame(req_data)
record = test.to_dict()

record = {k: v[0] for k, v in record.items()}
print(record)
f = open('targetencodemodel.sav', 'rb')
t = pickle.load(f)
print("t.............................", t)
print("f.............................", f)
f.close()
print("Model Loaded")
print(t)
print('\n')
print('........................')
test = t.transform(test)
print('........................')
print('done')
file = open('randomforest.sav', 'rb')
model = pickle.load(file)
level = model.predict(test)
record['predicted_level'] = str(level)
client = MongoClient(
    'mongodb+srv://%s:%s@cluster0.0vg1ud3.mongodb.net/' % (username, password))
db = client.fyproj
collection = db.ques
print("connection_created")
collection.insert_one(record)
client.close()
print("Inserted")
print(level)
