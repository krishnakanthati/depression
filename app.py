import flask
from flask_cors import CORS, cross_origin
import pickle
import numpy as np
import pandas as pd
from pymongo import MongoClient
import urllib.parse
from flask import render_template, request
from textblob import TextBlob
username = urllib.parse.quote_plus('kris')
password = urllib.parse.quote_plus('@Krishna8')


# sentry_sdk.init(
#     dsn="https://89a7204bf6eb4860843afe8b5188ecaf@o372533.ingest.sentry.io/5196045",
#     integrations=[FlaskIntegration()]
# )

app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/screen2', methods=['POST'])
@cross_origin()
def screen2():
    try:
        req_data = flask.request.get_json()
        print(req_data)
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
        return str(level)
    except Exception as e:
        capture_exception(e)
        flask.abort(400)


@app.route('/data', methods=['GET'])
@cross_origin()
def get_data():
    try:
        client = MongoClient(
            'mongodb+srv://%s:%s@cluster0.0vg1ud3.mongodb.net/' % (username, password))
        db = client.fyproj
        collection = db.ques
        print("connection_created")
        cursor = collection.find()
        df = pd.DataFrame(list(cursor))
        df.to_csv('downloaded.csv')
        print("Saved to downloaded.csv")
        client.close()
        return "Saved"
    except Exception as e:
        capture_exception(e)
        flask.abort(400)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/quiz')
def quiz():
    return render_template('quiz.html')


@app.route('/text')
def text():
    return render_template('text.html')


@app.route('/text', methods=['POST'])
def analyse():
    text = request.form['analyse']
    res = TextBlob(text)

    if res.sentiment.polarity < 0:
        result = "Negative"
    elif res.sentiment.polarity == 0:
        result = "Neutral"
    else:
        result = "Positive"

    return render_template("text.html", result=result + " ( " + str(res.sentiment.polarity) + " ) ")


@app.route('/r2')
def r2():
    return render_template('r2.html')


@app.route('/r1')
def r1():
    return render_template('r1.html')


@app.route('/r0')
def r0():
    return render_template('r0.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/dashboard')
def dashboard():
    return "Hello"


if __name__ == "__main__":
    app.run(debug=True)
