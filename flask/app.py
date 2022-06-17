# https://code.visualstudio.com/docs/python/tutorial-flask
# Set-ExecutionPolicy Unrestricted -Scope Process
# python -m flask run

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, GRU
import mysql.connector
import logging
import time

app = Flask(__name__)
def generate_model(data, year):
    lookback_window = 12
    # convert to x,y
    x = []
    y = []
    for i in range(lookback_window, len(data)):
        x.append(data[i - lookback_window:i])
        y.append(data[i])
    x = np.array(x)
    y = np.array(y)

    i = Input(shape=(lookback_window, 1))
    m = GRU(512, activation='relu')(i)
    m = Dense(64, activation='linear')(m)
    m = Dense(1, activation='linear')(m)

    model = Model(inputs=[i], outputs=[m])
    model.compile('adam','mae')
    model.fit(x, y, epochs=100, verbose=0)
    model.save('model/GRU_{}.h5'.format(year))

def make_prediction(model, data):
    lookback_window = 12
    # convert x y
    x = []
    y = []
    for i in range(lookback_window, len(data)):
        x.append(data[i - lookback_window:i])
        y.append(data[i])
    x = np.array(x)
    y = np.array(y)

    predict = model.predict(x)
    n = len(data) - 13

    new_x = np.concatenate((x[n:], predict[n:]), axis=None)
    new_x = np.array(new_x)
    new_x = np.delete(new_x, 0)
    new_x = np.array([new_x.astype(int)])
    
    # predict the future for how long
    future_months = 12
    new_arr = np.zeros(shape=(1, 12))
    new_arr = new_arr.astype(int)

    for loop in range(future_months):
        pred_2022 = model.predict(new_x)
        new_x = np.concatenate((new_x, pred_2022), axis=None)
        new_x = np.array(new_x)
        new_x = np.delete(new_x, 0)
        new_x = np.array([new_x.astype(int)])
        new_arr = np.append(new_arr, new_x, axis=0)
    new_arr = np.delete(new_arr, 0, 0)
    new_y = model.predict(new_arr)
    return new_y

@app.route("/", methods=['GET', 'POST'])

# upload
def home():
    # return "upload interface"
    if request.method == 'GET':
        return render_template('upload.html')
    elif request.method == 'POST':
        try:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            logging.basicConfig(filename="log/log_{}.log".format(timestr), level=logging.DEBUG)
            
            new_data = pd.read_excel(request.files.get('file'))
            month_list = ['january','february','march','april','may','june','july','august','september','october','november','december']
            jumlah_komoditas = len(new_data)
            
            db = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                port=3307,
                database='pricely'
            )
            cursor= db.cursor()

            # get lastest year
            cursor.execute('SELECT MAX(year) FROM prices;')
            year = 0
            for x in cursor:
                year = x[0]
            
            # UPDATE DB : replace pred to real
            logging.debug("tahap 1 : update pred to real\n")
            list_data = []
            for idx in range(len(new_data)):
                komoditas = new_data.iloc[0:, 5:].transpose()
                data = komoditas.iloc[0:, idx:idx+1]
                data = data.values
                list_data.append(data)
            for i, komoditas_real in enumerate(list_data):
                for j, data in enumerate(komoditas_real):  
                    logging.debug("{} {} {} {}\n".format(data[0], i, year, month_list[j]))
                    cursor.execute("UPDATE prices SET price = {} WHERE id_product = {} AND year = {} and month = '{}';".format(data[0], i+1, year, month_list[j]))

            # get data
            logging.debug("tahap 2 : get data form db\n")
            list_data = []
            for i in range(jumlah_komoditas):
                cursor.execute('SELECT price FROM prices WHERE id_product={} ORDER BY year, month;'.format(i+1))
                temp = []
                for x in cursor:
                    temp.append([x[0]])
                list_data.append(temp)

            # CREATE NEW MODEL H5
            logging.debug("tahap 3 : generate h5\n")
            generate_model(list_data[0], year)

            # PREDICTION
            logging.debug("tahap 4 : prediksi\n")
            saved_model = tf.keras.models.load_model('model/GRU_{}.h5'.format(year))
            list_pred = []
            for d in list_data:
                logging.debug("pred {}\n".format(d))
                pred = make_prediction(saved_model, d)
                list_pred.append(pred)

            # UPDATE DB
            logging.debug("tahap 5 : simpan prediksi ke db\n")
            for i, komoditas_pred in enumerate(list_pred):
                for j, monthly_pred in enumerate(komoditas_pred):
                    logging.debug("{} {} {} {}\n".format(i, year+1, month_list[j], monthly_pred[0]))
                    cursor.execute("INSERT INTO prices (id_product, year, month, price) VALUES ({}, {}, '{}', {});".format(i+1, year+1, month_list[j], monthly_pred[0]))
            db.commit()
            logging.info("SUCCESS : data successfully collected, trained, and save to db")
            return render_template('success.html')

        except Exception as e:
            logging.error("ERROR : "+str(e))
            return render_template('failed.html')
