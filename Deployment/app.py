import flask
from flask import request
import numpy as np
import pickle
import pandas as pd

scaler = pickle.load(open('model/scaler4.pkl', 'rb'))
model = pickle.load(open('model/modelfp4.pkl', 'rb'))

app = flask.Flask(__name__, template_folder='templates')

@app.route('/')
def main():
    return(flask.render_template('main.html'))
if __name__ == '__main__':
    app.run(debug=True)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    depth = float(request.form['depth'])
    mag = float(request.form['mag'])
    predict_list = [[depth, mag]]
    predict = scaler.transform(predict_list)
    feat_cols = ['depth','mag']
    predict = pd.DataFrame(predict,columns=feat_cols)
    prediction = model.predict(predict)
    output = {0: 'Klaster 0', 1: 'Klaster 1',2:'Klaster 2'}
    return flask.render_template('main.html', prediction_text='Gempa termasuk dalam {}'.format(output[prediction[0]]))