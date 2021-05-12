from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn
import pandas as pd
import xgboost as xg

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
model2 = pickle.load(open('diamonds.pkl', 'rb'))
model3 = pickle.load(open('bcancer.pkl', 'rb'))

@app.route('/')
@app.route('/Home')
def hello_world():
   return render_template('index.html')


@app.route('/Home Loan')
def home_loan():
    return render_template('home-loan.html')


@app.route('/breast_castyle=ncer')
def cancer():
    return render_template('breastcancer.html')


@app.route('/Covid 19')
def Covid():
    return render_template('covid.html')


@app.route('/Covidmap')
def Covidmap():
    return render_template('indcvd.html')



@app.route('/diamond')
def diamonds():
    return render_template('diamonds.html')


@app.route('/about')
def about_us():
    return render_template('about.html')



@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction = model.predict(final)

    if prediction == 0:
        return render_template('homeloanresult.html')
    else:
        return render_template('homeloanneg.html')

@app.route('/predict3', methods=['POST', 'GET'])
def predict3():
    int_features3 = [float(x) for x in request.form.values()]
    final3 = [np.array(int_features3)]
    print(int_features3)
    print(final3)
    prediction3 = model3.predict(final3)

    if prediction3 == 0:
        return render_template('breastcancer.html', pred='The cancer is malignant')
    else:
        return render_template('breastcancer.html', pred='The cancer is benign')


@app.route('/predict1',methods=['POST' , 'GET'])
def predict1():
    int_features = [(x) for x in request.form.values()]
    headers = ['carat', 'cut', 'color', 'clarity', 'depth', 'table']
    input_variables = pd.DataFrame([int_features],
                                columns=headers, 
                                dtype=float,
                                index=['input'])
    # Get the model's prediction
    prediction = model2.predict(input_variables)
    output = prediction


    return render_template('diamonds.html', prediction_text='Diamond price is $ {}'.format(output))





if __name__ == "__main__":
    #app.run(host='0.0.0.0', port=8080)
    app.run(debug=True)
