import pickle
from flask import Flask ,jsonify, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standart_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Build a DataFrame with the same column names that were used when the scaler was fitted.
        feature_names = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']
        input_df = pd.DataFrame([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]], columns=feature_names)

        new_data_scaled = standart_scaler.transform(input_df)
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', results=result[0])
    else:
        return render_template('home.html')

if __name__ == '__main__':
    application.run(host='0.0.0.0')