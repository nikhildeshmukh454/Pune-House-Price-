from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from main import gd

model = pickle.load(open('lr_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    size = float(request.form['size'])
    bath = float(request.form['bath'])
    balcony = float(request.form['balcony'])
    sqft = float(request.form['sqft'])
    area_type = request.form['area_type']

    input_data = pd.DataFrame([[area_type, size, sqft, bath, balcony, location]], 
                              columns=['area_type', 'size', 'total_sqft', 'bath', 'balcony', 'location'])

    result = model.predict(input_data)
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

