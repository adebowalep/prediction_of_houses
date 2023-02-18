from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np
from config import PORT, DEBUG_MODE
app = Flask(__name__)

# Load the model from the file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

cols = [ 'bedrooms', 'bathrooms', 
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement',  'lat', 
       'sqft_living15']    

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = model.predict(data_unseen)
    prediction = int(prediction[0])
    return render_template('home.html',pred='Expected Bill will be {}'.format(prediction))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE)
