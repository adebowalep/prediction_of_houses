from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np
from config import PORT, DEBUG_MODE
app = Flask(__name__)

#model = load_model('deployment_28042020')
#cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

# load the model from disk
pickle.load(open('finalized_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template("home.html")
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE)
