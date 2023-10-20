import json
import pickle
import bz2file as bz2
from flask import Flask,request,app,jsonify,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data
regmodel = decompress_pickle('copper_reg_model.pbz2')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=np.array(list(data.values())).reshape(1,-1)
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Copper selling price is {}".format(output**2))



if __name__=="__main__":
    app.run(debug=True)