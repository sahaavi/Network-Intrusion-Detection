import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# create app
app = Flask(__name__)
# load model
mlp_model = pickle.load(open('model/mlp.pkl', 'rb'))
scalar = pickle.load(open('model/scalar.pkl', 'rb'))

# create route that renders index.html template
@app.route("/")
def home():
    return render_template("index.html")

# create predict route that takes in data and returns prediction
@app.route("/predict_api", methods=['POST'])
def predict_api():
    # get data
    data=request.get_json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    # transform data
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    # predict
    output = mlp_model.predict(new_data)
    # send back to browser
    output = output[0]
    return jsonify(output[0])

# run app
if __name__ == "__main__":
    app.run(debug=True)
