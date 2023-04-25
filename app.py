import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# create app
app = Flask(__name__)
# load model
mlp_model = pickle.load(open('model/mlp_sf.pkl', 'rb'))
scaler = pickle.load(open('model/scaler_sf.pkl', 'rb'))

# create route that renders index.html template
@app.route("/")
def home():
    return render_template("index.html")

# create predict route that takes in data and returns prediction
@app.route("/predict_api", methods=['POST'])
def predict_api():
    # get data
    data=request.get_json()['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    # transform data
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    # predict
    output = mlp_model.predict(new_data)[0]
    # send back to browser
    output = output.tolist()
    return jsonify(output)

@app.route("/predict", methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = mlp_model.predict(final_input)[0]
    if output == 0:
        output = 'Normal'
    else:
        output = 'an Attack/Intrusion'
    return render_template('index.html', prediction_text='The connection is {}'.format(output))

# run app
if __name__ == "__main__":
    app.run(debug=True)