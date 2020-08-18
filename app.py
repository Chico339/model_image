import os
import sys
import numpy as np
import predict_local
from flask import Flask, request, jsonify, render_template
import pickle
import mlflow
from mlflow import sklearn

port = 1234

app = Flask(__name__)

#model = pickle.load(open('model_final.pkl', 'rb'))

#   d8c9f88cbbab4111a9590408ae8bad6b first
#   0c5228ca02594d4f8b10e76dd751dfef  avec dataframe
#   7d5a9a88235c4666a97c688d6a29cdc8
########################################################
#run_id = "7d5a9a88235c4666a97c688d6a29cdc8"
#model_uri = "runs:/"+run_id+"/model"
#model= mlflow.sklearn.load_model(model_uri=model_uri)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    int_features = [int(x) for x in request.form.values()] #recup les val des featu

    final_features = [np.array(int_features)]

    prediction = int(predict_local.req(final_features))

    if prediction :
        return render_template('index.html', prediction_text='Survived')
    return render_template('index.html',prediction_text='Died')


if __name__ == "__main__":

    model="d419301627c7419f9ff85234e671ac48"
    if  sys.argv[1] == 'serve':
        if sys.argv.__len__() > 2:
            model=sys.argv[2]
        os.system("start /b mlflow models serve -m mlruns/0/"+model+"/artifacts/model/ -h 0.0.0.0 -p 1237")
        app.run(debug=False, port=5001)

    elif sys.argv[1] == 'ui':
        os.system("start /b mlflow ui")
        os.system("start /b mlflow run . --no-conda")
