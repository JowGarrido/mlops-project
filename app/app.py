from flask import Flask, request, jsonify
from google.cloud import aiplatform
import numpy as np
import pickle

app = Flask(__name__)

FEATURESTORE_NAME = "iris"
PROJECT = "garrido-ml-demos"
LOCATION = "us-central1"
FEATURESTORE_NAME = "iris"
ENTITY_TYPE = "id"
FEATURES = ["*"]

# Iniciando o Client da Vertex AI
aiplatform.init(project=PROJECT, location=LOCATION)

# Carregando o modelo em memória
_model = pickle.load(open("model.pkl", "rb"))

# Dicionário de indexes dos targets do modelo
targets = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

# Instanciando o client do Feature Store
_ids = aiplatform.featurestore.EntityType(
    entity_type_name=ENTITY_TYPE, featurestore_id=FEATURESTORE_NAME
)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"healthy"})

@app.route("/predict", methods=["POST"])
def predict(): 
    payload = request.json 
    
    _id = payload["instances"][0]

    # Leitura de features do Feature Store e inferência
    features = _ids.read(entity_ids=_id, feature_ids=["*"])

    # Inferência
    predictions = _model.predict([features.to_dict("split")["data"][0][1:]]).tolist()
    probas = _model.predict_proba([features.to_dict("split")["data"][0][1:]]).tolist()
    classes = _model.classes_.tolist()

    # Response
    response = { 
        "predictions":[
            {
                "prediction":predictions,
                "semantic":[targets[pred] for pred in predictions]
            }],
        "probabilities":[
            {
            "classes":classes, 
            "probas":probas
            }]
        }

    return jsonify(response)
    #return jsonify({"response":features_norm})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)