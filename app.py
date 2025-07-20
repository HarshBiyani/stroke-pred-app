from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder="build", static_url_path="")
CORS(app)

# Load models
models = {
    "Logistic Regression": pickle.load(open("logistic_model.pkl", "rb")),
    "K-Nearest Neighbors": pickle.load(open("knn_model.pkl", "rb")),
    "Decision Tree": pickle.load(open("dt_model.pkl", "rb")),
    "Gaussian Naive Bayes": pickle.load(open("gnb_model.pkl", "rb")),
    "Bernoulli Naive Bayes": pickle.load(open("bnb_model.pkl", "rb")),
    "Support Vector Machine": pickle.load(open("svm_model.pkl", "rb")),
}

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([list(data["features"].values())]).reshape(1, -1)
    model_name = data.get("model")

    if model_name not in models:
        return jsonify({"error": "Invalid model selected"}), 400

    model = models[model_name]
    prediction = model.predict(features)[0]
    prob = getattr(model, "predict_proba", None)

    if callable(prob):
        probability = model.predict_proba(features)[0][1]
    else:
        probability = None

    return jsonify({
        "prediction": int(prediction),
        "probability": probability
    })

if __name__ == "__main__":
    app.run(debug=True)
