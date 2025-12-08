from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import pandas as pd

# Import the 7-day helper (create forecast_utils.py as shown previously)
from forecastutils import forecast_7days_iterative

app = Flask(__name__)
CORS(app)  # Fixes connection errors between React and Flask

# Load models safely
model_files = ['knn_reg.pkl', 'knn_clf.pkl', 'scaler.pkl', 'rain_le.pkl']
if not all(os.path.exists(f) for f in model_files):
    print("Error: Model files missing. Please run 'python model.py' first.")
    exit()

knn_reg = pickle.load(open('knn_reg.pkl', 'rb'))
knn_clf = pickle.load(open('knn_clf.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
rain_le = pickle.load(open('rain_le.pkl', 'rb'))

@app.route("/")
def home():
    return jsonify({"message": "Weather forecast API — /predict (1-day) and /predict_multi (7-day)"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Features must match the order in model.py
        features = [
            float(data["MinTemp"]),
            float(data["Humidity9am"]),
            float(data["Pressure9am"]),
            float(data["WindSpeed9am"]),
            float(data["Temp9am"])
        ]
        
        # Scale the input
        X_input = scaler.transform([features])
        
        # Predict
        temp_pred = knn_reg.predict(X_input)[0] # Predicted Max Temp
        rain_idx = knn_clf.predict(X_input)[0]  # Predicted Rain Index (0 or 1)
        rain_pred = rain_le.inverse_transform([rain_idx])[0] # Convert 0/1 back to No/Yes
        
        return jsonify({
            "predicted_max_temp": round(float(temp_pred), 2),
            "rain_tomorrow": rain_pred,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ---- New endpoint: 7-day iterative forecast ----
@app.route("/predict_multi", methods=["POST"])
def predict_multi():
    """
    Expects JSON like:
    {
      "MinTemp": 12.3,
      "Humidity9am": 65,
      "Pressure9am": 1016,
      "WindSpeed9am": 10,
      "Temp9am": 15.2
    }
    Returns JSON list of 7-day predictions (records).
    """
    try:
        data = request.get_json(force=True)
        start = [
            float(data["MinTemp"]),
            float(data["Humidity9am"]),
            float(data["Pressure9am"]),
            float(data["WindSpeed9am"]),
            float(data["Temp9am"])
        ]
    except Exception:
        return jsonify({"error": "Provide MinTemp, Humidity9am, Pressure9am, WindSpeed9am, Temp9am"}), 400

    try:
        df7 = forecast_7days_iterative(knn_reg, knn_clf, scaler, rain_le, start, days=7)
        # return JSON array of rows
        return df7.to_json(orient="records")
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
