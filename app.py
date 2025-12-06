from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app) # Fixes connection errors between React and Flask

# Load models safely
model_files = ['knn_reg.pkl', 'knn_clf.pkl', 'scaler.pkl', 'rain_le.pkl']
if not all(os.path.exists(f) for f in model_files):
    print("Error: Model files missing. Please run 'python model.py' first.")
    exit()

knn_reg = pickle.load(open('knn_reg.pkl', 'rb'))
knn_clf = pickle.load(open('knn_clf.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
rain_le = pickle.load(open('rain_le.pkl', 'rb'))

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
            "predicted_max_temp": round(temp_pred, 2),
            "rain_tomorrow": rain_pred,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)