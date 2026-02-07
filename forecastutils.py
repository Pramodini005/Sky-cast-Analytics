# forecast_utils.py
import numpy as np
import pandas as pd

def forecast_7days_iterative(knn_reg, knn_clf, scaler, rain_le, start_features, days=7):
    """
    start_features: [MinTemp, Humidity9am, Pressure9am, WindSpeed9am, Temp9am]
    Returns a pandas.DataFrame with 7 rows for days 1..days.
    """
    results = []
    cur = list(start_features)

    for i in range(days):
        # scale and predict
        Xs = scaler.transform([cur])
        pred_max = float(knn_reg.predict(Xs)[0])
        rain_idx = int(knn_clf.predict(Xs)[0])
        # inverse_transform may require 1D array-like
        try:
            rain_pred = rain_le.inverse_transform([rain_idx])[0]
        except Exception:
            # fallback: if rain_le is label encoder returning strings
            rain_pred = str(rain_idx)

        day_row = {
            "day": i+1,
            "MinTemp": round(cur[0], 2),
            "Humidity9am": round(cur[1], 2),
            "Pressure9am": round(cur[2], 2),
            "WindSpeed9am": round(cur[3], 2),
            "Temp9am": round(cur[4], 2),
            "PredMaxTemp": round(pred_max, 2),
            "RainTomorrow": rain_pred
        }
        results.append(day_row)

        # Build next day's inputs (simple heuristic)
        next_temp9am = 0.6 * cur[4] + 0.4 * pred_max
        next_mintemp = pred_max - 4.0
        next_pressure = cur[2]
        next_humidity = cur[1]
        next_windspeed = cur[3]

        cur = [next_mintemp, next_humidity, next_pressure, next_windspeed, next_temp9am]

    return pd.DataFrame(results)
