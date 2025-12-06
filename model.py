import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score

# 1. Load Data
print("Loading dataset...")
try:
    df = pd.read_csv("weatherAUS.csv")
except FileNotFoundError:
    print("Error: 'weatherAUS.csv' not found. Please put the CSV file in this folder.")
    exit()

# 2. Select Relevant Columns
# We use 9am data to predict the Max Temp and Tomorrow's Rain
cols_to_keep = [
    'MinTemp', 'Humidity9am', 'Pressure9am', 'WindSpeed9am', 'Temp9am', # Inputs
    'MaxTemp', 'RainTomorrow' # Targets
]
df = df[cols_to_keep].dropna() # Drop rows with missing values

# 3. Define Features (X) and Targets (y)
feature_cols = ['MinTemp', 'Humidity9am', 'Pressure9am', 'WindSpeed9am', 'Temp9am']
X = df[feature_cols]

# Target 1: Max Temperature (Regression)
y_temp = df['MaxTemp']

# Target 2: Rain Tomorrow (Classification)
le_rain = LabelEncoder()
y_rain = le_rain.fit_transform(df['RainTomorrow']) # Converts 'Yes'/'No' to 1/0

# 4. Split Data
X_train, X_test, y_temp_train, y_temp_test, y_rain_train, y_rain_test = train_test_split(
    X, y_temp, y_rain, test_size=0.2, random_state=42
)

# 5. Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Models
print("Training models...")
# KNN Regression for Temperature
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_scaled, y_temp_train)

# KNN Classification for Rain
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_scaled, y_rain_train)

# 7. Evaluate
print("\n--- Evaluation ---")
temp_preds = knn_reg.predict(X_test_scaled)
print(f"Temperature RMSE: {np.sqrt(mean_squared_error(y_temp_test, temp_preds)):.2f} °C")

rain_preds = knn_clf.predict(X_test_scaled)
print(f"Rain Prediction Accuracy: {accuracy_score(y_rain_test, rain_preds) * 100:.2f}%")

# 8. Save Files
print("\nSaving model files...")
pickle.dump(knn_reg, open('knn_reg.pkl', 'wb'))
pickle.dump(knn_clf, open('knn_clf.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(le_rain, open('rain_le.pkl', 'wb'))
print("Success! Models saved.")