import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
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
print("Generating EDA Graphs...")

# Graph 1: Target Balance (Rain vs No Rain)
plt.figure(figsize=(6, 4))
sns.countplot(x='RainTomorrow',hue='RainTomorrow' ,data=df, palette='coolwarm',legend=False)
plt.title('Distribution of Rain Tomorrow')
plt.show(block=False)
plt.pause(2)   # display for 2 seconds
plt.close()


# Graph 2: Correlation Heatmap
plt.figure(figsize=(10, 8))
# Select numeric columns only for correlation
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show(block=False)
plt.pause(2)   # display for 2 seconds
plt.close()


# Graph 3: Scatter Plot (Humidity vs Temp)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Humidity9am', y='MaxTemp', hue='RainTomorrow', data=df, palette='viridis', alpha=0.6)
plt.title('Humidity vs Max Temp (Colored by Rain)')
plt.show(block=False)
plt.pause(2)   # display for 2 seconds  
plt.close()
# -------------------------------

# 3. Define Features (X) and Targets (y)
feature_cols = ['MinTemp', 'Humidity9am', 'Pressure9am', 'WindSpeed9am', 'Temp9am']
X = df[feature_cols]

# Target 1: Max Temperature (Regression)
y_temp = df['MaxTemp']

# Target 2: Rain Tomorrow (Classification)
rain_le = LabelEncoder()
y_rain = rain_le.fit_transform(df['RainTomorrow'])

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
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_scaled, y_temp_train)

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
pickle.dump(rain_le, open('rain_le.pkl', 'wb'))
print("Success! Models saved.")

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
