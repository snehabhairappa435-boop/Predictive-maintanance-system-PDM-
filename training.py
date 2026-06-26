import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import joblib
import os

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
df = pd.read_csv("converted_predictive_maintenance.csv")

# Keep only sensor columns
features = ["temperature", "vibration", "pressure", "humidity"]
X = df[features]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# ---------------------------------------------------
# MODEL 1: ISOLATION FOREST
# ---------------------------------------------------
print("Training Isolation Forest...")

iso_model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)
iso_model.fit(X_scaled)

joblib.dump(iso_model, "models/isolation_forest.pkl")
print("Isolation Forest saved!")

# ---------------------------------------------------
# MODEL 2: AUTOENCODER
# ---------------------------------------------------
print("Training Autoencoder...")

input_dim = X_scaled.shape[1]

input_layer = Input(shape=(input_dim,))
encoder = Dense(8, activation="relu")(input_layer)
encoder = Dense(4, activation="relu")(encoder)
decoder = Dense(8, activation="relu")(encoder)
output_layer = Dense(input_dim, activation="linear")(decoder)

autoencoder = Model(inputs=input_layer, outputs=output_layer)

autoencoder.compile(optimizer=Adam(0.001), loss="mse")

autoencoder.fit(
    X_scaled, X_scaled,
    epochs=50,
    batch_size=16,
    verbose=1
)

autoencoder.save("models/autoencoder_model.keras")
print("Autoencoder saved!")

print("\n BOTH MODELS TRAINED & SAVED SUCCESSFULLY!")
