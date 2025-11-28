import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load datasets
train = pd.read_csv("train.csv")
features = pd.read_csv("features.csv")
stores = pd.read_csv("stores.csv")

# Merge data
df = train.merge(features, on=["Store", "Date"], how="left")
df = df.merge(stores, on="Store", how="left")

# Clean and preprocess
df["Date"] = pd.to_datetime(df["Date"])
df = df.dropna()
df["Type"] = df["Type"].map({"A": 0, "B": 1, "C": 2})

X = df[["Store", "Dept", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Size", "Type"]]
y = df["Weekly_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = XGBRegressor()
model.fit(X_train, y_train)

# Save model
import os
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")
