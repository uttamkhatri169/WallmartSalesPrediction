import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

# Load training data
train = pd.read_csv("train.csv")
features = pd.read_csv("features.csv")
stores = pd.read_csv("stores.csv")

# Merge
df = train.merge(features, on=["Store", "Date"], how="left")
df = df.merge(stores, on="Store", how="left")

# Preprocess
df["Date"] = pd.to_datetime(df["Date"])
df = df.dropna()
df["Type"] = df["Type"].map({"A": 0, "B": 1, "C": 2})

# Feature columns and target
X = df[["Store", "Dept", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Size", "Type"]]
y_true = df["Weekly_Sales"]
y_pred = model.predict(X)

# Metrics
import numpy as np
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# Optional: plot
plt.figure(figsize=(10, 6))
plt.scatter(y_true[:100], y_pred[:100], alpha=0.5, color='green')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Predicted vs Actual Sales (Sample)')
plt.grid(True)
plt.show()
