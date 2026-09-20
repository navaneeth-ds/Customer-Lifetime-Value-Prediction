import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the dataset
file_path = "Online Retail.xlsx"
df = pd.read_excel(file_path)

# 2. Display basic information
print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

# 3. Clean the dataset

# Remove rows without CustomerID
df = df.dropna(subset=["CustomerID"])

# Remove cancelled orders
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Keep only positive quantities
df = df[df["Quantity"] > 0]

# Keep only positive prices
df = df[df["UnitPrice"] > 0]

# Convert InvoiceDate to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Calculate total amount for each transaction
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

print("\nCleaned dataset shape:")
print(df.shape)

print("\nCleaned dataset:")
print(df.head())

# 4. Create customer-level features

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

customer_data = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    TotalSpend=("TotalAmount", "sum")
)

customer_data["AOV"] = (
    customer_data["TotalSpend"] / customer_data["Frequency"]
)

print("\nCustomer-level data:")
print(customer_data.head())

print("\nNumber of customers:")
print(len(customer_data))

# 5. Create future customer value (prediction target)

cutoff_date = pd.Timestamp("2011-09-01")

history = df[df["InvoiceDate"] < cutoff_date]
future = df[df["InvoiceDate"] >= cutoff_date]

# Customer features from historical purchases
features = history.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (cutoff_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    TotalSpend=("TotalAmount", "sum")
)

features["AOV"] = features["TotalSpend"] / features["Frequency"]

# Future spending = target we want the model to predict
target = future.groupby("CustomerID")["TotalAmount"].sum()

model_data = features.join(target.rename("FutureValue"), how="left")

# Customers with no future purchases have value 0
model_data["FutureValue"] = model_data["FutureValue"].fillna(0)

print("\nModel dataset:")
print(model_data.head())

print("\nModel dataset shape:")
print(model_data.shape)

print("\nFuture value statistics:")
print(model_data["FutureValue"].describe())

# 6. Train the Customer Lifetime Value prediction model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Features used by the model
X = model_data[["Recency", "Frequency", "TotalSpend", "AOV"]]

# Target value
y = model_data["FutureValue"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("\nModel Performance:")
print("MAE:", mae)
print("RMSE:", rmse)

# 7. Create and save customer LTV predictions

model_data["Predicted_LTV"] = model.predict(
    model_data[["Recency", "Frequency", "TotalSpend", "AOV"]]
)

# Save predictions to CSV
model_data.reset_index().to_csv(
    "Customer_LTV_Predictions.csv",
    index=False
)

print("\nLTV prediction file created successfully!")
print("Saved as: Customer_LTV_Predictions.csv")

# 8. Save the trained model

import joblib

joblib.dump(model, "customer_ltv_model.pkl")

print("\nTrained model saved successfully!")
print("Saved as: customer_ltv_model.pkl")

# 9. Create visualizations

import matplotlib.pyplot as plt

# 1. LTV distribution
plt.figure(figsize=(10, 6))
plt.hist(model_data["FutureValue"], bins=50)
plt.title("Customer Future Value Distribution")
plt.xlabel("Future Customer Value")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("ltv_distribution.png")
plt.close()

# 2. Actual vs Predicted LTV
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Future Value")
plt.ylabel("Predicted Future Value")
plt.title("Actual vs Predicted Customer LTV")
plt.tight_layout()
plt.savefig("actual_vs_predicted_ltv.png")
plt.close()

# 3. Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importance.plot(kind="bar")
plt.title("Customer LTV Model - Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

print("\nVisualizations created successfully!")