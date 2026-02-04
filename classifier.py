import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("car_prediction_data.csv")

# Drop unused column
df.drop("Car_Name", axis=1, inplace=True)

# Encode categorical columns
df["Fuel_Type"] = df["Fuel_Type"].map({"Petrol": 0, "Diesel": 1, "CNG": 2})
df["Seller_Type"] = df["Seller_Type"].map({"Dealer": 0, "Individual": 1})
df["Transmission"] = df["Transmission"].map({"Manual": 0, "Automatic": 1})

# Features & target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model & columns
pickle.dump(model, open("carprice_model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))

print("✅ Model trained and saved successfully")
