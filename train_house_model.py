import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

print("Loading dataset...")

# Load dataset (file must be in same folder)
data = pd.read_csv("House Price India.csv")

print("Dataset loaded successfully!")

# Select target column
y = data["Price"]

# Select features (remove Price column)
X = data.drop("Price", axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("house_price_model.pkl", "wb"))

print("✅ Model trained and saved successfully!")
