import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("weather.csv")

# Features
X = df[['Humidity', 'WindSpeed', 'Pressure']]

# Target
y = df['Temperature']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("weather_model.pkl", "wb"))

print("Model Trained Successfully")