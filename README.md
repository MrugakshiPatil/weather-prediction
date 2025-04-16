# weather-prediction

This project uses a simple Linear Regression model to predict temperature based on weather features like humidity, wind speed, and pressure. Built in Python using Pandas, Matplotlib, and Scikit-Learn, it includes data generation, visualization, model training, and evaluation. Ideal for beginners exploring machine learning with environmental data.

#code 
# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the dataset
# If you just created the file in the same session:
df = pd.read_csv('weather_data_500.csv')

# If you're uploading the file manually:
# from google.colab import files
# uploaded = files.upload()
# df = pd.read_csv('weather_data_500.csv')

# Step 3: Display basic info
print("Dataset preview:")
display(df.head())

print("\nDataset summary:")
print(df.describe())

# Step 4: Visualize relationships
sns.pairplot(df)
plt.suptitle('Feature Relationships', y=1.02)
plt.show()

# Step 5: Split into features and target
X = df[['humidity', 'wind_speed', 'pressure']]
y = df['temperature']

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Make Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 10: Plot Predictions
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted Temperature")
plt.grid(True)
plt.show()


