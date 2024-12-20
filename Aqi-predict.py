
# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'AirQualityUCI.csv'  # Replace with your file's path
data = pd.read_csv(file_path, sep=';', decimal=',', engine='python')

# Combine Date and Time into a single Datetime column
data['Datetime'] = pd.to_datetime(
    data['Date'] + ' ' + data['Time'], 
    format='%d/%m/%Y %H:%M:%S', 
    errors='coerce'
)
data.drop(columns=['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'], inplace=True)

# Replace -200 with NaN to mark missing values
data.replace(-200, np.nan, inplace=True)

# Handle missing values (e.g., fill with column mean)
data.fillna(data.mean(), inplace=True)

# Select features and target (assuming `CO(GT)` as AQI, adapt as needed)
target = 'CO(GT)'  # Replace with your target column
features = [col for col in data.columns if col not in [target, 'Datetime']]

# Drop rows with missing target values
data = data.dropna(subset=[target])

# Split the data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Plot predicted vs actual
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('Actual vs Predicted AQI')
plt.show()