# Air Quality Index (AQI) Prediction

## Objective
This project aims to predict the Air Quality Index (AQI) using pollutant and meteorological data.

## Dataset
- The dataset contains 9,358 instances of hourly readings from sensors and meteorological parameters.
- Missing values have been handled appropriately by replacing sensor anomalies (-200) with column averages.

## Steps
1. **Data Preprocessing**
   - Combined `Date` and `Time` into a single `Datetime` column.
   - Removed unnecessary columns (`Unnamed: 15`, `Unnamed: 16`).
   - Replaced missing values (-200) with the column mean.

2. **Modeling**
   - A Random Forest Regressor was trained on features derived from sensor and weather data.
   - Split data into training (80%) and testing (20%) subsets.
   - Evaluated the model using Root Mean Squared Error (RMSE).

3. **Results**
   - Achieved RMSE of approximately ` 0.5344666833053828`.

4. **Requirements**
   Install dependencies using:
   ```bash
   pip install -r requirements.txt
5. **Execution**
   Run the Python script:
   ```bash
   python Aqi-predict.py

