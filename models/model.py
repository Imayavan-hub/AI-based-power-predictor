import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

# Load dataset
df = pd.read_csv('/home/imayavan/Downloads/smart-grid-management-main/models/smart_home_energy_usage_dataset.csv')

# Define input and target columns
input_columns = ['temperature_setting_C', 'occupancy_status', 'appliance', 
                 'usage_duration_minutes', 'season', 'day_of_week']
target_columns = ['energy_consumption_kWh']

# Encode categorical features
label_encoders = {}
for col in input_columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Extract input features and target variable
X = df[input_columns]
y = df[target_columns].values.ravel()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict on test data
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb)
accuracy_xgb = 1 - mape_xgb

# Save model
joblib.dump(xgb_model, 'xgboost_model.pkl')

# Print performance metrics
print(f'XGBoost - Mean Squared Error (MSE): {mse_xgb:.2f}')
print(f'XGBoost - Root Mean Squared Error (RMSE): {rmse_xgb:.2f}')
print(f'XGBoost - R² Score: {r2_xgb:.2f}')
print(f'XGBoost - Accuracy: {accuracy_xgb:.2%}')

# Create subplots for multiple graphs in a single window
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

sns.lineplot(x=df['temperature_setting_C'], y=df['energy_consumption_kWh'], ax=axes[0, 0])
axes[0, 0].set_title('Temperature Setting vs Energy Consumption')

sns.lineplot(x=df['occupancy_status'], y=df['energy_consumption_kWh'], ax=axes[0, 1])
axes[0, 1].set_title('Occupancy Status vs Energy Consumption')

sns.lineplot(x=df['appliance'], y=df['energy_consumption_kWh'], ax=axes[0, 2])
axes[0, 2].set_title('Appliance vs Energy Consumption')

sns.lineplot(x=df['usage_duration_minutes'], y=df['energy_consumption_kWh'], ax=axes[1, 0])
axes[1, 0].set_title('Usage Duration vs Energy Consumption')

sns.lineplot(x=df['season'], y=df['energy_consumption_kWh'], ax=axes[1, 1])
axes[1, 1].set_title('Season vs Energy Consumption')

sns.lineplot(x=df['day_of_week'], y=df['energy_consumption_kWh'], ax=axes[1, 2])
axes[1, 2].set_title('Day of Week vs Energy Consumption')

plt.tight_layout()  # Adjust layout for better visibility
plt.show()

