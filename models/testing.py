import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR

df = pd.read_csv('/home/imayavan/Downloads/smart-grid-management-main/models/smart_home_energy_usage_dataset.csv')
input_columns = ['temperature_setting_C', 'occupancy_status', 'appliance', 
                 'usage_duration_minutes', 'season', 'day_of_week']
target_column = 'energy_consumption_kWh'
for col in input_columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])
X = df[input_columns]
y = df[target_column].values.ravel()

X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
mse, rmse, r2, mape = mean_squared_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred), mean_absolute_percentage_error(y_test, y_pred)

print(f'MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}, Accuracy: {(1 - mape):.2%}')

