# Importing Libraries
# General Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# Random Forest
from sklearn.ensemble import RandomForestRegressor

# XGBoost
import xgboost as xgb
from xgboost import XGBRegressor

# SVM
from sklearn.svm import SVR

# Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Visualization Settings
sns.set(rc={'figure.figsize': (10, 8)})
sns.set_style('white')

# Load and Explore Data
clean = pd.read_csv('boston_cleaned_corrected.csv')
print(clean.head())

# Neighborhood Analysis
neighbour = clean['neighbourhood_cleansed'].value_counts()
(neighbour / clean.shape[0]).plot(kind='bar')
plt.title('Neighbour Cities in Boston Airbnb')
plt.xlabel('Neighbour Cities')
plt.show()

# Cancellation Policy Analysis
policy = clean['cancellation_policy'].value_counts()
(policy / clean.shape[0]).plot(kind='bar')
plt.title('Cancellation Policies in Boston Airbnb')
plt.xlabel('Cancellation Type')
plt.show()

# Room Type Analysis
room = clean['room_type'].value_counts()
(room / clean.shape[0]).plot(kind='bar')
plt.title('Room Types in Boston Airbnb')
plt.xlabel('Room Type')
plt.show()

# Subset Data for Price Analysis
sub_6 = clean[clean['price'] < 600]

# Top 10 Neighborhoods
neighbour_counts = clean['neighbourhood_cleansed'].value_counts()
top10_neighbourhoods = neighbour_counts.head(10).index.tolist()
sub_top10 = sub_6[sub_6['neighbourhood_cleansed'].isin(top10_neighbourhoods)]

# Neighborhood Price Distribution
sns.violinplot(data=sub_top10, x='neighbourhood_cleansed', y='price', palette='OrRd')
plt.xticks(rotation=90)
plt.title('Price Distribution in Top 10 Neighborhoods')
plt.show()

# Scatterplot of Prices with Longitude and Latitude
map_data = pd.read_csv('boston_cleaned_corrected_map.csv')
sub_map = map_data[map_data['price'] < 600]

plt.scatter(sub_map['longitude'], sub_map['latitude'], c=sub_map['price'], cmap='jet', alpha=0.4)
plt.colorbar(label='Price')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Scatterplot of Prices by Location')
plt.show()

# Data Preparation for Modeling
data = pd.read_csv('boston_clean_python_2.csv')
data = data.drop(columns=['experiences_offered'])
data = data.replace({'t': 1, 'f': 0})

X = data.drop(columns=['price'])
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=125, max_depth=9, max_features=80, random_state=66)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Feature Importance
feature_importances = rf_model.feature_importances_
features_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

plt.barh(features_df['Feature'].head(20), features_df['Importance'].head(20), color='darkgreen')
plt.gca().invert_yaxis()
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
plt.show()

# Random Forest Metrics
rf_metrics = {
    'MSE': mean_squared_error(y_test, y_pred_rf),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'R2': r2_score(y_test, y_pred_rf),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_rf)
}
print('Random Forest Metrics:', rf_metrics)

# XGBoost Model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=8, subsample=0.8, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# XGBoost Metrics
xgb_metrics = {
    'MSE': mean_squared_error(y_test, y_pred_xgb),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
    'R2': r2_score(y_test, y_pred_xgb),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_xgb)
}
print('XGBoost Metrics:', xgb_metrics)

# SVM Model
svr_model = SVR(kernel='rbf', C=300, gamma=0.05, epsilon=0.3)
svr_model.fit(X_train, y_train)
y_pred_svm = svr_model.predict(X_test)

# SVM Metrics
svm_metrics = {
    'MSE': mean_squared_error(y_test, y_pred_svm),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_svm)),
    'R2': r2_score(y_test, y_pred_svm),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_svm)
}
print('SVM Metrics:', svm_metrics)

# Neural Network Model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

nn_model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
nn_model.fit(X_train_scaled, y_train, epochs=150, batch_size=27, validation_data=(X_test_scaled, y_test))

y_pred_nn = nn_model.predict(X_test_scaled).flatten()

# Neural Network Metrics
nn_metrics = {
    'MSE': mean_squared_error(y_test, y_pred_nn),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_nn)),
    'R2': r2_score(y_test, y_pred_nn),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_nn)
}
print('Neural Network Metrics:', nn_metrics)
