# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from scipy.stats import zscore
import warnings

warnings.filterwarnings("ignore")

# Load the CSV file into a DataFrame
file_path = 'C:/Users/Ranshika/Documents/Prolog/bed_occupancy_rate_pro.csv'
df = pd.read_csv(file_path)

# Check for missing and duplicate values
print("Missing values per column:\n", df.isnull().sum())
print("Number of duplicates:", df.duplicated().sum())

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Identify outliers using z-scores
z_scores = df.select_dtypes(include=np.number).apply(zscore)
threshold = 3
outliers = (z_scores.abs() > threshold).any(axis=1)
outlier_indices = df[outliers].index
print("Outlier Indices (Z-Score):", list(outlier_indices))
print("Number of Outliers:", len(outlier_indices))

# Replace outliers with the median value for each column
df_cleaned = df.copy()
for col in df.select_dtypes(include=np.number).columns:
    median = df_cleaned[col].median()
    df_cleaned.loc[outlier_indices, col] = median

# Verify the shape after replacing outliers
print("Shape of the dataset after replacing outliers with median:", df_cleaned.shape)

# Feature selection and target definition
X = df_cleaned.drop(columns=["Date", "Bed_occupancy_rate_%"])
y = df_cleaned["Bed_occupancy_rate_%"]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Verify the scaled features
scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print("Scaled Feature Set:\n", scaled_df.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the SVM regression model
svm_model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Calculate residuals
residuals = y_test - y_pred

# Identify and eliminate residual outliers using z-scores
residual_z_scores = zscore(residuals)
residual_threshold = 3
valid_indices = np.abs(residual_z_scores) <= residual_threshold

# Filter out residual outliers
X_test_filtered = X_test[valid_indices]
y_test_filtered = y_test[valid_indices]
y_pred_filtered = y_pred[valid_indices]
residuals_filtered = residuals[valid_indices]

# Recalculate evaluation metrics after removing outliers
mse_filtered = mean_squared_error(y_test_filtered, y_pred_filtered)
r2_filtered = r2_score(y_test_filtered, y_pred_filtered)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation results
print("Mean Squared Error (MSE) after filtering residual outliers:", mse_filtered)
print("R² Score after filtering residual outliers:", r2_filtered)

# Plot residual distribution after filtering
plt.figure(figsize=(8, 6))
sns.histplot(residuals_filtered, kde=True, color='skyblue')
plt.title("Filtered Residual Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Scatter plot of residuals after filtering
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_filtered, residuals_filtered, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Predictions (Filtered)")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()

# Hyperparameter tuning with filtered data
param_grid = {
    'C': [1, 10, 100, 1000],
    'gamma': [0.01, 0.1, 1, 10],
    'epsilon': [0.1, 0.2, 0.5, 1]
}
grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best R² Score:", grid_search.best_score_)

from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, max_error, median_absolute_error, mean_squared_log_error
import numpy as np

# Calculate additional metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
evs = explained_variance_score(y_test, y_pred)
max_err = max_error(y_test, y_pred)
med_ae = median_absolute_error(y_test, y_pred)
msle = mean_squared_log_error(y_test, y_pred)

# Adjusted R²
n = len(y_test)
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Symmetric Mean Absolute Percentage Error (SMAPE)
smape = 100 / len(y_test) * np.sum(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred)))

# Print all metrics
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("Explained Variance Score (EVS):", evs)
print("Max Error:", max_err)
print("Median Absolute Error:", med_ae)
print("Mean Squared Logarithmic Error (MSLE):", msle)
print("Adjusted R² Score:", adj_r2)
print("Symmetric Mean Absolute Percentage Error (SMAPE):", smape)


# Permutation importance with filtered data
perm_importance = permutation_importance(svm_model, X_test_filtered, y_test_filtered, n_repeats=10, random_state=42)
features = X.columns
importance = perm_importance.importances_mean

plt.figure(figsize=(10, 6))
plt.barh(features, importance, color='skyblue')
plt.title("Feature Importance (Permutation)")
plt.xlabel("Mean Importance Score")
plt.ylabel("Features")
plt.show()

# Random Forest for comparison
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest with filtered data
rf_pred = rf_model.predict(X_test_filtered)
print("Random Forest MSE (Filtered):", mean_squared_error(y_test_filtered, rf_pred))
print("Random Forest R² Score (Filtered):", r2_score(y_test_filtered, rf_pred))

# Prepare future data (2024-2025)
future_data = {
    "Date": pd.date_range(start="2024-01-01", end="2025-12-31", freq="M"),
    "Total_hospital_deaths": [160.98] * 24,  # Replace with realistic/extracted data
    "Death_rate_%": [3.21] * 24,             # Replace with realistic/extracted data
    "Bed_turnover": [11.46] * 24             # Replace with realistic/extracted data
}

# Convert to DataFrame
future_df = pd.DataFrame(future_data)

# Feature columns from the training data
feature_columns = X.columns.tolist()

# Ensure all necessary features are present, matching the training features
# Add missing columns with default values (if necessary)
missing_features = set(feature_columns) - set(future_df.columns)
for feature in missing_features:
    future_df[feature] = 0  # Assign default value or an appropriate estimate

# Reorder columns to match the training data (this ensures feature alignment)
future_df = future_df[["Date"] + feature_columns]  # Keep "Date" as the first column

# Scale the future data using the scaler
X_future_scaled = scaler.transform(future_df[feature_columns])

# Predict future values
future_predictions = svm_model.predict(X_future_scaled)

# Add predictions to the DataFrame
future_df["Predicted_Bed_Occupancy_Rate"] = future_predictions

# Print Date and Predictions
print(future_df[["Date", "Predicted_Bed_Occupancy_Rate"]])

# Save predictions to a CSV file
future_df.to_csv("future_forecasts_2024_2025.csv", index=False)

# Visualize predictions
plt.figure(figsize=(12, 6))
plt.plot(future_df["Date"], future_predictions, marker='o', label="Forecasted Bed Occupancy Rate")
plt.xlabel("Date")
plt.ylabel("Bed Occupancy Rate (%)")
plt.title("Forecasted Bed Occupancy Rates (2024-2025)")
plt.legend()
plt.grid()
plt.show()


