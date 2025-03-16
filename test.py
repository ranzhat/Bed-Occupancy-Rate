# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro, probplot
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
file_path = 'C:/Users/Ranshika/Documents/Prolog/bed_occupancy_rate_pro.csv'
data = pd.read_csv(file_path)

first_6_rows = data.tail(6)
print(first_6_rows)

# Preprocessing
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m')  # Convert to datetime
data.set_index('Date', inplace=True)  # Set Date as the index

# Target variable
target = data['Bed_occupancy_rate_%']

# Variance stabilization: Log-transform the target variable
target_log = np.log(target) 

# Exogenous variables (drop columns not needed)
exog = data.drop(columns=['Bed_occupancy_rate_%'])  # Exclude the original target

# Step 1: Visualize the transformed target variable
plt.figure(figsize=(12, 6))
plt.plot(target_log, label='Log Transformed Bed Occupancy Rate (%)')
plt.title('Log Transformed Bed Occupancy Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Log Occupancy Rate')
plt.legend()
plt.show()

# Step 2: Handle outliers
# Z-Score and IQR methods to identify outliers
z_scores = np.abs((target_log - target_log.mean()) / target_log.std())
outliers_z = target_log[z_scores > 3]

Q1 = target_log.quantile(0.25)
Q3 = target_log.quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = target_log[(target_log < (Q1 - 1.5 * IQR)) | (target_log > (Q3 + 1.5 * IQR))]

# Handle Outliers: Replace with interpolated values
outlier_dates = outliers_z.index.union(outliers_iqr.index)
target_log.loc[outlier_dates] = np.nan
target_log = target_log.interpolate(method='time')  # Time-based interpolation

# Step 3: Feature Refinement with VIF
# Calculate VIF to remove multicollinearity
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

# Iteratively drop features with VIF > 10
def refine_features_vif(df, threshold=10):
    while True:
        vif = calculate_vif(df)
        max_vif = vif["VIF"].max()
        if max_vif > threshold:
            feature_to_drop = vif[vif["VIF"] == max_vif]["feature"].values[0]
            print(f"Dropping '{feature_to_drop}' with VIF = {max_vif}")
            df = df.drop(columns=[feature_to_drop])
        else:
            break
    return df

# Refine exogenous features
exog_refined = refine_features_vif(exog)

# Print refined features
print("Refined Features:", exog_refined.columns.tolist())

# Step 4: Split into training and testing sets
train_size = int(len(target_log) * 0.8)
train_data = target_log[:train_size]
test_data = target_log[train_size:]

train_exog = exog_refined[:train_size]
test_exog = exog_refined[train_size:]

# Step 5: Fit an ARIMAX model
arima_order = (2, 1, 0)
arimax_model = ARIMA(train_data, order=arima_order, exog=train_exog)
arimax_fitted = arimax_model.fit()

# Print model summary
print(arimax_fitted.summary())

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, probplot, zscore

# Step 6: Residual Analysis
residuals = arimax_fitted.resid

# Function to detect and replace outliers using Z-score
def detect_outliers(residuals, threshold=3):
    z_scores = np.abs(zscore(residuals))
    outliers = residuals[z_scores > threshold]
    return outliers, z_scores

# Replace outliers with the median of residuals
def replace_outliers_with_median(residuals, z_scores, threshold=3):
    outliers = residuals[np.abs(z_scores) > threshold]
    median_value = residuals.median()
    residuals_cleaned = residuals.copy()
    residuals_cleaned[np.abs(z_scores) > threshold] = median_value
    return residuals_cleaned

# Detect outliers and clean residuals
z_scores = zscore(residuals)
residuals_cleaned = replace_outliers_with_median(residuals, z_scores)

# Residual diagnostics
# 1. Histogram of cleaned residuals
sns.histplot(residuals_cleaned, kde=True)
plt.title('Cleaned Residuals Distribution')
plt.xlabel('Residuals')
plt.show()

# 2. Shapiro-Wilk test for normality on cleaned residuals
shapiro_test = shapiro(residuals_cleaned)
print("Shapiro-Wilk Test: W =", shapiro_test.statistic, ", p-value =", shapiro_test.pvalue)

# 3. QQ-plot for cleaned residuals
probplot(residuals_cleaned, dist="norm", plot=plt)
plt.title('QQ Plot of Cleaned Residuals')
plt.show()

# Save cleaned residuals for SVM integration
residuals_cleaned.to_csv("arima_residuals_cleaned.csv", index=True)


# Step 7: Forecast and evaluate performance
forecast = arimax_fitted.forecast(steps=len(test_data), exog=test_exog)

# Evaluate MSE on the original scale
mse = mean_squared_error(np.exp(test_data), np.exp(forecast))
print("Mean Squared Error (MSE):", mse)

# Visualize the forecast
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, np.exp(test_data), label='Actual Test Data (Original Scale)', color='green')
plt.plot(test_data.index, np.exp(forecast), label='ARIMAX Forecast (Original Scale)', color='orange')
plt.plot(train_data.index, np.exp(train_data), label='Training Data (Original Scale)', color='blue')
plt.title('ARIMAX Model Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Bed Occupancy Rate (%)')
plt.legend()
plt.show()

residuals.to_csv("arima_residuals.csv", index=True)

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Load residuals and predictors
residuals = pd.read_csv("arima_residuals.csv", index_col=0)
features = exog_refined.loc[residuals.index]  # Match indices

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(features, residuals, test_size=0.2, random_state=42)

# SVM Model
svm_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svm_model.fit(X_train, y_train)

# Predict and Evaluate
residual_predictions = svm_model.predict(X_test)
mse_residuals = mean_squared_error(y_test, residual_predictions)
print("Residuals MSE:", mse_residuals)

# Extend residual_predictions to match the forecast index length by padding with NaN
residual_predictions_padded = np.pad(residual_predictions, (0, len(forecast) - len(residual_predictions)), mode='constant', constant_values=np.nan)

# Create a Series for residual_predictions with the full forecast index
residual_predictions_series = pd.Series(residual_predictions_padded, index=forecast.index)

# Now align the forecast with residual predictions
forecast_aligned = forecast + residual_predictions_series

final_forecast = forecast + residual_predictions_series

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Assuming 'df' is your DataFrame and 'actual_values' is the column with actual observed data
actual_values = data['Bed_occupancy_rate_%']

final_forecast = final_forecast.fillna(final_forecast.mean())  # Or use .fillna(0) to replace with zero


print(actual_values.isna().sum())  # Check if there are NaN values in actual_values
print(final_forecast.isna().sum())  # Check if there are NaN values in final_forecast


# Make sure the index aligns with your forecast
actual_values = actual_values.loc[final_forecast.index]

# Calculate MAE, MSE, and RMSE
mae = mean_absolute_error(actual_values, final_forecast)
mse = mean_squared_error(actual_values, final_forecast)
rmse = np.sqrt(mse)

print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

import matplotlib.pyplot as plt

# Plotting actual values vs forecast
plt.figure(figsize=(10, 6))
plt.plot(actual_values, label='Actual Values', color='blue')
plt.plot(final_forecast, label='Forecasted Values', color='red', linestyle='dashed')
plt.title('Actual vs Forecasted Bed Occupancy Rate')
plt.xlabel('Date')
plt.ylabel('Bed Occupancy Rate (%)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate residuals
residuals = actual_values - final_forecast

# Plotting residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals', color='purple')
plt.title('Residuals of the Forecast')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.axhline(y=0, color='black', linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Assuming you have lower and upper bounds of the forecast (you would need to calculate these separately)
# Lower bound and upper bound example (adjust according to your model's output)
lower_bound = final_forecast - 1.96 * residuals.std()
upper_bound = final_forecast + 1.96 * residuals.std()

# Plot actual, forecasted, and confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(actual_values, label='Actual Values', color='blue')
plt.plot(final_forecast, label='Forecasted Values', color='red', linestyle='dashed')
plt.fill_between(actual_values.index, lower_bound, upper_bound, color='gray', alpha=0.2, label='95% Confidence Interval')
plt.title('Forecast with 95% Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Bed Occupancy Rate (%)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and preprocess the data
# (Assumes your earlier preprocessing and ARIMA-SVM workflow is complete)

# Ensure alignment of actual and forecasted values
actual_values = data['Bed_occupancy_rate_%']
final_forecast = final_forecast.fillna(final_forecast.mean())  # Replace NaN values in forecast (e.g., with mean)

# Ensure index alignment
actual_values = actual_values.loc[final_forecast.index]

# Performance Metrics Calculation
# MAE
mae = mean_absolute_error(actual_values, final_forecast)

# MSE
mse = mean_squared_error(actual_values, final_forecast)

# RMSE
rmse = np.sqrt(mse)

# MAPE
mape = np.mean(np.abs((actual_values - final_forecast) / actual_values)) * 100

# sMAPE
smape = 100 * np.mean(
    2 * np.abs(actual_values - final_forecast) / (np.abs(actual_values) + np.abs(final_forecast))
)

# R-squared
r2 = r2_score(actual_values, final_forecast)

# Print all the metrics
print(f"Performance Metrics:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}%")
print(f"sMAPE: {smape:.4f}%")
print(f"R-squared: {r2:.4f}")

# Visualization of Actual vs Forecasted Values
plt.figure(figsize=(10, 6))
plt.plot(actual_values, label='Actual Values', color='blue')
plt.plot(final_forecast, label='Forecasted Values', color='red', linestyle='dashed')
plt.title('Actual vs Forecasted Bed Occupancy Rate')
plt.xlabel('Date')
plt.ylabel('Bed Occupancy Rate (%)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Residual Analysis
residuals = actual_values - final_forecast

# Residuals Plot
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals', color='purple')
plt.title('Residuals of the Forecast')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.axhline(y=0, color='black', linestyle='--')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Confidence Intervals
# Assuming standard deviation of residuals is normally distributed
residual_std = residuals.std()
lower_bound = final_forecast - 1.96 * residual_std
upper_bound = final_forecast + 1.96 * residual_std

# Plot Actual, Forecast, and Confidence Interval
plt.figure(figsize=(10, 6))
plt.plot(actual_values, label='Actual Values', color='blue')
plt.plot(final_forecast, label='Forecasted Values', color='red', linestyle='dashed')
plt.fill_between(final_forecast.index, lower_bound, upper_bound, color='gray', alpha=0.2, label='95% Confidence Interval')
plt.title('Forecast with 95% Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Bed Occupancy Rate (%)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Import additional libraries
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from scipy.stats import yeojohnson

# Step 1: Advanced Outlier Detection using Isolation Forest
isolation_forest = IsolationForest(contamination=0.01, random_state=42)  # Adjust contamination rate
residuals_df = pd.DataFrame(residuals, columns=["residuals"])
residuals_df["outlier_flag"] = isolation_forest.fit_predict(residuals_df[["residuals"]])

# Separate inliers and outliers
inliers = residuals_df[residuals_df["outlier_flag"] == 1]["residuals"]
outliers = residuals_df[residuals_df["outlier_flag"] == -1]["residuals"]

# Replace outliers with inlier boundaries
lower_limit = inliers.min()
upper_limit = inliers.max()
residuals_cleaned = residuals.copy()
residuals_cleaned[residuals < lower_limit] = lower_limit
residuals_cleaned[residuals > upper_limit] = upper_limit

# Step 2: Transform Residuals using Yeo-Johnson (Optional)
residuals_transformed, _ = yeojohnson(residuals_cleaned)

# Step 3: Plot Cleaned Residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals, label="Original Residuals", color="gray", linestyle="dotted", alpha=0.6)
plt.plot(residuals_cleaned, label="Cleaned Residuals (Isolation Forest)", color="purple")
plt.axhline(y=0, color="black", linestyle="--")
plt.title("Residuals Before and After Outlier Handling")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 4: Update Confidence Intervals with Cleaned Residuals
residual_std_cleaned = residuals_cleaned.std()
lower_bound_ci = final_forecast - 1.96 * residual_std_cleaned
upper_bound_ci = final_forecast + 1.96 * residual_std_cleaned

# Step 5: Plot Actual vs Forecast with Updated Confidence Intervals
plt.figure(figsize=(10, 6))
plt.plot(actual_values, label="Actual Values", color="blue")
plt.plot(final_forecast, label="Forecasted Values", color="red", linestyle="dashed")
plt.fill_between(
    final_forecast.index,
    lower_bound_ci,
    upper_bound_ci,
    color="gray",
    alpha=0.2,
    label="95% Confidence Interval (Cleaned)",
)
plt.title("Forecast with Updated Confidence Intervals (After Outlier Removal)")
plt.xlabel("Date")
plt.ylabel("Bed Occupancy Rate (%)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 6: Recalculate Performance Metrics
mae_cleaned = mean_absolute_error(actual_values, final_forecast)
mse_cleaned = mean_squared_error(actual_values, final_forecast)
rmse_cleaned = np.sqrt(mse_cleaned)
mape_cleaned = np.mean(np.abs((actual_values - final_forecast) / actual_values)) * 100
smape_cleaned = 100 * np.mean(
    2 * np.abs(actual_values - final_forecast) / (np.abs(actual_values) + np.abs(final_forecast))
)
r2_cleaned = r2_score(actual_values, final_forecast)

# Print cleaned metrics
print("Cleaned Performance Metrics:")
print(f"MAE: {mae_cleaned:.4f}")
print(f"MSE: {mse_cleaned:.4f}")
print(f"RMSE: {rmse_cleaned:.4f}")
print(f"MAPE: {mape_cleaned:.4f}%")
print(f"sMAPE: {smape_cleaned:.4f}%")
print(f"R-squared: {r2_cleaned:.4f}")



# Step 1: Forecast the full period (2024-2025)
future_dates = pd.date_range(start='2024-01-01', end='2025-12-31', freq='M')  # Monthly frequency
future_exog = pd.DataFrame(index=future_dates)

# Fill the future exogenous variables with reasonable assumptions (mean of training data for simplicity here)
for col in exog_refined.columns:
    future_exog[col] = exog_refined[col].mean()  # Replace with your method for forecasting exog variables

# Step 2: ARIMA Forecast for the full period (2024-2025)
arimax_forecast = arimax_fitted.get_forecast(steps=len(future_dates), exog=future_exog)
arima_forecast_values = arimax_forecast.predicted_mean

# Ensure ARIMA forecast is correctly generated
print("ARIMA Forecast Values:\n", arima_forecast_values.head())

# Step 3: SVM Residual Forecast for the full period (2024-2025)
svm_residual_predictions = svm_model.predict(future_exog)
future_residuals_series = pd.Series(svm_residual_predictions, index=future_dates)

# Ensure SVM residuals are generated
print("SVM Residual Predictions:\n", future_residuals_series.head())

# Step 4: Combine ARIMA and SVM residuals for the final forecast
final_forecast = arima_forecast_values + future_residuals_series
print("Final Forecast (combined ARIMA + SVM):\n", final_forecast.head())

# Step 5: Inverse the log transformation to return to the original scale if needed
final_forecast_original_scale = np.exp(final_forecast)
print("Final Forecast on Original Scale:\n", final_forecast_original_scale.head())

# Step 6: Visualize the forecast (focus on 2025)
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Bed_occupancy_rate_%'], label='Actual Data', color='blue')
plt.plot(final_forecast_original_scale.index, final_forecast_original_scale, label='Forecast (2024-2025)', color='red')
plt.title('Bed Occupancy Rate Forecast (2024-2025)')
plt.xlabel('Date')
plt.ylabel('Bed Occupancy Rate (%)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 7: Save forecast to a CSV file
forecast_output = pd.DataFrame({
    'Date': final_forecast_original_scale.index,
    'Forecasted_Bed_Occupancy_Rate': final_forecast_original_scale
})
forecast_output.to_csv("bed_occupancy_forecast_2024_2025.csv", index=False)

# Print forecast for verification
print(forecast_output.tail())  # Check the forecast for 2025

# Check if ARIMA model has been trained on the correct data and if the forecast length is correct.
print(f"ARIMA Forecast Length: {len(arima_forecast_values)}")  # Should match the number of periods forecasted (e.g., 24 months for 2024-2025)

# Also check the values of the exogenous variables to make sure they are populated for the forecast period.
print(f"Exogenous Variables for Forecast: \n{future_exog.head()}")

# Check the length of SVM residual predictions and their alignment with ARIMA forecast
print(f"SVM Residual Predictions Length: {len(future_residuals_series)}")
print(f"SVM Residual Predictions: \n{future_residuals_series.head()}")

# Combine ARIMA forecast and SVM residuals
combined_forecast = arima_forecast_values + future_residuals_series

# Check for NaN values in the combined forecast
print(f"NaN values in combined forecast: {combined_forecast.isnull().sum()}")

# If NaN values exist, try to handle them with interpolation
combined_forecast_filled = combined_forecast.interpolate(method='linear')
print(f"Interpolated Combined Forecast:\n {combined_forecast_filled.head()}")


# Filter the ARIMA forecast values to only include the forecast period (2024-2025)
arima_forecast_values = arima_forecast_values.loc['2024-01-01':'2025-12-31']

# Filter the SVM residuals to match the same period
future_residuals_series = future_residuals_series.loc['2024-01-01':'2025-12-31']

# Combine them by adding the residuals to the ARIMA forecast
combined_forecast = arima_forecast_values.add(future_residuals_series, fill_value=0)

# Check the combined forecast
print(f"Combined Forecast (ARIMA + SVM):\n {combined_forecast.head()}")

print(f"ARIMA Forecast Length: {len(arima_forecast_values)}")
print(f"SVM Residuals Length: {len(future_residuals_series)}")

# If any NaN values remain, apply interpolation to fill them
combined_forecast_filled = combined_forecast.interpolate(method='linear')

# Check the forecast for missing values
print(f"Interpolated Combined Forecast:\n {combined_forecast_filled.head()}")

# If there was any log transformation earlier, reverse it
final_forecast_original_scale = np.exp(combined_forecast_filled)

# Generate the final forecast DataFrame
forecast_output = pd.DataFrame({
    'Date': combined_forecast_filled.index,
    'Forecasted_Bed_Occupancy_Rate': final_forecast_original_scale
})

# Save the forecast to a CSV file for further analysis
forecast_output.to_csv("bed_occupancy_forecast_2024_2025.csv", index=False)
print(forecast_output.tail())






from statsmodels.tsa.stattools import adfuller

# Perform the ADF test on the log-transformed target variable
adf_result = adfuller(target_log)

# Print the results
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Critical Values:")
for key, value in adf_result[4].items():
    print(f"   {key}: {value}")

# Interpret the p-value
if adf_result[1] < 0.05:
    print("The time series is stationary (reject null hypothesis).")
else:
    print("The time series is not stationary (fail to reject null hypothesis).")




