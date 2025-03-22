# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import warnings
from scipy.stats import zscore

warnings.filterwarnings("ignore")

# Load the dataset
file_path = 'C:/Users/Ranshika/Documents/Prolog/bed_occupancy_rate_pro.csv'
data = pd.read_csv(file_path)

# Preprocessing
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m')  # Convert to datetime 
data.set_index('Date', inplace=True)  # Set Date as the index

# Target variable
target = data['Bed_occupancy_rate_%']



# Exogenous variables (drop columns not needed)
exog = data.drop(columns=['Bed_occupancy_rate_%'])  # Exclude the target

# Visualize the target variable
plt.figure(figsize=(12, 6))
plt.plot(target, label='Bed Occupancy Rate (%)')
plt.title('Bed Occupancy Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Occupancy Rate (%)')
plt.legend()
plt.show()

# Step 1: Correlation Analysis
correlation_matrix = data.corr()

# Visualize correlations with the target variable
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix[['Bed_occupancy_rate_%']].sort_values(by='Bed_occupancy_rate_%', ascending=False),
            annot=True, cmap='coolwarm')
plt.title('Correlation of Features with Bed Occupancy Rate (%)')
plt.show()

# Step 2: Identify Outliers
# Calculate Z-scores
z_scores = np.abs((target - target.mean()) / target.std())
outliers_z = target[z_scores > 3]

# Calculate IQR
Q1 = target.quantile(0.25)
Q3 = target.quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = target[(target < (Q1 - 1.5 * IQR)) | (target > (Q3 + 1.5 * IQR))]

print("Z-Score Outliers:\n", outliers_z)
print("IQR Outliers:\n", outliers_iqr)

# Handle Outliers using Interpolation
# Replace outliers (both Z-Score and IQR identified) with interpolated values
outlier_dates = outliers_z.index.union(outliers_iqr.index)  # Combine dates of outliers from both methods
target.loc[outlier_dates] = np.nan  # Set outliers to NaN

# Interpolate to fill missing (outlier) values
target = target.interpolate(method='time')  # Time-based interpolation

# Verify that outliers have been replaced
plt.figure(figsize=(12, 6))
plt.plot(target, label='Bed Occupancy Rate (%) (After Outlier Handling)')
plt.title('Bed Occupancy Rate After Handling Outliers')
plt.xlabel('Date')
plt.ylabel('Occupancy Rate (%)')
plt.legend()
plt.show()

# Split into training and testing sets
train_size = int(len(target) * 0.8)
train_data = target[:train_size]
test_data = target[train_size:]

# Split exogenous variables similarly
train_exog = exog[:train_size]
test_exog = exog[train_size:]

# Step 3: Fit a basic ARIMAX model with all features and evaluate p-values
arima_order = (5, 1, 0)  # Modify p, d, q based on ACF/PACF
arimax_model = ARIMA(train_data, order=arima_order, exog=train_exog)
arimax_fitted = arimax_model.fit()

# Print the summary to check p-values
print(arimax_fitted.summary())

# Step 4: Recursive Feature Elimination (RFE)
# Fit a linear regression model for RFE
lr = LinearRegression()
rfe = RFE(lr, n_features_to_select=5)  # Select the top 5 features
rfe.fit(train_exog, train_data)

# Display selected features
selected_features = train_exog.columns[rfe.support_]
print("Selected Features:", selected_features)

# Step 5: Evaluate different feature sets
feature_sets = [
    ['Total_no_of_Admissions', 'Total_no_of_new_Registrations'],  # Example set 1
    ['Average_Daily_inpatients', 'Total_inpatients_days'],         # Example set 2
    exog.columns.tolist()                                          # All features
]

for i, features in enumerate(feature_sets, 1):
    arimax_model = ARIMA(train_data, order=arima_order, exog=train_exog[features])
    arimax_fitted = arimax_model.fit()
    forecast = arimax_fitted.forecast(steps=len(test_data), exog=test_exog[features])
    mse = mean_squared_error(test_data, forecast)
    print(f"Feature Set {i}: MSE = {mse}")

# Step 6: Visualize the best model
# For demonstration, using the last feature set
best_features = exog.columns.tolist()  # Replace with the selected features if applicable
arimax_model = ARIMA(train_data, order=arima_order, exog=train_exog[best_features])
arimax_fitted = arimax_model.fit()
forecast = arimax_fitted.forecast(steps=len(test_data), exog=test_exog[best_features])

plt.figure(figsize=(12, 6))
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Actual Test Data', color='green')
plt.plot(test_data.index, forecast, label='ARIMAX Forecast', color='orange')
plt.title('ARIMAX Model Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Bed Occupancy Rate (%)')
plt.legend()
plt.show()



# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from scipy.stats import shapiro, probplot
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
file_path = 'C:/Users/Ranshika/Documents/Prolog/bed_occupancy_rate_augemented.csv'
data = pd.read_csv(file_path)

# Preprocessing
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m')  # Convert to datetime
data.set_index('Date', inplace=True)  # Set Date as the index

# Target variable
target = data['Bed_occupancy_rate_%']

# Variance stabilization: Log-transform the target variable
target_log = np.log(target)

# Handle Outliers: Z-Score and IQR methods combined
z_scores = np.abs(zscore(target_log))
Q1 = target_log.quantile(0.25)
Q3 = target_log.quantile(0.75)
IQR = Q3 - Q1

# Identify outliers
outliers = target_log[(z_scores > 3) | (target_log < (Q1 - 1.5 * IQR)) | (target_log > (Q3 + 1.5 * IQR))]
print(f"Outliers identified: {len(outliers)}")

# Winsorize outliers (cap at 1st and 99th percentiles)
lower_bound = target_log.quantile(0.01)
upper_bound = target_log.quantile(0.99)
target_log = target_log.clip(lower=lower_bound, upper=upper_bound)

# Feature Selection (Exogenous Variables)
exog = data.drop(columns=['Bed_occupancy_rate_%'])

# Step 1: Split into train and test sets
train_size = int(len(target_log) * 0.8)
train_data = target_log[:train_size]
test_data = target_log[train_size:]

train_exog = exog[:train_size]
test_exog = exog[train_size:]

