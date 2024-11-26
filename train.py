import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
print("Loading dataset...")
df = pd.read_csv('transactions_cleaned.csv')
print(f"Loaded {len(df)} transactions")

# Prepare features and target
features = ['INSTANCE_DATE', 'PROCEDURE_AREA', 'ROOMS_EN', 'TOTAL_BUYER', 'TOTAL_SELLER', 'LATITUDE', 'LONGITUDE', 'BURJ_DIST', 'DUBAIMALL_DIST', 'MARINA_DIST']
target = 'TRANS_VALUE'

# Convert INSTANCE_DATE to numeric
print("Converting dates...")
df['INSTANCE_DATE'] = pd.to_datetime(df['INSTANCE_DATE'])
df['INSTANCE_DATE'] = (df['INSTANCE_DATE'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')

# Encode categorical variables
print("Encoding categorical variables...")
le = LabelEncoder()
df['ROOMS_EN'] = le.fit_transform(df['ROOMS_EN'])

# Prepare features and target
X = df[features]
y = df[target]

# Log-transform the target variable (optional, for better performance)
print("Transforming target variable...")
y = np.log1p(y)

# Split into train and test sets
print("Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define LightGBM model
print("Setting up train model...")
model = lgb.LGBMRegressor(random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'num_leaves': [31, 50, 70]
}

# Grid search with cross-validation
print("Starting GridSearchCV...")
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_scaled, y_train)

# Best model
print("\nBest Parameters:")
print(grid_search.best_params_)
best_model = grid_search.best_estimator_

# Save the trained model
print("\nSaving the trained model...")
joblib.dump(best_model, 'best_lightgbm_model.pkl')
print("Model saved as 'best_lightgbm_model.pkl'")

# Predictions
print("\nPredicting on test data...")
y_pred = best_model.predict(X_test_scaled)

# Reverse log-transform predictions and true values
y_pred = np.expm1(y_pred)
y_test = np.expm1(y_test)

# Model evaluation
print("\nEvaluating model performance...")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:,.2f} AED")
print(f"R² Score: {r2:.4f}")

# Plot actual vs predicted
print("\nGenerating visualizations...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price (AED)')
plt.ylabel('Predicted Price (AED)')
plt.title('Actual vs Predicted Property Prices')
plt.tight_layout()
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Price (AED)')
plt.ylabel('Residuals (AED)')
plt.title('Residual Plot')
plt.tight_layout()
plt.show()

# Feature importance plot
print("\nGenerating feature importance plot...")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

print("\nAnalysis complete!")
