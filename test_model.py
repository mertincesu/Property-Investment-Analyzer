import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load just the models
xgb_model = joblib.load('best_xgboost_model.pkl')
lgb_model = joblib.load('best_lightgbm_model.pkl')

# Create and fit the encoders with known values
rooms_encoder = LabelEncoder()
# Assuming these are the possible room values in your training data
rooms_encoder.fit(['0', '1', '2', '3', '4', '5', '6', '7', '8'])

# Define the features in the same order as during training
features = ['INSTANCE_DATE', 'PROCEDURE_AREA', 'ROOMS_EN', 'TOTAL_BUYER', 'TOTAL_SELLER', 'LATITUDE', 'LONGITUDE', 'BURJ_DIST', 'DUBAIMALL_DIST', 'MARINA_DIST']

def get_user_input_and_predict():
    # Get user input for each feature
    user_input = {}
    user_input['INSTANCE_DATE'] = input("Enter the instance date (YYYY-MM-DD): ")
    user_input['PROCEDURE_AREA'] = float(input("Enter the procedure area: "))
    user_input['ROOMS_EN'] = input("Enter the number of rooms: ")
    user_input['TOTAL_BUYER'] = int(input("Enter the total number of buyers: "))
    user_input['TOTAL_SELLER'] = int(input("Enter the total number of sellers: "))
    user_input['LATITUDE'] = float(input("Enter the latitude: "))
    user_input['LONGITUDE'] = float(input("Enter the longitude: "))
    user_input['BURJ_DIST'] = float(input("Enter the distance to Burj Khalifa: "))
    user_input['DUBAIMALL_DIST'] = float(input("Enter the distance to Dubai Mall: "))
    user_input['MARINA_DIST'] = float(input("Enter the distance to Marina: "))

    # Convert to DataFrame first
    user_df = pd.DataFrame([user_input])

    # Convert INSTANCE_DATE to numeric
    user_df['INSTANCE_DATE'] = pd.to_datetime(user_df['INSTANCE_DATE'])
    user_df['INSTANCE_DATE'] = (user_df['INSTANCE_DATE'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')

    # Encode ROOMS_EN
    user_df['ROOMS_EN'] = rooms_encoder.transform(user_df['ROOMS_EN'])

    # Create a new scaler and fit it with reasonable ranges based on your data
    scaler = StandardScaler()
    # Create a sample dataset with expected ranges to fit the scaler
    sample_data = pd.DataFrame({
        'INSTANCE_DATE': [user_df['INSTANCE_DATE'].iloc[0], user_df['INSTANCE_DATE'].iloc[0] + 365],
        'PROCEDURE_AREA': [50, 500],
        'ROOMS_EN': [0, 8],
        'TOTAL_BUYER': [1, 5],
        'TOTAL_SELLER': [1, 5],
        'LATITUDE': [25.0, 25.3],
        'LONGITUDE': [55.0, 55.5],
        'BURJ_DIST': [0, 30],
        'DUBAIMALL_DIST': [0, 30],
        'MARINA_DIST': [0, 30]
    })
    scaler.fit(sample_data)

    # Scale the features
    user_df_scaled = scaler.transform(user_df[features])

    # Make predictions
    xgb_prediction = xgb_model.predict(user_df_scaled)
    lgb_prediction = lgb_model.predict(user_df_scaled)

    # Reverse log-transform the predictions (if your target was log-transformed during training)
    xgb_prediction = np.expm1(xgb_prediction)
    lgb_prediction = np.expm1(lgb_prediction)

    print(f"\nPredictions:")
    print(f"XGB - Predicted Property Price (AED): {xgb_prediction[0]:,.2f}")
    print(f"LGB - Predicted Property Price (AED): {lgb_prediction[0]:,.2f}")

# Run the function
if __name__ == "__main__":
    get_user_input_and_predict()
