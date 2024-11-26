import pandas as pd
import numpy as np
from scipy import stats

def eliminate_outliers(df, z_thresh=3):
    # Remove rows with NaN values
    df = df.dropna()

    # Calculate Z-scores for each numeric column
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))

    # Filter out rows with Z-scores above the threshold
    df = df[(z_scores < z_thresh).all(axis=1)]

    return df

def clean_data(df):
    # Eliminate outliers
    df = eliminate_outliers(df)

    # Additional cleaning steps can be added here if needed

    return df

def main():
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('transactions_with_landmark_distances.csv')
    print(f"Loaded {len(df)} transactions")

    # Clean data
    print("Cleaning data...")
    df = clean_data(df)
    print(f"Data cleaned. Remaining transactions: {len(df)}")

    # Save cleaned dataset
    df.to_csv('transactions_cleaned.csv', index=False)
    print("Cleaned dataset saved as 'transactions_cleaned.csv'")

if __name__ == "__main__":
    main()
