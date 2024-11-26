import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import acos, sin, cos


dubaimall_lat = np.radians(25.197344)
dubaimall_lon = np.radians(55.278768)

burj_lat = np.radians(25.197179)
burj_lon = np.radians(55.274154)

marina_lat = np.radians(25.080542)
marina_lon = np.radians(55.140343)

def add_landmark_dist_data(df, row):

    lat = np.radians(df.at[row, 'LATITUDE'])
    lon = np.radians(df.at[row, 'LONGITUDE'])

    # Clip cosine values to [-1,1] to prevent rounding errors
    dubaimall_cos = min(1, max(-1, sin(lat)*sin(dubaimall_lat) + cos(lat)*cos(dubaimall_lat)*cos(dubaimall_lon - lon)))
    burj_cos = min(1, max(-1, sin(lat)*sin(burj_lat) + cos(lat)*cos(burj_lat)*cos(burj_lon - lon)))
    marina_cos = min(1, max(-1, sin(lat)*sin(marina_lat) + cos(lat)*cos(marina_lat)*cos(marina_lon - lon)))

    dubaimall_distance = acos(dubaimall_cos) * 6371
    burj_distance = acos(burj_cos) * 6371
    marina_distance = acos(marina_cos) * 6371

    df.at[row, 'DUBAIMALL_DIST'] = dubaimall_distance
    df.at[row, 'BURJ_DIST'] = burj_distance
    df.at[row, 'MARINA_DIST'] = marina_distance

    
def main():

    print("Started running...")

    df = pd.read_csv('transactions_filtered_v1.csv')
    print(f"Number of rows: {len(df)}")

    print("File read...")

    df['DUBAIMALL_DIST'] = np.nan
    df['BURJ_DIST'] = np.nan
    df['MARINA_DIST'] = np.nan

    num_rows = len(df)

    for r in range(num_rows):
        add_landmark_dist_data(df, r)

    print("New file is being saved...")
    df.to_csv('transactions_with_landmark_distances.csv', index=False)
    print("New file saved successfully!")

if __name__ == "__main__":
    main()