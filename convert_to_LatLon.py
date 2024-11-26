import pandas as pd
import googlemaps
from datetime import datetime
import csv

# Initialize Google Maps client
gmaps = googlemaps.Client(key='GOOGLE_MAPS_API_KEY')

# Input and output file paths
input_file = 'filtered_transactions_20241119003217.csv'
output_file = f'filtered_transactions_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'

# Read the input CSV file
df = pd.read_csv(input_file)

# Get unique addresses
unique_addresses = set(df['FULL_ADDRESS'].unique())
print(f"\nFound {len(unique_addresses)} unique addresses to geocode\n")

# Create a dictionary to store geocoding results
geocode_dict = {}

# Geocode each unique address
for i, address in enumerate(unique_addresses, 1):
    print(f"\nGeocoding address {i}/{len(unique_addresses)}:")
    print(f"Address: {address}")
    try:
        # Make API call
        result = gmaps.geocode(address)
        if result:
            location = result[0]['geometry']['location']
            lat = location['lat']
            lon = location['lng']
            geocode_dict[address] = (lat, lon)
            print(f"Success! Coordinates: ({lat}, {lon})")
        else:
            geocode_dict[address] = (None, None)
            print("Failed - No results found")
            
    except Exception as e:
        print(f"Error geocoding address: {address}")
        print(f"Error: {e}")
        geocode_dict[address] = (None, None)

# Create new columns using the geocoding results
df['LATITUDE'] = df['FULL_ADDRESS'].map(lambda x: geocode_dict.get(x, (None, None))[0])
df['LONGITUDE'] = df['FULL_ADDRESS'].map(lambda x: geocode_dict.get(x, (None, None))[1])

# Save the updated dataset
df.to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL)
print(f"\nUpdated dataset saved as {output_file}")
