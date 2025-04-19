import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load the raw data
print("Loading raw data...")
raw_data = pd.read_csv('/Users/gouthamsankar/Linux/maand/Tariff Calculations plus Population.csv', sep=';')
print(f"Raw data loaded with shape: {raw_data.shape}")

# Clean and process the data
print("Processing data...")

# Handle missing values
raw_data = raw_data.dropna(subset=['US 2024 Deficit', 'US 2024 Exports', 'US 2024 Imports (Customs Basis)'])

# Convert numeric columns from strings to float (handling commas in numbers)
numeric_columns = ['US 2024 Deficit', 'US 2024 Exports', 'US 2024 Imports (Customs Basis)']
for col in numeric_columns:
    raw_data[col] = raw_data[col].str.replace(',', '').astype(float)

# Convert percentage strings to floats
raw_data['Trump Tariffs Alleged'] = raw_data['Trump Tariffs Alleged'].str.rstrip('%').astype('float') / 100
raw_data['Trump Response'] = raw_data['Trump Response'].str.rstrip('%').astype('float') / 100

# Handle missing population data
raw_data = raw_data.dropna(subset=['Population'])
# Convert Population to numeric
raw_data['Population'] = pd.to_numeric(raw_data['Population'], errors='coerce')

# Create additional features
raw_data['Export_Import_Ratio'] = raw_data['US 2024 Exports'] / raw_data['US 2024 Imports (Customs Basis)']
raw_data['Per_Capita_Exports'] = raw_data['US 2024 Exports'] / raw_data['Population']
raw_data['Per_Capita_Imports'] = raw_data['US 2024 Imports (Customs Basis)'] / raw_data['Population']
raw_data['Tariff_Differential'] = raw_data['Trump Tariffs Alleged'] - raw_data['Trump Response']

# Save processed data
processed_data = raw_data.copy()
processed_data.to_csv('/Users/gouthamsankar/Linux/maand/processed_data.csv', index=False)
print(f"Processed data saved with shape: {processed_data.shape}")

# Define features and target
X = processed_data.drop(['Country', 'US 2024 Deficit'], axis=1)
y = processed_data['US 2024 Deficit']

# Create and fit the scaler
print("Creating and fitting scaler...")
scaler = StandardScaler()
scaler.fit(X)

# Save the scaler
joblib.dump(scaler, '/Users/gouthamsankar/Linux/maand/scaler.pkl')
print("Scaler saved successfully!")