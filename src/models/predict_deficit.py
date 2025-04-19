import pandas as pd
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('/Users/gouthamsankar/Linux/maand/best_trade_deficit_model.pkl')
scaler = joblib.load('/Users/gouthamsankar/Linux/maand/scaler.pkl')

def preprocess_data(data):
    # Convert percentage strings to floats if needed
    if isinstance(data['Trump Tariffs Alleged'], str):
        data['Trump Tariffs Alleged'] = data['Trump Tariffs Alleged'].str.rstrip('%').astype('float') / 100
    if isinstance(data['Trump Response'], str):
        data['Trump Response'] = data['Trump Response'].str.rstrip('%').astype('float') / 100
    
    # Create additional features
    data['Export_Import_Ratio'] = data['US 2024 Exports'] / data['US 2024 Imports (Customs Basis)']
    data['Per_Capita_Exports'] = data['US 2024 Exports'] / data['Population']
    data['Per_Capita_Imports'] = data['US 2024 Imports (Customs Basis)'] / data['Population']
    data['Tariff_Differential'] = data['Trump Tariffs Alleged'] - data['Trump Response']
    
    return data

def predict_trade_deficit(country_name, exports, imports, tariffs_alleged, response_tariffs, population):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Country': [country_name],
        'US 2024 Exports': [exports],
        'US 2024 Imports (Customs Basis)': [imports],
        'Trump Tariffs Alleged': [tariffs_alleged],
        'Trump Response': [response_tariffs],
        'Population': [population]
    })
    
    # Preprocess the input data
    processed_input = preprocess_data(input_data)
    
    # Extract features (excluding Country)
    X_input = processed_input.drop(['Country'], axis=1)
    
    # Scale the features
    X_input_scaled = scaler.transform(X_input)
    
    # Make prediction
    prediction = model.predict(X_input_scaled)
    
    return prediction[0]

# Example usage
if __name__ == "__main__":
    # Example: predict for a new country
    predicted_deficit = predict_trade_deficit(
        country_name="Example Country",
        exports=50000,
        imports=75000,
        tariffs_alleged=0.10,  # 10%
        response_tariffs=0.05,  # 5%
        population=10000000
    )
    
    print(f"Predicted Trade Deficit: ${predicted_deficit:,.2f}")
    
    # Interactive mode
    print("\nTrade Deficit Predictor")
    print("----------------------")
    
    while True:
        try:
            country = input("\nEnter country name (or 'quit' to exit): ")
            if country.lower() == 'quit':
                break
                
            exports = float(input("Enter US exports (in USD): "))
            imports = float(input("Enter US imports (in USD): "))
            tariffs = float(input("Enter Trump tariffs alleged (%): ")) / 100
            response = float(input("Enter response tariffs (%): ")) / 100
            pop = float(input("Enter population: "))
            
            deficit = predict_trade_deficit(country, exports, imports, tariffs, response, pop)
            print(f"\nPredicted trade deficit for {country}: ${deficit:,.2f}")
            
        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except Exception as e:
            print(f"Error: {e}")