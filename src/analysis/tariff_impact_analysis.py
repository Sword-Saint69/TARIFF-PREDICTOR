import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the model and scaler
model = joblib.load('/Users/gouthamsankar/Linux/maand/best_trade_deficit_model.pkl')
scaler = joblib.load('/Users/gouthamsankar/Linux/maand/scaler.pkl')
processed_data = pd.read_csv('/Users/gouthamsankar/Linux/maand/processed_data.csv')

def predict_with_tariff_change(country_name, tariff_change):
    """
    Predict how changing tariffs would affect the trade deficit for a specific country
    
    Parameters:
    country_name (str): Name of the country to analyze
    tariff_change (float): Amount to add to current tariffs (percentage points)
    
    Returns:
    dict: Results of the prediction
    """
    # Get the country data
    country_data = processed_data[processed_data['Country'] == country_name]
    
    if len(country_data) == 0:
        return {"error": f"Country '{country_name}' not found in dataset"}
    
    # Create a copy for modification
    modified_data = country_data.copy()
    
    # Store original values
    original_tariff = modified_data['Trump Tariffs Alleged'].values[0]
    
    # Modify tariffs (add percentage points)
    modified_data['Trump Tariffs Alleged'] = original_tariff + (tariff_change / 100)
    
    # Recalculate derived features
    modified_data['Tariff_Differential'] = modified_data['Trump Tariffs Alleged'] - modified_data['Trump Response']
    
    # Prepare features
    X_modified = modified_data.drop(['Country', 'US 2024 Deficit'], axis=1)
    
    # Scale features
    X_modified_scaled = scaler.transform(X_modified)
    
    # Make predictions
    original_deficit = country_data['US 2024 Deficit'].values[0]
    predicted_deficit = model.predict(X_modified_scaled)[0]
    
    # Calculate change
    deficit_change = predicted_deficit - original_deficit
    percent_change = (deficit_change / abs(original_deficit)) * 100 if original_deficit != 0 else float('inf')
    
    return {
        "country": country_name,
        "original_tariff": original_tariff * 100,  # Convert back to percentage
        "new_tariff": (original_tariff + (tariff_change / 100)) * 100,  # Convert back to percentage
        "original_deficit": original_deficit,
        "predicted_deficit": predicted_deficit,
        "deficit_change": deficit_change,
        "percent_change": percent_change,
        "tariff_change": tariff_change  # Add this line to include the tariff_change key
    }

def analyze_multiple_countries(countries, tariff_changes):
    """
    Analyze the impact of different tariff changes on multiple countries
    
    Parameters:
    countries (list): List of country names
    tariff_changes (list): List of tariff changes to analyze (percentage points)
    """
    results = []
    
    for country in countries:
        country_results = []
        for change in tariff_changes:
            result = predict_with_tariff_change(country, change)
            if "error" not in result:
                country_results.append(result)
        
        if country_results:
            results.append(country_results)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    for country_results in results:
        country = country_results[0]["country"]
        changes = [r["tariff_change"] for r in country_results]
        deficit_changes = [r["deficit_change"] for r in country_results]
        
        plt.plot([r["new_tariff"] for r in country_results], 
                 [r["predicted_deficit"] for r in country_results], 
                 marker='o', label=country)
    
    plt.xlabel('Tariff Rate (%)')
    plt.ylabel('Predicted Trade Deficit')
    plt.title('Impact of Tariff Changes on Trade Deficit')
    plt.legend()
    plt.grid(True)
    plt.savefig('/Users/gouthamsankar/Linux/maand/tariff_impact.png')
    plt.show()
    
    return results

# Example usage
if __name__ == "__main__":
    # Analyze major trading partners
    countries_to_analyze = ["China", "Mexico", "Canada", "Japan", "European Union"]
    tariff_changes = [-5, 0, 5, 10, 15, 20]  # Percentage points
    
    results = analyze_multiple_countries(countries_to_analyze, tariff_changes)
    
    # Print detailed results
    print("\nDetailed Tariff Impact Analysis:")
    print("================================")
    
    for country_results in results:
        country = country_results[0]["country"]
        print(f"\n{country}:")
        print(f"Original deficit: ${country_results[0]['original_deficit']:,.2f}")
        
        for result in country_results:
            print(f"  Tariff: {result['new_tariff']:.1f}% → Deficit: ${result['predicted_deficit']:,.2f} (Change: {result['percent_change']:+.2f}%)")