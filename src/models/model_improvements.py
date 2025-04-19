import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def test_additional_features(processed_data, output_dir='/Users/gouthamsankar/Linux/maand/output/plots'):
    """
    Test how adding GDP and industry-specific features might improve the model.
    This is a simulation since we don't have the actual data.
    """
    print("\nTesting additional features...")
    
    # Create a copy of the data for simulation
    enhanced_data = processed_data.copy()
    
    # Simulate GDP data (in reality, you would load this from an external source)
    np.random.seed(42)
    enhanced_data['GDP'] = enhanced_data['Population'] * np.random.normal(50000, 20000, size=len(enhanced_data))
    enhanced_data['GDP_per_capita'] = enhanced_data['GDP'] / enhanced_data['Population']
    
    # Simulate industry-specific trade data
    industries = ['Manufacturing', 'Agriculture', 'Services', 'Technology']
    for industry in industries:
        # Simulate industry exports as a percentage of total exports
        enhanced_data[f'{industry}_Export_Pct'] = np.random.uniform(0.1, 0.4, size=len(enhanced_data))
        # Simulate industry imports as a percentage of total imports
        enhanced_data[f'{industry}_Import_Pct'] = np.random.uniform(0.1, 0.4, size=len(enhanced_data))
        
        # Calculate actual values
        enhanced_data[f'{industry}_Exports'] = enhanced_data['US 2024 Exports'] * enhanced_data[f'{industry}_Export_Pct']
        enhanced_data[f'{industry}_Imports'] = enhanced_data['US 2024 Imports (Customs Basis)'] * enhanced_data[f'{industry}_Import_Pct']
        
        # Calculate industry-specific trade balance
        enhanced_data[f'{industry}_Balance'] = enhanced_data[f'{industry}_Exports'] - enhanced_data[f'{industry}_Imports']
    
    # Define feature sets to test
    feature_sets = {
        'Original': processed_data.drop(['Country', 'US 2024 Deficit'], axis=1).columns.tolist(),
        'With GDP': processed_data.drop(['Country', 'US 2024 Deficit'], axis=1).columns.tolist() + ['GDP', 'GDP_per_capita'],
        'With Industry Data': processed_data.drop(['Country', 'US 2024 Deficit'], axis=1).columns.tolist() + 
                            [f'{industry}_Balance' for industry in industries],
        'All Features': processed_data.drop(['Country', 'US 2024 Deficit'], axis=1).columns.tolist() + 
                       ['GDP', 'GDP_per_capita'] + 
                       [f'{industry}_Balance' for industry in industries]
    }
    
    # Test each feature set
    results = {}
    for name, features in feature_sets.items():
        # Prepare data
        X = enhanced_data[features]
        y = enhanced_data['US 2024 Deficit']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        # Evaluate
        y_pred = lr.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'RMSE': rmse,
            'R²': r2,
            'Feature Count': len(features)
        }
        
        # Feature importance for the best model
        if name == 'All Features':
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Coefficient': lr.coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
    
    return results, feature_importance, enhanced_data

def test_nonlinear_models(processed_data, output_dir='/Users/gouthamsankar/Linux/maand/output/plots'):
    """
    Test various non-linear models to see if they capture complex relationships better.
    """
    print("\nTesting non-linear models...")
    
    # Prepare data
    X = processed_data.drop(['Country', 'US 2024 Deficit'], axis=1)
    y = processed_data['US 2024 Deficit']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to test
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Support Vector Machine': SVR(kernel='rbf'),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
    }
    
    # Test each model
    results = {}
    feature_importances = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
        
        results[name] = {
            'RMSE': rmse,
            'R²': r2,
            'CV RMSE': -cv_scores.mean()
        }
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importances[name] = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
    
    return results, feature_importances

def simulate_time_series_analysis(processed_data, output_dir='/Users/gouthamsankar/Linux/maand/output/plots'):
    """
    Simulate time-series analysis to see how trade deficits might evolve.
    In reality, you would use historical data.
    """
    print("\nSimulating time-series analysis...")
    
    # Select a few major trading partners
    countries = ['China', 'Mexico', 'Canada', 'Japan', 'European Union']
    country_data = processed_data[processed_data['Country'].isin(countries)].copy()
    
    # Simulate historical data (5 years back)
    years = range(2019, 2025)
    historical_data = []
    
    for country in country_data['Country']:
        country_row = country_data[country_data['Country'] == country].iloc[0]
        
        for year in years:
            # Create a copy of the country data
            year_data = country_row.copy()
            year_data['Year'] = year
            
            # Adjust values based on year (simulating historical trends)
            if year < 2024:
                # Simulate growth rates
                export_growth = np.random.uniform(0.02, 0.05)  # 2-5% annual growth
                import_growth = np.random.uniform(0.03, 0.06)  # 3-6% annual growth
                
                # Calculate values for the year
                years_back = 2024 - year
                year_data['US 2024 Exports'] = year_data['US 2024 Exports'] / ((1 + export_growth) ** years_back)
                year_data['US 2024 Imports (Customs Basis)'] = year_data['US 2024 Imports (Customs Basis)'] / ((1 + import_growth) ** years_back)
                
                # Recalculate deficit
                year_data['US 2024 Deficit'] = year_data['US 2024 Exports'] - year_data['US 2024 Imports (Customs Basis)']
                
                # Adjust tariffs (simulating policy changes)
                year_data['Trump Tariffs Alleged'] = max(0, year_data['Trump Tariffs Alleged'] - 0.01 * years_back)
                year_data['Trump Response'] = max(0, year_data['Trump Response'] - 0.005 * years_back)
            
            historical_data.append(year_data)
    
    # Create DataFrame
    ts_data = pd.DataFrame(historical_data)
    
    # Forecast future values (2025-2027)
    future_years = range(2025, 2028)
    forecast_data = []
    
    for country in countries:
        country_ts_data = ts_data[ts_data['Country'] == country]
        
        # Get the latest data (2024)
        latest_data = country_ts_data[country_ts_data['Year'] == 2024].iloc[0]
        
        for year in future_years:
            # Create a copy of the latest data
            year_data = latest_data.copy()
            year_data['Year'] = year
            
            # Project future values
            years_forward = year - 2024
            
            # Simulate growth rates
            export_growth = np.random.uniform(0.02, 0.05)  # 2-5% annual growth
            import_growth = np.random.uniform(0.03, 0.06)  # 3-6% annual growth
            
            # Calculate values for the year
            year_data['US 2024 Exports'] = year_data['US 2024 Exports'] * ((1 + export_growth) ** years_forward)
            year_data['US 2024 Imports (Customs Basis)'] = year_data['US 2024 Imports (Customs Basis)'] * ((1 + import_growth) ** years_forward)
            
            # Recalculate deficit
            year_data['US 2024 Deficit'] = year_data['US 2024 Exports'] - year_data['US 2024 Imports (Customs Basis)']
            
            forecast_data.append(year_data)
    
    # Add forecast to time series data
    forecast_df = pd.DataFrame(forecast_data)
    ts_data = pd.concat([ts_data, forecast_df])
    
    # Plot time series for each country
    plt.figure(figsize=(14, 8))
    
    for country in countries:
        country_ts = ts_data[ts_data['Country'] == country]
        plt.plot(country_ts['Year'], country_ts['US 2024 Deficit'], marker='o', label=country)
    
    plt.axvline(x=2024, color='gray', linestyle='--', label='Current Year')
    plt.axvspan(2025, 2027, alpha=0.2, color='gray', label='Forecast')
    
    plt.xlabel('Year')
    plt.ylabel('Trade Deficit')
    plt.title('Historical and Projected Trade Deficits')
    plt.legend()
    plt.grid(True)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, 'time_series_forecast.png'))
    
    return ts_data