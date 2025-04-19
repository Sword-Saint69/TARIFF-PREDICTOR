import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model, scaler and data
model = joblib.load('/Users/gouthamsankar/Linux/maand/best_trade_deficit_model.pkl')
scaler = joblib.load('/Users/gouthamsankar/Linux/maand/scaler.pkl')
processed_data = pd.read_csv('/Users/gouthamsankar/Linux/maand/processed_data.csv')

# 1. Analyze elasticity - how much does deficit change with 1% change in each variable?
def calculate_elasticities():
    elasticities = {}
    
    # Get average values for all features
    avg_data = processed_data.select_dtypes(include=[np.number]).mean().to_dict()
    base_data = pd.DataFrame([avg_data])
    
    # Calculate base prediction
    X_base = base_data.drop(['US 2024 Deficit'], axis=1)
    X_base_scaled = scaler.transform(X_base)
    base_prediction = model.predict(X_base_scaled)[0]
    
    # Calculate elasticity for each feature
    for feature in X_base.columns:
        # Skip derived features
        if feature in ['Export_Import_Ratio', 'Per_Capita_Exports', 'Per_Capita_Imports', 'Tariff_Differential']:
            continue
            
        # Create modified data with 1% increase
        modified_data = base_data.copy()
        modified_data[feature] = modified_data[feature] * 1.01
        
        # Recalculate derived features if needed
        if feature in ['US 2024 Exports', 'US 2024 Imports (Customs Basis)', 'Population']:
            modified_data['Export_Import_Ratio'] = modified_data['US 2024 Exports'] / modified_data['US 2024 Imports (Customs Basis)']
            modified_data['Per_Capita_Exports'] = modified_data['US 2024 Exports'] / modified_data['Population']
            modified_data['Per_Capita_Imports'] = modified_data['US 2024 Imports (Customs Basis)'] / modified_data['Population']
        
        if feature in ['Trump Tariffs Alleged', 'Trump Response']:
            modified_data['Tariff_Differential'] = modified_data['Trump Tariffs Alleged'] - modified_data['Trump Response']
        
        # Make prediction
        X_mod = modified_data.drop(['US 2024 Deficit'], axis=1)
        X_mod_scaled = scaler.transform(X_mod)
        mod_prediction = model.predict(X_mod_scaled)[0]
        
        # Calculate percent change in prediction
        pct_change = (mod_prediction - base_prediction) / abs(base_prediction) * 100
        
        # Store elasticity
        elasticities[feature] = pct_change
    
    return elasticities

# 2. Analyze country clusters - group countries by trade patterns
def analyze_country_clusters():
    # Select relevant features
    features = ['US 2024 Exports', 'US 2024 Imports (Customs Basis)', 
                'Trump Tariffs Alleged', 'Trump Response', 'Population',
                'Export_Import_Ratio']
    
    # Prepare data for clustering
    cluster_data = processed_data[['Country'] + features].copy()
    
    # Normalize data for clustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    X_cluster = cluster_data[features]
    X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
    
    # Find optimal number of clusters
    from sklearn.metrics import silhouette_score
    
    silhouette_scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_cluster_scaled)
        silhouette_scores.append(silhouette_score(X_cluster_scaled, cluster_labels))
    
    optimal_k = np.argmax(silhouette_scores) + 2
    
    # Perform clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_data['Cluster'] = kmeans.fit_predict(X_cluster_scaled)
    
    # Analyze clusters
    cluster_summary = cluster_data.groupby('Cluster').agg({
        'Country': 'count',
        'US 2024 Exports': 'mean',
        'US 2024 Imports (Customs Basis)': 'mean',
        'Trump Tariffs Alleged': 'mean',
        'Trump Response': 'mean',
        'Export_Import_Ratio': 'mean'
    }).rename(columns={'Country': 'Count'})
    
    # Get example countries for each cluster
    cluster_examples = {}
    for cluster in cluster_data['Cluster'].unique():
        countries = cluster_data[cluster_data['Cluster'] == cluster]['Country'].values[:5]
        cluster_examples[cluster] = ', '.join(countries)
    
    return cluster_data, cluster_summary, cluster_examples

# 3. Analyze prediction errors by country characteristics
def analyze_prediction_errors():
    # Make predictions on all data
    X = processed_data.drop(['Country', 'US 2024 Deficit'], axis=1)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    
    # Calculate errors
    error_data = processed_data[['Country', 'US 2024 Deficit']].copy()
    error_data['Predicted'] = predictions
    error_data['Error'] = error_data['US 2024 Deficit'] - error_data['Predicted']
    error_data['Abs_Error'] = error_data['Error'].abs()
    error_data['Pct_Error'] = (error_data['Error'] / error_data['US 2024 Deficit'].abs()) * 100
    
    # Join with original features
    error_analysis = pd.merge(error_data, processed_data, on=['Country', 'US 2024 Deficit'])
    
    # Analyze correlations between errors and features - only use numeric columns
    numeric_error_data = error_analysis.select_dtypes(include=[np.number])
    error_correlations = numeric_error_data.corr()['Abs_Error'].sort_values(ascending=False)
    
    return error_analysis, error_correlations

# 4. Test additional features (GDP, industry-specific data)
def test_additional_features():
    """
    Test how adding GDP and industry-specific features might improve the model.
    This is a simulation since we don't have the actual data.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
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

# 5. Explore non-linear models
def test_nonlinear_models():
    """
    Test various non-linear models to see if they capture complex relationships better.
    """
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    
    print("\nTesting non-linear models...")
    
    # Prepare data
    X = processed_data.drop(['Country', 'US 2024 Deficit'], axis=1)
    y = processed_data['US 2024 Deficit']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
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

# 6. Time-series analysis (simulated)
def simulate_time_series_analysis():
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
    plt.savefig('/Users/gouthamsankar/Linux/maand/time_series_forecast.png')
    
    return ts_data

# Run the original analyses
print("Calculating elasticities...")
elasticities = calculate_elasticities()
print("\nElasticities (% change in deficit for 1% increase in variable):")
for feature, elasticity in sorted(elasticities.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"{feature}: {elasticity:.4f}%")

print("\nAnalyzing country clusters...")
cluster_data, cluster_summary, cluster_examples = analyze_country_clusters()
print("\nCountry Cluster Summary:")
print(cluster_summary)
print("\nExample countries in each cluster:")
for cluster, examples in cluster_examples.items():
    print(f"Cluster {cluster}: {examples}")

print("\nAnalyzing prediction errors...")
error_analysis, error_correlations = analyze_prediction_errors()
print("\nTop factors correlated with prediction errors:")
print(error_correlations.head(10))

# Create visualizations
plt.figure(figsize=(10, 6))
plt.bar(elasticities.keys(), elasticities.values())
plt.xticks(rotation=45, ha='right')
plt.title('Trade Deficit Elasticities')
plt.ylabel('% Change in Deficit for 1% Increase in Variable')
plt.tight_layout()
plt.savefig('/Users/gouthamsankar/Linux/maand/elasticities.png')

# Plot clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=cluster_data, 
    x='US 2024 Exports', 
    y='US 2024 Imports (Customs Basis)',
    hue='Cluster',
    size='Population',
    sizes=(20, 500),
    alpha=0.7
)
plt.title('Country Clusters by Trade Patterns')
plt.xscale('log')
plt.yscale('log')
for i, row in cluster_data.iterrows():
    plt.annotate(row['Country'], (row['US 2024 Exports'], row['US 2024 Imports (Customs Basis)']), 
                 fontsize=8, alpha=0.7)
plt.tight_layout()
plt.savefig('/Users/gouthamsankar/Linux/maand/country_clusters.png')

print("\nAnalysis complete. Check the generated plots for more insights.")

# Run new analyses
print("\n--- TESTING MODEL IMPROVEMENTS ---")

# Test additional features
feature_results, best_features, enhanced_data = test_additional_features()
print("\nResults with additional features:")
for name, metrics in feature_results.items():
    print(f"{name}: RMSE = {metrics['RMSE']:.2f}, R² = {metrics['R²']:.4f}, Features = {metrics['Feature Count']}")

print("\nTop 10 most important features in enhanced model:")
print(best_features.head(10))

# Test non-linear models
nonlinear_results, nonlinear_importances = test_nonlinear_models()
print("\nResults with non-linear models:")
for name, metrics in nonlinear_results.items():
    print(f"{name}: RMSE = {metrics['RMSE']:.2f}, R² = {metrics['R²']:.4f}, CV RMSE = {metrics['CV RMSE']:.2f}")

# If Random Forest performed well, show its feature importances
if 'Random Forest' in nonlinear_importances:
    print("\nRandom Forest feature importances:")
    print(nonlinear_importances['Random Forest'].head(10))

# Simulate time-series analysis
ts_data = simulate_time_series_analysis()
print("\nTime-series analysis complete. Check the generated time_series_forecast.png for visualization.")

# Create a visualization comparing all models
plt.figure(figsize=(12, 6))
model_names = list(nonlinear_results.keys()) + list(feature_results.keys())
rmse_values = [metrics['RMSE'] for metrics in nonlinear_results.values()] + [metrics['RMSE'] for metrics in feature_results.values()]
r2_values = [metrics['R²'] for metrics in nonlinear_results.values()] + [metrics['R²'] for metrics in feature_results.values()]

x = np.arange(len(model_names))
width = 0.35

fig, ax1 = plt.subplots(figsize=(14, 8))
ax2 = ax1.twinx()

bars1 = ax1.bar(x - width/2, rmse_values, width, label='RMSE', color='skyblue')
bars2 = ax2.bar(x + width/2, r2_values, width, label='R²', color='lightgreen')

ax1.set_xlabel('Model')
ax1.set_ylabel('RMSE', color='blue')
ax2.set_ylabel('R²', color='green')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=45, ha='right')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title('Model Comparison')
plt.tight_layout()
plt.savefig('/Users/gouthamsankar/Linux/maand/model_comparison.png')

print("\nModel improvement analysis complete. Check the generated plots for more insights.")