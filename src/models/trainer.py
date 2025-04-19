import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('/Users/gouthamsankar/Linux/maand/Tariff Calculations plus Population.csv', sep=';')

# Data preprocessing
def preprocess_data(df):
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Convert percentage strings to floats
    data['Trump Tariffs Alleged'] = data['Trump Tariffs Alleged'].str.rstrip('%').astype('float') / 100
    data['Trump Response'] = data['Trump Response'].str.rstrip('%').astype('float') / 100
    
    # Handle commas in numeric columns and convert to float
    numeric_cols = ['US 2024 Deficit', 'US 2024 Exports', 'US 2024 Imports (Customs Basis)', 'Population']
    for col in numeric_cols:
        if data[col].dtype == object:
            data[col] = data[col].str.replace(',', '').astype('float')
    
    # Drop rows with missing values or handle them appropriately
    data = data.dropna()
    
    # Create additional features
    data['Export_Import_Ratio'] = data['US 2024 Exports'] / data['US 2024 Imports (Customs Basis)']
    data['Per_Capita_Exports'] = data['US 2024 Exports'] / data['Population']
    data['Per_Capita_Imports'] = data['US 2024 Imports (Customs Basis)'] / data['Population']
    data['Tariff_Differential'] = data['Trump Tariffs Alleged'] - data['Trump Response']
    
    return data

# Preprocess the data
try:
    processed_data = preprocess_data(df)
    print(f"Processed data shape: {processed_data.shape}")
except Exception as e:
    print(f"Error in preprocessing: {e}")
    # Fallback to simpler preprocessing if needed
    processed_data = df.dropna()
    print("Using simplified preprocessing due to error")

# Define features and target
X = processed_data.drop(['Country', 'US 2024 Deficit'], axis=1)
y = processed_data['US 2024 Deficit']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to evaluate and compare models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    return model, y_pred, rmse, r2

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    model_fitted, predictions, rmse, r2 = evaluate_model(
        model, X_train_scaled, X_test_scaled, y_train, y_test, name
    )
    results[name] = {
        'model': model_fitted,
        'predictions': predictions,
        'rmse': rmse,
        'r2': r2
    }

# Find the best model
best_model_name = min(results, key=lambda x: results[x]['rmse'])
print(f"\nBest model: {best_model_name} with RMSE: {results[best_model_name]['rmse']:.2f}")

# Feature importance for the best model (if applicable)
if best_model_name in ['Random Forest', 'XGBoost']:
    best_model = results[best_model_name]['model']
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title(f'Top 10 Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.savefig('/Users/gouthamsankar/Linux/maand/feature_importance.png')
    print("Feature importance plot saved to 'feature_importance.png'")

# Save the best model
import joblib
joblib.dump(results[best_model_name]['model'], f'/Users/gouthamsankar/Linux/maand/best_trade_deficit_model.pkl')
print(f"Best model saved to 'best_trade_deficit_model.pkl'")

# Function to make predictions for new countries
def predict_deficit(model, country_data, scaler):
    # Preprocess the input data
    processed_input = preprocess_data(country_data)
    X_input = processed_input.drop(['Country', 'US 2024 Deficit'], axis=1)
    
    # Scale the input
    X_input_scaled = scaler.transform(X_input)
    
    # Make prediction
    prediction = model.predict(X_input_scaled)
    
    return prediction[0]

print("\nModel training complete. You can now use the best model to predict trade deficits.")

# Hyperparameter tuning example for XGBoost
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

grid_search = GridSearchCV(
    estimator=models['XGBoost'], 
    param_grid=param_grid, 
    cv=3, 
    scoring='neg_mean_squared_error',
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_