import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# Load the model and processed data
model = joblib.load('/Users/gouthamsankar/Linux/maand/best_trade_deficit_model.pkl')
processed_data = pd.read_csv('/Users/gouthamsankar/Linux/maand/processed_data.csv')

# Define features and target
X = processed_data.drop(['Country', 'US 2024 Deficit'], axis=1)
y = processed_data['US 2024 Deficit']

# Load the scaler
scaler = joblib.load('/Users/gouthamsankar/Linux/maand/scaler.pkl')
X_scaled = scaler.transform(X)

# Make predictions on the entire dataset
predictions = model.predict(X_scaled)

# Calculate metrics
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y, predictions)

print(f"Performance on entire dataset:")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# Create a DataFrame with actual vs predicted values
results_df = pd.DataFrame({
    'Country': processed_data['Country'],
    'Actual': y,
    'Predicted': predictions,
    'Difference': y - predictions
})

# Sort by absolute difference
results_df['Abs_Difference'] = results_df['Difference'].abs()
results_df = results_df.sort_values('Abs_Difference', ascending=False)

print("\nTop 10 countries with largest prediction errors:")
print(results_df.head(10)[['Country', 'Actual', 'Predicted', 'Difference']])

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y, predictions, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Trade Deficit')
plt.ylabel('Predicted Trade Deficit')
plt.title('Actual vs Predicted Trade Deficit')
plt.savefig('/Users/gouthamsankar/Linux/maand/actual_vs_predicted.png')

# Check for feature importance (if Linear Regression)
if hasattr(model, 'coef_'):
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print("\nFeature Coefficients:")
    print(coefficients)
    
    # Plot feature coefficients
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Coefficient', y='Feature', data=coefficients)
    plt.title('Feature Coefficients')
    plt.tight_layout()
    plt.savefig('/Users/gouthamsankar/Linux/maand/feature_coefficients.png')

# Check for potential data leakage
print("\nChecking for potential data leakage...")
# Only include numeric columns in correlation calculation
numeric_data = processed_data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
target_correlations = correlation_matrix['US 2024 Deficit'].sort_values(ascending=False)
print("\nFeature correlations with target:")
print(target_correlations)

# Save the correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('/Users/gouthamsankar/Linux/maand/correlation_matrix.png')

print("\nAnalysis complete. Check the generated plots for more insights.")