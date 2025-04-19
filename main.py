import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import modules
from src.utils.data_loader import load_data
from src.analysis.basic_analysis import calculate_elasticities, analyze_country_clusters, analyze_prediction_errors
from src.models.model_improvements import test_additional_features, test_nonlinear_models, simulate_time_series_analysis

def main():
    # Create output directories
    os.makedirs('/Users/gouthamsankar/Linux/maand/output/plots', exist_ok=True)
    os.makedirs('/Users/gouthamsankar/Linux/maand/output/reports', exist_ok=True)
    
    # Load data
    print("Loading data...")
    model, scaler, processed_data = load_data()
    
    # Basic analysis
    print("\n--- BASIC ANALYSIS ---")
    
    print("Calculating elasticities...")
    elasticities = calculate_elasticities(model, scaler, processed_data)
    print("\nElasticities (% change in deficit for 1% increase in variable):")
    for feature, elasticity in sorted(elasticities.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"{feature}: {elasticity:.4f}%")
    
    print("\nAnalyzing country clusters...")
    cluster_data, cluster_summary, cluster_examples = analyze_country_clusters(processed_data)
    print("\nCountry Cluster Summary:")
    print(cluster_summary)
    print("\nExample countries in each cluster:")
    for cluster, examples in cluster_examples.items():
        print(f"Cluster {cluster}: {examples}")
    
    print("\nAnalyzing prediction errors...")
    error_analysis, error_correlations = analyze_prediction_errors(model, scaler, processed_data)
    print("\nTop factors correlated with prediction errors:")
    print(error_correlations.head(10))
    
    # Model improvements
    print("\n--- TESTING MODEL IMPROVEMENTS ---")
    
    # Test additional features
    feature_results, best_features, enhanced_data = test_additional_features(processed_data)
    print("\nResults with additional features:")
    for name, metrics in feature_results.items():
        print(f"{name}: RMSE = {metrics['RMSE']:.2f}, R² = {metrics['R²']:.4f}, Features = {metrics['Feature Count']}")
    
    print("\nTop 10 most important features in enhanced model:")
    print(best_features.head(10))
    
    # Test non-linear models
    nonlinear_results, nonlinear_importances = test_nonlinear_models(processed_data)
    print("\nResults with non-linear models:")
    for name, metrics in nonlinear_results.items():
        print(f"{name}: RMSE = {metrics['RMSE']:.2f}, R² = {metrics['R²']:.4f}, CV RMSE = {metrics['CV RMSE']:.2f}")
    
    # If Random Forest performed well, show its feature importances
    if 'Random Forest' in nonlinear_importances:
        print("\nRandom Forest feature importances:")
        print(nonlinear_importances['Random Forest'].head(10))
    
    # Simulate time-series analysis
    ts_data = simulate_time_series_analysis(processed_data)
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
    plt.savefig('/Users/gouthamsankar/Linux/maand/output/plots/model_comparison.png')
    
    print("\nAnalysis complete. Check the generated plots in the output/plots directory.")
    
    # Generate a summary report
    with open('/Users/gouthamsankar/Linux/maand/output/reports/analysis_summary.txt', 'w') as f:
        f.write("# Trade Deficit Analysis Summary\n\n")
        
        f.write("## Elasticity Analysis\n")
        f.write("Feature elasticities (% change in deficit for 1% increase in variable):\n")
        for feature, elasticity in sorted(elasticities.items(), key=lambda x: abs(x[1]), reverse=True):
            f.write(f"- {feature}: {elasticity:.4f}%\n")
        
        f.write("\n## Country Clusters\n")
        f.write(f"Optimal number of clusters: {len(cluster_summary)}\n\n")
        for cluster, examples in cluster_examples.items():
            f.write(f"Cluster {cluster}: {examples}\n")
        
        f.write("\n## Model Performance\n")
        f.write("### Linear Models with Additional Features\n")
        for name, metrics in feature_results.items():
            f.write(f"- {name}: RMSE = {metrics['RMSE']:.2f}, R² = {metrics['R²']:.4f}\n")
        
        f.write("\n### Non-linear Models\n")
        for name, metrics in nonlinear_results.items():
            f.write(f"- {name}: RMSE = {metrics['RMSE']:.2f}, R² = {metrics['R²']:.4f}, CV RMSE = {metrics['CV RMSE']:.2f}\n")
        
        f.write("\n## Conclusion\n")
        best_model = min(nonlinear_results.items(), key=lambda x: x[1]['RMSE'])[0]
        f.write(f"The best performing model is {best_model} based on RMSE.\n")
        f.write("Key findings:\n")
        f.write("1. Tariffs have relatively small impact on trade deficits compared to import/export volumes\n")
        f.write("2. Countries cluster into distinct groups based on their trade patterns\n")
        f.write("3. Non-linear models can capture more complex relationships in the data\n")
    
    print("Summary report generated at: /Users/gouthamsankar/Linux/maand/output/reports/analysis_summary.txt")

if __name__ == "__main__":
    main()